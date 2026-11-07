#!/bin/bash
# Queue the whole pipeline for one or more datasets with a single command. Every job waits for the jobs it needs
# (sbatch --dependency=afterok), so you can submit everything now and walk away.
#
#   bash datasets/run_all.sh nq hotpotqa fever
#   bash datasets/run_all.sh msmarco-v2-all          # msmarco-v2, then the 25m/50m/100m subsamples once it's embedded
#   STAGES="bins pacmann" bash datasets/run_all.sh nq      # only some stages (data already prepared)
#   DRY_RUN=1 bash datasets/run_all.sh nq            # print the sbatch commands, submit nothing
#
# Stages, in order (default: all):
#   prep     download + convert                 datasets/prep_datasets.slurm
#   embed    e5 + PCA embeddings                 datasets/embed.slurm (msmarco-v2: PCA -> shards -> merge)
#   tree-db  Lucene index + tree artifacts       datasets/tree_artifacts.slurm          (after embed)
#   bins     bins grid, high + low tiers         run_bins_high.slurm, run_bins_low.slurm (after embed)
#   pacmann  pacmann grid                        run_pacmann.slurm                       (after embed)
#   tree     tree grid                           run_tree_pir_old.slurm                  (after tree-db)
# A stage you leave out is assumed to be done already. If a job fails, the jobs that depend on it stay pending
# with reason DependencyNeverSatisfied: fix it, scancel them and re-run with STAGES set to the remaining stages.
#
# Run from the repo root after build.sh (pir_app must exist). Array sizes are worked out with each script's --count.
set -euo pipefail
cd "$(dirname "$0")/.."

# One dataset folder for every stage. The prep/embed scripts default to ../datasets but the run scripts to
# ../../datasets, so pin both to the same place here (override with DATASETS_ROOT=...).
export DATASETS_ROOT="${DATASETS_ROOT:-../../datasets}"
export DATASET_DIR="$DATASETS_ROOT"

STAGES="${STAGES:-prep embed tree-db bins pacmann tree}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p logs

submit() {   # submit <dependency job ids, space separated, may be empty> <sbatch args...> -> prints job id
    local deps="$1"; shift
    local dep_arg=()
    deps="$(echo $deps)"
    [[ -n "$deps" ]] && dep_arg=(--dependency="afterok:${deps// /:}" --kill-on-invalid-dep=no)
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "  sbatch ${dep_arg[*]} $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "${dep_arg[@]}" "$@"
    fi
}

count_of() {   # count_of <env assignment> <script> -> number of array tasks
    env "$1" SLURM_SUBMIT_DIR="$PWD" bash "$2" --count 2>/dev/null | grep -oE '^[0-9]+' || echo 0
}

is_big() { [[ "$1" == msmarco-v2* ]]; }

# submit_dataset <name> [extra dependency] ; sets EMBED_JOB to the job the dataset's embeddings come from
submit_dataset() {
    local d="$1" extra="${2:-}" prep="" embed="" treedb=""
    echo "== $d"

    if has prep; then
        prep=$(submit "$extra" --export=ALL,DATASET="$d" --job-name="prep_$d" datasets/prep_datasets.slurm)
        echo "  prep     $prep"
    fi

    embed="$prep $extra"
    if has embed && [[ ! "$d" =~ ^msmarco-v2-[0-9]+m$ ]]; then   # subsamples copy the full set's embeddings
        if [[ "$d" == "msmarco-v2" ]]; then
            local n=${NUM_SHARDS:-24} pca shards
            pca=$(submit "$prep $extra" --export=ALL,DATASET="$d",MODE=pca,NUM_SHARDS=$n --time=0-02:00:00 \
                  --job-name="pca_$d" datasets/embed.slurm)
            shards=$(submit "$pca" --export=ALL,DATASET="$d",MODE=shard,NUM_SHARDS=$n --array=0-$((n - 1)) \
                  --job-name="emb_$d" datasets/embed.slurm)
            embed=$(submit "$shards" --export=ALL,DATASET="$d",MODE=finish,NUM_SHARDS=$n --time=0-06:00:00 \
                  --job-name="merge_$d" datasets/embed.slurm)
            echo "  embed    pca=$pca shards=$shards merge=$embed"
        else
            embed=$(submit "$prep $extra" --export=ALL,DATASET="$d" --job-name="emb_$d" datasets/embed.slurm)
            echo "  embed    $embed"
        fi
    fi
    EMBED_JOB="$embed"

    if has tree-db; then
        local big=()
        is_big "$d" && big=(--partition=bigmem --mem=1000G)
        treedb=$(submit "$embed" --export=ALL,DATASET="$d",WORKERS="${WORKERS:-$(is_big "$d" && echo 12 || echo 16)}" \
                 "${big[@]}" --job-name="treedb_$d" datasets/tree_artifacts.slurm)
        echo "  tree-db  $treedb"
    fi

    if has bins; then
        local tier n
        for tier in high low; do
            n=$(count_of BINS_DATASETS="$d" "run_bins_${tier}.slurm")
            if [[ "$n" -gt 0 ]]; then
                echo "  bins-$tier $(submit "$embed" --export=ALL,BINS_DATASETS="$d" --array=0-$((n - 1)) \
                     --job-name="bins_${tier}_$d" "run_bins_${tier}.slurm") ($n tasks)"
            fi
        done
    fi

    if has pacmann; then
        local n big=()
        n=$(count_of PACMANN_DATASETS="$d" run_pacmann.slurm)
        is_big "$d" && big=(--partition=bigmem --mem=1500G)
        echo "  pacmann  $(submit "$embed" --export=ALL,PACMANN_DATASETS="$d" --array=0-$((n - 1)) "${big[@]}" \
             --job-name="pacmann_$d" run_pacmann.slurm) ($n tasks)"
    fi

    if has tree; then
        local n big=()
        n=$(count_of DATASET="$d" run_tree_pir_old.slurm)
        is_big "$d" && big=(--partition=bigmem --mem=1000G)
        echo "  tree     $(submit "$treedb $embed" --export=ALL,DATASET="$d" --array=0-$((n - 1)) "${big[@]}" \
             --job-name="tree_$d" run_tree_pir_old.slurm) ($n tasks)"
    fi
}

[[ $# -gt 0 ]] || { sed -n 2,22p "$0"; exit 1; }
[[ -x ./pir_app || -n "${DRY_RUN:-}" ]] || echo "WARNING: ./pir_app not found, run build.sh before the run jobs start" >&2

for d in "$@"; do
    if [[ "$d" == "msmarco-v2-all" ]]; then
        submit_dataset msmarco-v2
        full="$EMBED_JOB"
        for sub in msmarco-v2-25m msmarco-v2-50m msmarco-v2-100m; do
            submit_dataset "$sub" "$full"
        done
    else
        submit_dataset "$d"
    fi
done
echo "Queued. Watch with: squeue -u \$USER"
