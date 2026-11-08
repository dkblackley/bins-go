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
#   bins-index Bluge index the bins need         datasets/bins_index.slurm              (after prep)
#   bins     bins grid, high + low tiers         run_bins_high.slurm, run_bins_low.slurm (after embed + bins-index)
#   pacmann  pacmann grid                        run_pacmann.slurm                       (after embed)
#   tree     tree grid                           run_tree_pir_old.slurm                  (after tree-db)
#            SWEEP=legacy|oat|grid|all picks the tree grid (tree_sweep_grid.sh, default all). msmarco/scifact
#            already have their data, so for them use STAGES="tree-db tree". Tree runs are split by estimated RAM:
#            <=100 GB on the normal partition, 250/500/1000/1500 GB on bigmem, one job array per bucket.
# A stage you leave out is assumed to be done already. If a job fails, the jobs that depend on it stay pending
# with reason DependencyNeverSatisfied: fix it, scancel them and re-run with STAGES set to the remaining stages.
#
# Run from the repo root after build.sh (pir_app must exist). Array sizes are worked out with each script's --count.
set -euo pipefail
cd "$(dirname "$0")/.."

# One dataset folder for every stage (default ../datasets, i.e. run from the scratch copy of the repo). The run
# scripts on their own default to ../../datasets, so pin both here (override with DATASETS_ROOT=...).
export DATASETS_ROOT="${DATASETS_ROOT:-../datasets}"
export DATASET_DIR="$DATASETS_ROOT"

STAGES="${STAGES:-prep embed tree-db bins-index bins pacmann tree}"
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

is_big() { [[ "$1" == msmarco-v2* || "$1" == miracl-en ]]; }

source ./slurm_common.sh
LIST_DIR="logs/lists/$(date +%Y%m%d_%H%M%S)"   # CONFIG_LIST files, kept so you can see what each array runs
mkdir -p "$LIST_DIR"

# true if any of these upstream jobs is being queued now (their outputs don't exist yet, so don't check inputs)
inputs_pending() { [[ -n "$(echo $1)" ]]; }

# submit_groups <method> <dataset> <plan file> <dependencies> <script for normal> <script for bigmem>
submit_groups() {
    local method=$1 d=$2 plan=$3 deps=$4 s_normal=$5 s_big=$6 gb min n f chunk=${MAX_ARRAY:-1000} off
    local total; total=$(grep -c . "$plan" || true)
    while read -r gb min n f; do
        case $gb in
            missing) echo "  $method  SKIPPED $min of $total configs: inputs not there yet (first missing: $n)"; continue ;;
            over)    echo "  $method  SKIPPED $min configs: estimated above ${MEM_BUCKETS##* } GB"; continue ;;
        esac
        local part script
        part=$(partition_for "$gb"); [[ $part == normal ]] && script=$s_normal || script=$s_big
        for (( off = 0; off < n; off += chunk )); do
            local m=$(( n - off < chunk ? n - off : chunk ))
            echo "  $method  $(submit "$deps" --export=ALL,CONFIG_LIST="$f",ARRAY_OFFSET=$off --array=0-$((m - 1)) \
                 --partition="$part" --mem="${gb}G" --time="$(slurm_time "$min")" \
                 --job-name="${method}_${d}_${gb}G" "$script") (${gb}G ${min}min $part: $m configs, $(basename "$f"))"
        done
    done < <(group_plan "$plan" "$LIST_DIR" "${method}_${d}")
}

# submit_dataset <name> [extra dependency] ; sets EMBED_JOB to the job the dataset's embeddings come from
submit_dataset() {
    local d="$1" extra="${2:-}" prep="" embed="" treedb="" binsidx=""
    echo "== $d"

    if has prep; then
        prep=$(submit "$extra" --export=ALL,DATASET="$d" --job-name="prep_$d" datasets/prep_datasets.slurm)
        echo "  prep     $prep"
    fi

    embed="$prep $extra"
    if has embed && [[ ! "$d" =~ ^msmarco-v2-[0-9]+m$ ]]; then   # subsamples copy the full set's embeddings
        if [[ "$d" == "msmarco-v2" || "$d" == "miracl-en" ]]; then
            local n=${NUM_SHARDS:-$([[ "$d" == miracl-en ]] && echo 8 || echo 24)} pca shards
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
        treedb=$(submit "$embed" --export=ALL,DATASET="$d",SWEEP="${SWEEP:-all}",WORKERS="${WORKERS:-$(is_big "$d" && echo 12 || echo 16)}" \
                 "${big[@]}" --job-name="treedb_$d" datasets/tree_artifacts.slurm)
        echo "  tree-db  $treedb"
    fi

    if has bins-index; then
        local big=()
        is_big "$d" && big=(--partition=bigmem --mem=500G)
        binsidx=$(submit "$prep $extra" --export=ALL,DATASET="$d" "${big[@]}" --job-name="binsidx_$d" \
                  datasets/bins_index.slurm)
        echo "  bins-index $binsidx"
    fi

    if has bins; then
        # every config of both tiers, estimated (measured usage if any, else the model in slurm_common.sh), split
        # into one array per (memory, time) bucket: <=100 GB -> run_bins_low.slurm on normal, else run_bins_high on bigmem
        local plan="$LIST_DIR/bins_${d}.plan"
        { BINS_DATASETS="$d" SLURM_SUBMIT_DIR="$PWD" bash run_bins_low.slurm --list-raw
          BINS_DATASETS="$d" SLURM_SUBMIT_DIR="$PWD" bash run_bins_high.slurm --list-raw; } \
            | CHECK_INPUTS=$(inputs_pending "$prep $embed $binsidx" && echo 0 || echo 1) bins_plan > "$plan"
        submit_groups bins "$d" "$plan" "$embed $binsidx" run_bins_low.slurm run_bins_high.slurm
    fi

    if has pacmann; then
        local n big=()
        n=$(count_of PACMANN_DATASETS="$d" run_pacmann.slurm)
        is_big "$d" && big=(--partition=bigmem --mem=1500G)
        echo "  pacmann  $(submit "$embed" --export=ALL,PACMANN_DATASETS="$d" --array=0-$((n - 1)) "${big[@]}" \
             --job-name="pacmann_$d" run_pacmann.slurm) ($n tasks)"
    fi

    if has tree; then
        # every config of the sweep, estimated and split by (memory, time) like bins; one script for both partitions
        local plan="$LIST_DIR/tree_${d}.plan"
        ( DATASET="$d" SWEEP="${SWEEP:-all}" CHECK_INPUTS=$(inputs_pending "$prep $embed $treedb" && echo 0 || echo 1) \
            bash tree_sweep_grid.sh --plan ) > "$plan"
        submit_groups tree "$d" "$plan" "$treedb $embed" run_tree_pir_old.slurm run_tree_pir_old.slurm
    fi
}

[[ $# -gt 0 ]] || { sed -n 2,22p "$0"; exit 1; }
needs_app() { has bins-index || has bins || has pacmann || has tree; }
[[ -x ./pir_app || -n "${DRY_RUN:-}" ]] || ! needs_app || echo "WARNING: ./pir_app not found, run build.sh before the run jobs start" >&2

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
