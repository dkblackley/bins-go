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
#
# GPU=<profile> sends the embed jobs to another GPU queue instead of embed.slurm's DGX A100.40gb (fp16, short time
# limits, 32 GB RAM, so they backfill sooner; a job that hits its limit requeues itself and resumes). Run it next to a series that's already queued: whichever embed finishes
# first wins, the other one exits 3 and its dependants stay pending (DependencyNeverSatisfied) for you to scancel.
#   a100       A100.80gb on gpuq OR contrib-gpuq (96 GPUs; preemptible on contrib, jobs requeue + resume)   <- default pick
#   any        any GPU at all (--gres=gpu:1 on gpuq, contrib-gpuq, contrib-h100, incl. MIG slices), 4 h limit
#   a100-gpuq  A100.80gb on gpuq only (40 GPUs, never preempted)
#   mig40      MIG 3g.40gb slice on gpuq (24; ~0.4 of an A100)      mig20  MIG 2g.20gb (8; ~0.25 of an A100)
#   h100       H100.80gb on contrib-h100 (4, preemptible)           b200   B200.180gb on contrib-b200 (8, reserved)
#   dgx        A100.40gb on gpuq (what embed.slurm asks for), but fp16 + the shorter limits
# Job names get a _<profile> suffix (squeue -u $USER -o "%.10i %.9P %.28j %.2t %.10M %R" shows them in full).
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

# ---- GPU profile for the embed jobs (see header). Prints sbatch args for one embed MODE.
SFX="${GPU:+_$GPU}"     # job name suffix, so two series are told apart in squeue
gpu_args() {   # gpu_args <all|pca|shard|finish>
    [[ -z "${GPU:-}" ]] && return 0
    local part qos=gpu gres requeue="" cpus=16 t_all t_shard
    case "$GPU" in
        any)       part=gpuq,contrib-gpuq,contrib-h100; gres="";  cpus=4; requeue=1; t_all=4; t_shard=4 ;;
        a100)      part=gpuq,contrib-gpuq; gres=A100.80gb;  requeue=1; t_all=2;  t_shard=2 ;;
        a100-gpuq) part=gpuq;              gres=A100.80gb;             t_all=2;  t_shard=2 ;;
        dgx)       part=gpuq;              gres=A100.40gb;             t_all=2;  t_shard=2 ;;
        mig40)     part=gpuq;              gres=3g.40gb;    cpus=4;    t_all=6;  t_shard=6 ;;
        mig20)     part=gpuq;              gres=2g.20gb;    cpus=4;    t_all=10; t_shard=10 ;;
        h100)      part=contrib-h100;      gres=H100.80gb;  requeue=1; t_all=2;  t_shard=2 ;;
        b200)      part=contrib-b200;      gres=B200.180gb; requeue=1; t_all=2;  t_shard=2 ;;
        *) echo "Unknown GPU=$GPU (any a100 a100-gpuq dgx mig40 mig20 h100 b200)" >&2; exit 1 ;;
    esac
    part="${GPU_PARTITION:-$part}"; gres="${GPU_GRES:-$gres}"; qos="${GPU_QOS:-$qos}"
    local hours mem=32G
    case "$1" in
        all) hours=$t_all ;; shard) hours=$t_shard ;; pca) hours=1 ;; finish) hours=4; mem=64G ;;
    esac
    hours="${GPU_HOURS:-$hours}"
    echo "--partition=$part --qos=$qos --gres=gpu:${gres:+$gres:}1 --cpus-per-task=$cpus --mem=$mem --time=$(printf '0-%02d:00:00' "$hours") --open-mode=append ${requeue:+--requeue}"
}
# fp16 + a per-series resume key for the GPU= series (embed.slurm / collection_preproc.py read these)
EMB_ENV=""
[[ -n "${GPU:-}" ]] && EMB_ENV=",EMBED_FP16=${EMBED_FP16:-1},EMBED_TAG=${GPU}"

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
                 --job-name="${method}_${d}_${gb}G$SFX" "$script") (${gb}G ${min}min $part: $m configs, $(basename "$f"))"
        done
    done < <(group_plan "$plan" "$LIST_DIR" "${method}_${d}")
}

# submit_dataset <name> [extra dependency] ; sets EMBED_JOB to the job the dataset's embeddings come from
submit_dataset() {
    local d="$1" extra="${2:-}" prep="" embed="" treedb="" binsidx=""
    echo "== $d"

    # a msmarco-v2 subsample's "prep" copies the full set's embedding rows, so it's part of its embed stage too
    if has prep || { has embed && [[ "$d" =~ ^msmarco-v2-[0-9]+m$ ]]; }; then
        prep=$(submit "$extra" --export=ALL,DATASET="$d" --job-name="prep_$d$SFX" datasets/prep_datasets.slurm)
        echo "  prep     $prep"
    fi

    embed="$prep $extra"
    local pfx="$d"; [[ "$d" == trec-covid ]] && pfx=trec
    if has embed && [[ -f "$DATASETS_ROOT/$d/${pfx}_corpus_embed_f32.npy.done" ]]; then
        echo "  embed    already done ($d/${pfx}_corpus_embed_f32.npy.done), not queued"
    elif has embed && [[ ! "$d" =~ ^msmarco-v2-[0-9]+m$ ]]; then   # subsamples copy the full set's embeddings
        if [[ "$d" == "msmarco-v2" || "$d" == "miracl-en" ]]; then
            local n=${NUM_SHARDS:-$([[ "$d" == miracl-en ]] && echo 8 || echo 24)} pca shards
            pca=$(submit "$prep $extra" --export=ALL,DATASET="$d",MODE=pca,NUM_SHARDS=$n$EMB_ENV --time=0-02:00:00 \
                  $(gpu_args pca) --job-name="pca_$d$SFX" datasets/embed.slurm)
            shards=$(submit "$pca" --export=ALL,DATASET="$d",MODE=shard,NUM_SHARDS=$n$EMB_ENV --array=0-$((n - 1)) \
                  $(gpu_args shard) --job-name="emb_$d$SFX" datasets/embed.slurm)
            embed=$(submit "$shards" --export=ALL,DATASET="$d",MODE=finish,NUM_SHARDS=$n$EMB_ENV --time=0-06:00:00 \
                  $(gpu_args finish) --job-name="merge_$d$SFX" datasets/embed.slurm)
            echo "  embed    pca=$pca shards=$shards merge=$embed"
        else
            embed=$(submit "$prep $extra" --export=ALL,DATASET="$d"$EMB_ENV $(gpu_args all) --job-name="emb_$d$SFX" \
                  datasets/embed.slurm)
            echo "  embed    $embed"
        fi
    fi
    EMBED_JOB="$embed"

    if has tree-db; then
        local big=()
        is_big "$d" && big=(--partition=bigmem --mem=1000G)
        treedb=$(submit "$embed" --export=ALL,DATASET="$d",SWEEP="${SWEEP:-all}",WORKERS="${WORKERS:-$(is_big "$d" && echo 12 || echo 16)}" \
                 "${big[@]}" --job-name="treedb_$d$SFX" datasets/tree_artifacts.slurm)
        echo "  tree-db  $treedb"
    fi

    if has bins-index; then
        local big=()
        is_big "$d" && big=(--partition=bigmem --mem=500G)
        binsidx=$(submit "$prep $extra" --export=ALL,DATASET="$d" "${big[@]}" --job-name="binsidx_$d$SFX" \
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
             --job-name="pacmann_$d$SFX" run_pacmann.slurm) ($n tasks)"
    fi

    if has tree; then
        # every config of the sweep, estimated and split by (memory, time) like bins; one script for both partitions
        local plan="$LIST_DIR/tree_${d}.plan"
        ( DATASET="$d" SWEEP="${SWEEP:-all}" CHECK_INPUTS=$(inputs_pending "$prep $embed $treedb" && echo 0 || echo 1) \
            bash tree_sweep_grid.sh --plan ) > "$plan"
        submit_groups tree "$d" "$plan" "$treedb $embed" run_tree_pir_old.slurm run_tree_pir_old.slurm
    fi
}

[[ $# -gt 0 ]] || { sed -n 2,32p "$0"; exit 1; }
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
