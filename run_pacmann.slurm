#!/bin/bash -l
#SBATCH --job-name=Pacmann_Ablation
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-05:00:00
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --array=0-111
#SBATCH --output=logs/Pacmann_%A_%a.out
#SBATCH --error=logs/Pacmann_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

set -euo pipefail

# ---- the grid: THE one definition. datasets/run_all.sh, check_runs.sh, slurm_common.sh (queued_dirs) and
# local_run_go_simulated.sh all read it through --list, so array task i is line i+1 of --list everywhere.
#   bash run_pacmann.slurm --list    "index dataset k steps neighbours" per config
#   bash run_pacmann.slurm --count
#   PACMANN_GRID=g1 bash run_pacmann.slurm --list    an older grid (queued_dirs decodes still-queued old jobs with it)
# Jobs are named pacmann_<d>_<grid>[_sim] (run_all.sh); a name without a grid tag is g1.
#   g1 (until 2026-11-17)  k 10 100 x steps 5..30 x neighbours 8..48
#   g2 (2026-11-17)        k 10 x steps 1 3 5..30 x neighbours 2 4 8..48
#   g3 (2026-11-17)        k 10 x steps 3 5..30 x neighbours 2 4 8..48
#   g4 (2026-11-18)        g3 + pacmann_extra (per dataset, after the grid so g3's task numbers stay the same) <- current
# neighbours=1 can't run: BatchSize 1 gives 0 PIR partitions (pianopir NewSimpleBatchPianoPIR) and the NGT build
# finds no neighbour (it searches int(1.5*m) = 1 = the vertex itself).
# Changing the grid: add a new case here, make it the default, and set PACMANN_GRID in datasets/run_all.sh to it.
PACMANN_GRID="${PACMANN_GRID:-g4}"
case "$PACMANN_GRID" in
    g1) PACMANN_K="10 100"; PACMANN_STEPS="5 10 15 20 25 30";     PACMANN_NEIGHB="8 16 24 32 40 48" ;;
    g2) PACMANN_K="10";     PACMANN_STEPS="1 3 5 10 15 20 25 30"; PACMANN_NEIGHB="2 4 8 16 24 32 40 48" ;;
    g3|g4) PACMANN_K="10";  PACMANN_STEPS="3 5 10 15 20 25 30";   PACMANN_NEIGHB="2 4 8 16 24 32 40 48" ;;
    *)  echo "run_pacmann.slurm: unknown PACMANN_GRID=$PACMANN_GRID" >&2; exit 1 ;;
esac

# g4: extra "steps:neighbours" configs per dataset (k 10) that fill the MRR@10 values the grid jumps over, so the
# fixed-MRR plots have a PACMANN config within +/- 0.01 of every multiple of 0.05. Only the fixed-MRR scripts load
# them (they are not in plotting/current_grid.txt). The grid jumps most between steps 3 / 5 / 10 and neighbours
# 4 / 8 / 16; these mostly add steps at neighbour counts that already have an NGT graph. Expected MRR@10 is
# interpolated from the g3 runs (2026-11-18), so it is a guess:
#   scifact  0.15: 4:4 5:3 | 0.21-0.29: 6:4 7:4 8:4 9:4 | 0.35: 12:4 13:4 4:8 | 0.51: 4:16 8:8
#   fever    0.25: 6:16 4:32 7:16 | 0.31-0.34: 8:16 6:24 4:40
#   nq       0.25: 7:16 6:16 12:8 | 0.30: 8:16 4:40 4:48 17:8
#   dbpedia-entity (g3 itself has 20 configs it never ran)  0.25-0.40: 5:12 7:8 5:20 7:16 | 0.53: 20:12
pacmann_extra() {   # dataset -> "steps:neighbours ..." (g4 only)
    [[ $PACMANN_GRID == g4 ]] || return 0
    case "$1" in
        scifact)        echo "4:4 5:3 6:4 7:4 8:4 9:4 12:4 13:4 4:8 4:16 8:8" ;;
        fever)          echo "6:16 4:32 7:16 8:16 6:24 4:40" ;;
        nq)             echo "7:16 6:16 12:8 8:16 4:40 4:48 17:8" ;;
        dbpedia-entity) echo "5:12 7:8 5:20 7:16 20:12" ;;
    esac
}

DATA_ARR=()
K_ARR=()
STEPS_ARR=()
NEIGHB_ARR=()
# Build the setups for all datasets. Override the datasets (not the grid), e.g.
#   PACMANN_DATASETS="nq hotpotqa fever" bash run_pacmann.slurm --count
#   PACMANN_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_pacmann.slurm
# Big corpora (graph + vectors of 100M+ docs need bigmem, e.g. --partition=bigmem --mem=1500G):
#   PACMANN_DATASETS="msmarco-v2-100m msmarco-v2" sbatch --partition=bigmem --mem=1500G --array=... run_pacmann.slurm
PACMANN_DATASETS="${PACMANN_DATASETS:-msmarco scifact}"
for d in $PACMANN_DATASETS; do
    for k in $PACMANN_K; do
        for steps in $PACMANN_STEPS; do
            for neighb in $PACMANN_NEIGHB; do
                DATA_ARR+=("$d"); K_ARR+=("$k"); STEPS_ARR+=("$steps"); NEIGHB_ARR+=("$neighb")
            done
        done
    done
    for x in $(pacmann_extra "$d"); do
        DATA_ARR+=("$d"); K_ARR+=(10); STEPS_ARR+=("${x%%:*}"); NEIGHB_ARR+=("${x##*:}")
    done
done

case "${1:-}" in
    --count) echo "${#DATA_ARR[@]} configs (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"; exit 0 ;;
    --list)  for i in "${!DATA_ARR[@]}"; do echo "$i ${DATA_ARR[$i]} ${K_ARR[$i]} ${STEPS_ARR[$i]} ${NEIGHB_ARR[$i]}"; done; exit 0 ;;
esac

cd "$SLURM_SUBMIT_DIR"
source ./noram_cleanup.sh
# soft Go heap limit at 90% of the allocation (see slurm_common.sh)
if [[ -n "${SLURM_MEM_PER_NODE:-}" && -z "${GOMEMLIMIT:-}" ]]; then export GOMEMLIMIT="$(( SLURM_MEM_PER_NODE * 9 / 10 ))MiB"; fi

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
# pir_app finds libngt/libhnsw through its rpath; this also covers a pir_app built without go_env.sh
export LD_LIBRARY_PATH="${NGT_PREFIX:-$HOME/opt/ngt-amd}/lib:${HNSW_PREFIX:-$HOME/opt/hnsw}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

DATASET_DIR="${DATASET_DIR:-../../datasets}"
# -n for each dataset: hardcoded for the originals, read from num_docs.txt (written by datasets/prepare_dataset.py)
# for everything else (nq, hotpotqa, fever, msmarco-v2, msmarco-v2-100m, ...)
db_size_of() {
    case "$1" in
        msmarco)    echo 8841823 ;;
        scifact)    echo 5183 ;;
        trec-covid) echo 171332 ;;
        *)
            local f="${DATASET_DIR}/$1/num_docs.txt"
            [[ -f "$f" ]] || { echo "No $f, run datasets/prep_datasets.slurm for $1 first" >&2; exit 1; }
            cat "$f" ;;
    esac
}

task_id=$SLURM_ARRAY_TASK_ID

dataset=${DATA_ARR[$task_id]}
k_val=${K_ARR[$task_id]}
steps_val=${STEPS_ARR[$task_id]}
neighb_val=${NEIGHB_ARR[$task_id]}
method="pacmann"

DB_SIZE=$(db_size_of "$dataset")

FAIL_FILE="${SLURM_SUBMIT_DIR:-$PWD}/pacman_fails"
finished=0

on_exit() {
    local rc=$?
    if [[ "$finished" -ne 1 ]]; then
        echo "${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} rc=${rc} $(date '+%F %T')" >> "$FAIL_FILE"
    fi
    declare -F cleanup >/dev/null && cleanup || true
    noram_cleanup_job   # -noram folders a killed run left in temp_storage (noram_cleanup.sh)
}
trap on_exit EXIT
trap 'kill "${srun_pid:-}" 2>/dev/null; exit 143' TERM INT

# SIMULATE=1: simulated PIR (pir_app -simulate), results in their own folder (as slurm_common.sh does for bins/tree)
SIM_ARGS=()
if [[ "${SIMULATE:-0}" == 1 ]]; then RESULTS_DIR="${RESULTS_DIR:-/scratch/$USER/results_simulated}"; SIM_ARGS=(-simulate); fi
# NORAM=1: -noram, the .npy embeddings are mmap'd instead of read into RAM (for pacmann that is all -noram moves)
[[ "${NORAM:-0}" == 1 ]] && SIM_ARGS+=(-noram -noram-dir "${NORAM_DIR:-temp_storage}")
OUT_DIR="${RESULTS_DIR:-/scratch/$USER/results}/${method}_${dataset}_k${k_val}_steps${steps_val}_neighb${neighb_val}"
# OUT_DIR="../results/${method}_${dataset}_k${k_val}_steps${steps_val}_neighb${neighb_val}"
if [[ -s "${OUT_DIR}/metadata.json" && -z "${FORCE:-}" ]]; then
    echo "Already done: ${OUT_DIR} (set FORCE=1 to rerun)"; finished=1; exit 0
fi
mkdir -p "$OUT_DIR"

cat <<EOF > "${OUT_DIR}/slurm_name.txt"
========================================
Array Task ID: $SLURM_ARRAY_TASK_ID
Method:        $method
Dataset:       $dataset
K Value:       $k_val
"Steps:         $steps_val"
"Neighbours:    $neighb_val"
Output Dir:    $OUT_DIR
========================================
EOF

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Method:        $method"
echo "Dataset:       $dataset"
echo "K Value:       $k_val"
echo "Steps:         $steps_val"
echo "Neighbours:    $neighb_val"
echo "Output Dir:    $OUT_DIR"
echo "========================================"

cleanup() {
    # pir_app writes the graph to /scratch/$USER/pac (Pacmann/private-search.go) and never reads it back
    local prefix="${PACMANN_CACHE_DIR:-/scratch/$USER/pac}/${dataset}_corpus_embed_f32_${DB_SIZE}_192_${neighb_val}_neigh${neighb_val}_step${steps_val}_k${k_val}"
    rm -rf -- "${prefix}.ngt" "${prefix}_graph.npy"
}

# Execute binary
srun ./pir_app \
    -n "$DB_SIZE" \
    -t "$method" \
    -name "$dataset" \
    -k "$k_val" \
    -steps "$steps_val" \
    -neighb "$neighb_val" \
    -outDir "$OUT_DIR" \
    -outFile "results" \
    -dataset "$DATASET_DIR" \
    -debug 1 \
    ${SIM_ARGS[@]+"${SIM_ARGS[@]}"} \
    -dim 192 &

srun_pid=$!
wait "$srun_pid"

mv "logs/Pacmann_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
mv "logs/Pacmann_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true

finished=1
echo "All done for this node."