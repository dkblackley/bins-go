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
#SBATCH --array=0-143
#SBATCH --output=logs/Pacmann_%A_%a.out
#SBATCH --error=logs/Pacmann_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
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

DATA_ARR=()
K_ARR=()
STEPS_ARR=()
NEIGHB_ARR=()

# Helper function to generate the 16 ablations per dataset
add_pacmann_ablations() {
    local d=$1

    for k in 10 100; do
        for steps in 5 10 15 20 25 30; do
            for neighb in 8 16 24 32 40 48; do
                DATA_ARR+=("$d"); K_ARR+=("$k"); STEPS_ARR+=("$steps"); NEIGHB_ARR+=("$neighb")
            done
        done
    done
}

# Build the setups for all datasets. Override to add the new ones, e.g.
#   PACMANN_DATASETS="nq hotpotqa fever" bash run_pacmann.slurm --count
#   PACMANN_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_pacmann.slurm
# Big corpora (graph + vectors of 100M+ docs need bigmem, e.g. --partition=bigmem --mem=1500G):
#   PACMANN_DATASETS="msmarco-v2-100m msmarco-v2" sbatch --partition=bigmem --mem=1500G --array=... run_pacmann.slurm
PACMANN_DATASETS="${PACMANN_DATASETS:-msmarco scifact}"
for d in $PACMANN_DATASETS; do
    add_pacmann_ablations "$d"
done

if [[ "${1:-}" == "--count" ]]; then
    echo "${#DATA_ARR[@]} configs (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"
    exit 0
fi

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
    local prefix="${DATASET_DIR}/${dataset}/${dataset}_corpus_embed_f32_${DB_SIZE}_192_${neighb_val}_neigh${neighb_val}_step${steps_val}_k${k_val}"
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