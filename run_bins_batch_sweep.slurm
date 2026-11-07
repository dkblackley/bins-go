#!/bin/bash -l
#SBATCH --job-name=Bins_Batch
# PILLAR-Bin (two-stage, vec0) client storage vs embedding-DB batch size / rounds.
#
# Each query fetches its candidate embeddings in VecRounds sequential rounds of exactly
# VecBatch lookups. The client hint grows with sqrt(VecBatch/2 partitions), so a smaller
# batch saves client storage and costs rounds (latency). Two families are swept:
#   b<N>   -vecBatch N, rounds chosen automatically to cover as many docs as today
#   r<N>   -vecRounds N, batch = today's coverage / N
#   auto   today's setting, for reference
#
#   sbatch --array=0-$(( $(bash run_bins_batch_sweep.slurm --count | cut -d' ' -f1) - 1 )) run_bins_batch_sweep.slurm
#
# Results go to ${RESULTS_ROOT}/<knob>/bins_vec0_..., so each knob loads on its own:
#   RESULTS_DIR=${RESULTS_ROOT}/b32 python plotting/main.py
# metadata.json also records VecBatchSize, VecRounds and IDBatchSize.
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-07:30:00
#SBATCH --output=logs/BinsBatch_%A_%a.out
#SBATCH --error=logs/BinsBatch_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ "${1:-}" != "--count" ]]; then
    module --quiet purge
    module load hosts/hopper
    module load gnu9/9.3.0
fi
export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/dblackle/results_vecbatch}"
K_VAL=10

# knob tag | extra pir_app flags
KNOBS=(
    "auto|"
    "b512|-vecBatch 512"
    "b256|-vecBatch 256"
    "b128|-vecBatch 128"
    "b64|-vecBatch 64"
    "b32|-vecBatch 32"
    "b16|-vecBatch 16"
    "r2|-vecRounds 2"
    "r4|-vecRounds 4"
    "r8|-vecRounds 8"
    "r16|-vecRounds 16"
)
# dataset | hash table size | docs per bin values (edit to the configs you plot)
SETUPS=(
    "msmarco|1000000|10 50 100 250 500"
    "scifact|5000|10 50 100 250 500"
)

DATA_ARR=(); BS_ARR=(); DPB_ARR=(); KNOB_ARR=()
for setup in "${SETUPS[@]}"; do
    IFS='|' read -r d bs dpbs <<< "$setup"
    for dpb in $dpbs; do
        for knob in "${KNOBS[@]}"; do
            DATA_ARR+=("$d"); BS_ARR+=("$bs"); DPB_ARR+=("$dpb"); KNOB_ARR+=("$knob")
        done
    done
done

if [[ "${1:-}" == "--count" ]]; then
    echo "${#DATA_ARR[@]} configs (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"
    exit 0
fi

task_id=$SLURM_ARRAY_TASK_ID
dataset=${DATA_ARR[$task_id]}
bin_size=${BS_ARR[$task_id]}
docs_val=${DPB_ARR[$task_id]}
IFS='|' read -r knob knob_args <<< "${KNOB_ARR[$task_id]}"

case "$dataset" in
    msmarco) DB_SIZE=8841823 ;;
    scifact) DB_SIZE=5183 ;;
    *) echo "Unknown dataset: $dataset"; exit 1 ;;
esac

OUT_DIR="${RESULTS_ROOT}/${knob}/bins_vec0_${dataset}_k${K_VAL}_bs${bin_size}_dpb${docs_val}"
mkdir -p "$OUT_DIR"
echo "Task ${task_id}: ${dataset} bs=${bin_size} dpb=${docs_val} knob=${knob} (${knob_args:-automatic}) -> ${OUT_DIR}" \
    | tee "${OUT_DIR}/slurm_name.txt"

# shellcheck disable=SC2086  # knob_args is a word list
srun ./pir_app \
    -n "$DB_SIZE" -t bins -name "$dataset" -k "$K_VAL" \
    -binSize "$bin_size" -docsPerBin "$docs_val" \
    -outDir "$OUT_DIR" -outFile results -dataset "../../datasets" \
    -debug 1 -vectors=false -dim 192 \
    $knob_args &
srun_pid=$!
trap 'kill "${srun_pid:-}" 2>/dev/null; exit 143' TERM INT
wait "$srun_pid"

mv "logs/BinsBatch_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
mv "logs/BinsBatch_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true
