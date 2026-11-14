#!/bin/bash -l
#SBATCH --job-name=Bins_Batch
# PILLAR-Bin (two-stage, vec0) client storage vs embedding-DB batch size / rounds.
#
# Grid search over the bins-only client-storage knobs of the two-stage (vec0) layout.
# Nothing here touches the documents or embeddings: it only changes how many lookups
# each PIR batch makes (and so how many partitions, i.e. hints, the client keeps).
#
#   vec batch   -vecBatch   lookups per embedding-DB round. The max is the batch used
#               so far, min(3*dpb, 3000) for MS MARCO and min(dpb, 500) for SciFact;
#               the grid tries max, max/2, max/4, max/8
#   vec rounds  -vecRounds  sequential embedding-DB rounds: 1, 2, 4, 8
#               (candidates fetched = batch * rounds; the rest are dropped)
#   id batch    -idBatch    ID-DB batch: 24 (today), 16, 12, 8. A query with more
#               terms than the batch puts several lookups in one partition: more
#               sequential Piano queries and more frequent refreshes (it logs a warning)
#
# 4 x 4 x 4 = 64 combinations per (dataset, bs, dpb), plus "auto" (today's run).
# Some corners (1 round of max/8) will be bad on every metric; that is the point.
#
#   bash run_bins_batch_sweep.slurm --count
#   sbatch --array=0-$(( $(bash run_bins_batch_sweep.slurm --count | cut -d' ' -f1) - 1 ))%64 run_bins_batch_sweep.slurm
#
# Results go to ${RESULTS_ROOT}/<tag>/bins_vec0_..., tag = vb<batch>_vr<rounds>_ib<idbatch>
# (vb is the fraction of the max: 1, 2, 4, 8 = max/1 .. max/8, so a tag means the
# same thing for every dpb). Load one with:
#   RESULTS_DIR=${RESULTS_ROOT}/vb1_vr1_ib24 python plotting/main.py
# metadata.json records VecBatchSize, VecRounds and IDBatchSize.
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

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/$USER/results_vecbatch}"
K_VAL=10

VEC_DIVS=(1 2 4 8)     # vec batch = max / this
VEC_ROUNDS=(1 2 4 8)
ID_BATCHES=(24 16 12 8)
KNOBS=("auto")
for vd in "${VEC_DIVS[@]}"; do
    for vr in "${VEC_ROUNDS[@]}"; do
        for ib in "${ID_BATCHES[@]}"; do
            KNOBS+=("${vd}|${vr}|${ib}")
        done
    done
done
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
knob_spec=${KNOB_ARR[$task_id]}

case "$dataset" in
    msmarco) DB_SIZE=8841823 ;;
    scifact) DB_SIZE=5183 ;;
    *) echo "Unknown dataset: $dataset"; exit 1 ;;
esac

if [[ "$knob_spec" == "auto" ]]; then
    knob="auto"; knob_args=""
else
    IFS='|' read -r vd vr ib <<< "$knob_spec"
    if [[ "$dataset" == "msmarco" ]]; then
        vec_max=$(( 3 * docs_val < 3000 ? 3 * docs_val : 3000 ))
    else
        vec_max=$(( docs_val < 500 ? docs_val : 500 ))
    fi
    vb=$(( vec_max / vd )); if (( vb < 2 )); then vb=2; fi
    knob="vb${vd}_vr${vr}_ib${ib}"
    knob_args="-vecBatch ${vb} -vecRounds ${vr} -idBatch ${ib}"
fi

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
