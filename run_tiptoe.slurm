#!/bin/bash -l
#SBATCH --job-name=Tiptoe_Baseline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-12:00:00
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --array=0-27
#SBATCH --output=logs/Tiptoe_%A_%a.out
#SBATCH --error=logs/Tiptoe_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

# Tiptoe baseline (see baselines/README.md).
#   MODE=sim  (default) baselines/tiptoe/simulate_tiptoe.py: exact quality + exact DB shapes + lower-bound bytes,
#             server time from ./simplepir_bench (go build -o simplepir_bench ./cmd/simplepir_bench). No SEAL needed.
#   MODE=real ./pir_app_tiptoe -t tiptoe, the real upstream protocol (needs `baselines/setup_baselines.sh tiptoe`).
#
# The sweep is over the number of k-means clusters, which is Tiptoe's one real knob: fewer, bigger clusters
# = better recall but a bigger ranking answer (download grows with cluster size, upload with #clusters x 192).
# Tiptoe itself used ~2500 docs/cluster on MS MARCO.
#   2 datasets x 7 cluster sizes x 2 k = 28 tasks

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
MODE="${MODE:-sim}"
if [[ "$MODE" == "real" ]]; then
    module load gnu12 2>/dev/null || module load gnu9/9.3.0   # pir_app_tiptoe was built with a C++20 compiler
else
    module load gnu9/9.3.0
fi

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main

DATA_ARR=(); DPC_ARR=(); K_ARR=()
for d in msmarco scifact; do
    for dpc in 250 500 1000 2500 5000 10000 40000; do
        for k in 10 100; do
            DATA_ARR+=("$d"); DPC_ARR+=("$dpc"); K_ARR+=("$k")
        done
    done
done
# trec-covid / new datasets: add them to the loop above, to GetDatasets in main.go,
# to DATASET_EMBEDS in baselines/tiptoe/preprocess_tiptoe.py and to DATASETS in baselines/common.py

task_id=$SLURM_ARRAY_TASK_ID
dataset=${DATA_ARR[$task_id]}
dpc=${DPC_ARR[$task_id]}
k_val=${K_ARR[$task_id]}
method="tiptoe"

case "$dataset" in
    msmarco) DB_SIZE=8841823; MAXQ=1000 ;;   # the per-query token is the slow part; 1000 queries is plenty for MRR@10
    scifact) DB_SIZE=5183; MAXQ=0 ;;
    trec-covid) DB_SIZE=171332; MAXQ=0 ;;
    *) echo "Unknown dataset: $dataset"; exit 1 ;;
esac

DATASET_DIR="../../datasets"
RESULTS_ROOT="/scratch/dblackle/results"
FAIL_FILE="${SLURM_SUBMIT_DIR:-$PWD}/tiptoe_fails"
finished=0
trap '[[ $finished -eq 1 ]] || echo "${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} $(date "+%F %T")" >> "$FAIL_FILE"' EXIT
trap 'kill "${srun_pid:-}" 2>/dev/null; exit 143' TERM INT

# 1) Preprocess (shared by both k values; flock so the two tasks don't race)
TT_DIR="${DATASET_DIR}/tiptoe/${dataset}_dpc${dpc}"
mkdir -p "$(dirname "$TT_DIR")"
(
    flock 9
    if [[ ! -f "${TT_DIR}/tiptoe_meta.json" ]]; then
        python baselines/tiptoe/preprocess_tiptoe.py --dataset "$DATASET_DIR" --name "$dataset" \
            --docs-per-cluster "$dpc" --out "$TT_DIR"
    fi
) 9>"${TT_DIR}.lock"

C=$(python -c "import json;print(json.load(open('${TT_DIR}/tiptoe_meta.json'))['num_clusters'])")
OUT_DIR="${RESULTS_ROOT}/${method}_${dataset}_c${C}_k${k_val}"
mkdir -p "$OUT_DIR"

cat <<EOF | tee "${OUT_DIR}/slurm_name.txt"
========================================
Array Task ID:     $SLURM_ARRAY_TASK_ID
Method:            $method
Dataset:           $dataset
K Value:           $k_val
Docs per cluster:  $dpc  (clusters: $C)
Tiptoe dir:        $TT_DIR
Output Dir:        $OUT_DIR
========================================
EOF

# 2) Run
if [[ "$MODE" == "sim" ]]; then
    srun python baselines/tiptoe/simulate_tiptoe.py \
        --dataset "$DATASET_DIR" --name "$dataset" --tiptoe-dir "$TT_DIR" --k "$k_val" \
        --results-root "$RESULTS_ROOT" --bench-bin ./simplepir_bench --bench-max-gb 32 \
        --parallelism "${SLURM_CPUS_PER_TASK:-1}" &
    srun_pid=$!
    wait "$srun_pid"
    mv "logs/Tiptoe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
    mv "logs/Tiptoe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true
    finished=1
    echo "All done for this node."
    exit 0
fi

srun ./pir_app_tiptoe \
    -n "$DB_SIZE" \
    -t "$method" \
    -name "$dataset" \
    -k "$k_val" \
    -tiptoe-dir "$TT_DIR" \
    -tiptoe-max-queries "$MAXQ" \
    -outDir "$OUT_DIR" \
    -outFile "results" \
    -dataset "$DATASET_DIR" \
    -debug 0 \
    -dim 192 &
srun_pid=$!
wait "$srun_pid"

mv "logs/Tiptoe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
mv "logs/Tiptoe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true
finished=1
echo "All done for this node."
