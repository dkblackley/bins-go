#!/bin/bash -l
#SBATCH --job-name=Tiptoe_Baseline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --array=0-447%64
#SBATCH --output=logs/Tiptoe_%A_%a.out
#SBATCH --error=logs/Tiptoe_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

# Tiptoe baseline (see baselines/README.md).
#   MODE=real (default) ./pir_app_tiptoe -t tiptoe, the real upstream protocol (needs `baselines/setup_baselines.sh tiptoe`).
#   MODE=sim  baselines/tiptoe/simulate_tiptoe.py: exact quality + exact DB shapes + payload-only bytes,
#             server time from ./simplepir_bench. No SEAL needed. Use it as a cross-check, or if SEAL won't build.
#
# Knobs swept (see baselines/README.md section 3):
#   DPC    docs per k-means cluster: fewer, bigger clusters = better recall, bigger download (Tiptoe used ~2500)
#   MA     clusters each boundary doc is copied into (Tiptoe: 2)
#   BUNDLE docs per URL bundle (Tiptoe: 160): bigger = more docs returned, bigger round-2 download
#   HINT   embedding hint size in MB (Tiptoe: 500): moves bytes between upload (DB width) and download (DB height)
#   2 datasets x 7 DPC x 2 MA x 4 BUNDLE x 2 HINT x 2 k = 448 tasks.
# Real mode on MS MARCO is slow (SimplePIR hint generation over 8.8M docs, then a 32 MiB token per query),
# so MS MARCO runs 1000 queries. If the queue is tight, run MS MARCO with MODE=sim and scifact with MODE=real.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
MODE="${MODE:-real}"
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

DATA_ARR=(); DPC_ARR=(); MA_ARR=(); B_ARR=(); H_ARR=(); K_ARR=()
for d in scifact msmarco; do
  for dpc in 250 500 1000 2500 5000 10000 40000; do
    for ma in 1 2; do
      for bundle in 40 80 160 320; do
        for hint in 125 500; do
          for k in 10 100; do
            DATA_ARR+=("$d"); DPC_ARR+=("$dpc"); MA_ARR+=("$ma"); B_ARR+=("$bundle"); H_ARR+=("$hint"); K_ARR+=("$k")
          done
        done
      done
    done
  done
done
# trec-covid / new datasets: add them to the loop above, to GetDatasets in main.go,
# to DATASET_EMBEDS in baselines/tiptoe/preprocess_tiptoe.py and to DATASETS in baselines/common.py

task_id=$SLURM_ARRAY_TASK_ID
dataset=${DATA_ARR[$task_id]}
dpc=${DPC_ARR[$task_id]}
ma=${MA_ARR[$task_id]}
bundle=${B_ARR[$task_id]}
hint=${H_ARR[$task_id]}
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

# 1) Preprocess (shared across hint sizes and k; flock so tasks don't race)
TT_DIR="${DATASET_DIR}/tiptoe/${dataset}_dpc${dpc}_m${ma}_b${bundle}"
mkdir -p "$(dirname "$TT_DIR")"
(
    flock 9
    if [[ ! -f "${TT_DIR}/tiptoe_meta.json" ]]; then
        python baselines/tiptoe/preprocess_tiptoe.py --dataset "$DATASET_DIR" --name "$dataset" \
            --docs-per-cluster "$dpc" --multi-assign "$ma" --bundle-size "$bundle" --out "$TT_DIR"
    fi
) 9>"${TT_DIR}.lock"

C=$(python -c "import json;print(json.load(open('${TT_DIR}/tiptoe_meta.json'))['num_clusters'])")
OUT_DIR="${RESULTS_ROOT}/${method}_${dataset}_c${C}_m${ma}_b${bundle}_h${hint}_k${k_val}"
if [[ -f "${OUT_DIR}/metadata.json" ]]; then
    # small datasets: several (DPC, MA) settings can give the same cluster count; first one wins
    echo "${OUT_DIR} already has results, skipping"; finished=1; exit 0
fi
mkdir -p "$OUT_DIR"

cat <<EOF | tee "${OUT_DIR}/slurm_name.txt"
========================================
Array Task ID:     $SLURM_ARRAY_TASK_ID
Method:            $method
Dataset:           $dataset
K Value:           $k_val
Docs per cluster:  $dpc  (clusters: $C)
Multi-assign:      $ma
Bundle size:       $bundle
Emb hint MB:       $hint
Mode:              $MODE
Tiptoe dir:        $TT_DIR
Output Dir:        $OUT_DIR
========================================
EOF

# 2) Run
if [[ "$MODE" == "sim" ]]; then
    srun python baselines/tiptoe/simulate_tiptoe.py \
        --dataset "$DATASET_DIR" --name "$dataset" --tiptoe-dir "$TT_DIR" --k "$k_val" \
        --emb-hint-mb "$hint" --results-root "$RESULTS_ROOT" --bench-bin ./simplepir_bench --bench-max-gb 32 \
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
    -tiptoe-emb-hint-mb "$hint" \
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
