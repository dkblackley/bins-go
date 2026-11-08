#!/bin/bash -l
#SBATCH --job-name=Tiptoe_Baseline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-02:30:00
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --array=0-1343%64
#SBATCH --output=logs/Tiptoe_%A_%a.out
#SBATCH --error=logs/Tiptoe_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

# Tiptoe baseline (see baselines/README.md).
#   MODE=sim (default) baselines/tiptoe/simulate_tiptoe.py: exact quality + exact DB shapes + payload-only bytes,
#             server time from ./simplepir_bench. No SEAL needed.
#   MODE=real ./pir_app_tiptoe -t tiptoe, the real upstream protocol (needs `baselines/setup_baselines.sh tiptoe`).
#             Slow: SimplePIR hint generation over the whole corpus, then a 32 MiB token per query.
#
# Full grid search, nothing trimmed (unlike run_all_simulated.sh, which trims ma/bundle on big datasets locally):
#   DPC    docs per k-means cluster: fewer, bigger clusters = better recall, bigger download (Tiptoe used ~2500)
#   MA     clusters each boundary doc is copied into (Tiptoe: 2)
#   BUNDLE docs per URL bundle (Tiptoe: 160): bigger = more docs returned, bigger round-2 download
#   HINT   embedding hint size in MB (Tiptoe: 500): moves bytes between upload (DB width) and download (DB height)
#   k      10, 100
#   per dataset: 7 DPC x 2 MA x 4 BUNDLE x 2 HINT x 2 k = 224 tasks; 6 datasets = 1344.
#
# Usage:
#   bash run_tiptoe.slurm --count                                   # prints the --array range for $TIPTOE_DATASETS
#   sbatch run_tiptoe.slurm                                         # all of the default datasets below
#   TIPTOE_DATASETS="nq fever" bash run_tiptoe.slurm --count        # a subset (then sbatch --array=0-<N-1>%64 ...)
#   TIPTOE_DATASETS="nq fever" sbatch --array=0-447%64 run_tiptoe.slurm
# If the cluster's MaxArraySize is below 1344, submit the datasets in groups like that.
#
# Memory: preprocessing loads the whole float32 embedding matrix, and the simulator loads every doc ID from
# corpus.jsonl. 120G is fine up to MS MARCO (8.8M). msmarco-v2 (138M docs, ~106 GB of vectors) needs a lot more,
# e.g. TIPTOE_DATASETS="msmarco-v2-50m msmarco-v2" sbatch --mem=500G --array=0-447%16 run_tiptoe.slurm
#
# Preprocessing (k-means + bundling, ~minutes to an hour on the big sets) is shared by every (hint, k) of one
# (DPC, MA, BUNDLE) via flock, and the k-means itself by every (MA, BUNDLE) of one cluster count via
# --kmeans-cache (also flocked), so 1344 tasks still only run one k-means per (dataset, cluster count).

TIPTOE_DATASETS="${TIPTOE_DATASETS:-scifact nq fever msmarco msmarco-v2-50m msmarco-v2}"
DATA_ARR=(); DPC_ARR=(); MA_ARR=(); B_ARR=(); H_ARR=(); K_ARR=()
for d in $TIPTOE_DATASETS; do
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
if [[ "${1:-}" == "--count" ]]; then
    echo "${#DATA_ARR[@]} configs for: $TIPTOE_DATASETS (use --array=0-$(( ${#DATA_ARR[@]} - 1 ))%64)"
    exit 0
fi

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

task_id=$SLURM_ARRAY_TASK_ID
dataset=${DATA_ARR[$task_id]}
dpc=${DPC_ARR[$task_id]}
ma=${MA_ARR[$task_id]}
bundle=${B_ARR[$task_id]}
hint=${H_ARR[$task_id]}
k_val=${K_ARR[$task_id]}
method="tiptoe"

DATASET_DIR="${DATASET_DIR:-../../datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/dblackle/results}"

# -n and the real-mode query cap: hardcoded for the originals, num_docs.txt (datasets/prepare_dataset.py) otherwise
case "$dataset" in
    msmarco)    DB_SIZE=8841823; MAXQ=1000 ;;   # real mode: the per-query token is the slow part; 1000 is plenty
    scifact)    DB_SIZE=5183; MAXQ=0 ;;
    trec-covid) DB_SIZE=171332; MAXQ=0 ;;
    *)
        f="${DATASET_DIR}/${dataset}/num_docs.txt"
        [[ -f "$f" ]] || { echo "No $f, run datasets/prep_datasets.slurm for $dataset first" >&2; exit 1; }
        DB_SIZE=$(cat "$f"); MAXQ=1000 ;;
esac

FAIL_FILE="${SLURM_SUBMIT_DIR:-$PWD}/tiptoe_fails"
finished=0
trap '[[ $finished -eq 1 ]] || echo "${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} $(date "+%F %T")" >> "$FAIL_FILE"' EXIT
trap 'kill "${srun_pid:-}" 2>/dev/null; exit 143' TERM INT

# 1) Preprocess (shared across hint sizes and k; flock so tasks don't race; k-means shared via --kmeans-cache)
TT_DIR="${DATASET_DIR}/tiptoe/${dataset}_dpc${dpc}_m${ma}_b${bundle}"
mkdir -p "$(dirname "$TT_DIR")"
(
    flock 9
    if [[ ! -f "${TT_DIR}/tiptoe_meta.json" ]]; then
        python baselines/tiptoe/preprocess_tiptoe.py --dataset "$DATASET_DIR" --name "$dataset" \
            --docs-per-cluster "$dpc" --multi-assign "$ma" --bundle-size "$bundle" --out "$TT_DIR" \
            --kmeans-cache "${DATASET_DIR}/tiptoe/kmeans_cache"
    fi
) 9>"${TT_DIR}.lock"

C=$(python -c "import json;print(json.load(open('${TT_DIR}/tiptoe_meta.json'))['num_clusters'])")
OUT_DIR="${RESULTS_ROOT}/${method}_${dataset}_c${C}_m${ma}_b${bundle}_h${hint}_k${k_val}"
if [[ -f "${OUT_DIR}/metadata.json" ]]; then
    # several DPC values can give the same cluster count (1 on scifact, the 25196 cap on msmarco-v2); first one wins
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
