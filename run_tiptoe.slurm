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
#SBATCH --array=0-447%64          # for the default scifact + msmarco; other datasets: see --count below
#SBATCH --output=logs/Tiptoe_%A_%a.out
#SBATCH --error=logs/Tiptoe_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

# Tiptoe baseline (see baselines/README.md).
#   MODE=auto (default) real up to 50M docs, sim above (see REAL_MAX_DOCS below)
#   MODE=real ./pir_app_tiptoe -t tiptoe, the real upstream protocol (needs `baselines/setup_baselines.sh tiptoe`).
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
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATASET_DIR="${DATASET_DIR:-../../datasets}"
# -n per dataset: hardcoded for the originals, num_docs.txt (datasets/prepare_dataset.py) for the rest
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

# Datasets, e.g.
#   TIPTOE_DATASETS="nq dbpedia-entity fever" bash run_tiptoe.slurm --count     # prints the --array range to use
#   TIPTOE_DATASETS="nq dbpedia-entity fever" sbatch --array=0-<N-1> run_tiptoe.slurm
#   TIPTOE_DATASETS="msmarco-v2-25m msmarco-v2-50m msmarco-v2-100m msmarco-v2" sbatch --array=... --mem=240G run_tiptoe.slurm
# Corpora over BIG_DOCS (20M) get a smaller grid: each preprocessing pass is tens of GB of cluster data there.
# Their DPC values start above N/25196, Tiptoe's cap on the number of clusters.
TIPTOE_DATASETS="${TIPTOE_DATASETS:-scifact msmarco}"
BIG_DOCS=20000000
DATA_ARR=(); DPC_ARR=(); MA_ARR=(); B_ARR=(); H_ARR=(); K_ARR=()
for d in $TIPTOE_DATASETS; do
  n=$(db_size_of "$d")
  if (( n > BIG_DOCS )); then dpcs="6000 12000 25000 50000"; bundles="160"
  else dpcs="250 500 1000 2500 5000 10000 40000"; bundles="40 80 160 320"; fi
  for dpc in $dpcs; do
    for ma in 1 2; do
      for bundle in $bundles; do
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
    echo "${#DATA_ARR[@]} configs (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"
    exit 0
fi

task_id=$SLURM_ARRAY_TASK_ID
dataset=${DATA_ARR[$task_id]}
dpc=${DPC_ARR[$task_id]}
ma=${MA_ARR[$task_id]}
bundle=${B_ARR[$task_id]}
hint=${H_ARR[$task_id]}
k_val=${K_ARR[$task_id]}
method="tiptoe"
DB_SIZE=$(db_size_of "$dataset")
# The per-query token is the slow part of a real run; 1000 queries is plenty for MRR@10 on the big sets
if (( DB_SIZE > 1000000 )); then MAXQ=1000; else MAXQ=0; fi

# MODE=auto (default): real up to REAL_MAX_DOCS, simulated above it. Tiptoe's server keeps the whole ranking DB
# and its parsed corpus in memory, roughly 1 KB per doc, so 50M docs is about the limit of one big node.
REAL_MAX_DOCS="${REAL_MAX_DOCS:-50000000}"
MODE="${MODE:-auto}"
if [[ "$MODE" == "auto" ]]; then
    if (( DB_SIZE <= REAL_MAX_DOCS )); then MODE=real; else MODE=sim; fi
fi
# The real run needs Tiptoe's txt cluster files; the simulator reads the same content as compact arrays
FORMAT=txt; [[ "$MODE" == "sim" ]] && FORMAT=npy

module --quiet purge
module load hosts/hopper
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

RESULTS_ROOT="/scratch/dblackle/results"
FAIL_FILE="${SLURM_SUBMIT_DIR:-$PWD}/tiptoe_fails"
finished=0
trap '[[ $finished -eq 1 ]] || echo "${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} $(date "+%F %T")" >> "$FAIL_FILE"' EXIT
trap 'kill "${srun_pid:-}" 2>/dev/null; exit 143' TERM INT

# 1) Preprocess (shared across hint sizes and k; flock so tasks don't race)
TT_DIR="${DATASET_DIR}/tiptoe/${dataset}_dpc${dpc}_m${ma}_b${bundle}_${FORMAT}"
mkdir -p "$(dirname "$TT_DIR")"
(
    flock 9
    if [[ ! -f "${TT_DIR}/tiptoe_meta.json" ]]; then
        python baselines/tiptoe/preprocess_tiptoe.py --dataset "$DATASET_DIR" --name "$dataset" \
            --docs-per-cluster "$dpc" --multi-assign "$ma" --bundle-size "$bundle" --format "$FORMAT" --out "$TT_DIR"
    fi
) 9>"${TT_DIR}.lock"

# multi-assign as applied (it's capped at the number of clusters), so real and simulated runs name folders alike
read -r C MA_USED < <(python -c "import json;m=json.load(open('${TT_DIR}/tiptoe_meta.json'));print(m['num_clusters'],m['multi_assign'])")
OUT_DIR="${RESULTS_ROOT}/${method}_${dataset}_c${C}_m${MA_USED}_b${bundle}_h${hint}_k${k_val}"
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
