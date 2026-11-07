#!/bin/bash -l
#SBATCH --job-name=Bins_Text
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=0-07:30:00
#SBATCH --array=0-47
#SBATCH --output=logs/BinsText_%A_%a.out
#SBATCH --error=logs/BinsText_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

# Bins text mode (-text): the bins hold compressed document text (sorted Porter stems + tf, order-3 arithmetic
# coded) instead of embeddings, and the client re-ranks with BM25. No embedding files are needed.
#   BINS_DATASETS="msmarco nq hotpotqa fever scifact" bash run_bins_text.slurm --count
#   BINS_DATASETS="msmarco nq hotpotqa fever scifact" sbatch --array=0-<N-1> run_bins_text.slurm
# msmarco-v2 (138M docs) needs more memory: sbatch --mem=1500G ...
# Results go to bins_text_<dataset>_k<K>_bs<BS>_dpb<DPB> (plotting/load_results.py reads them as vec=2).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATASET_DIR="${DATASET_DIR:-../../datasets}"
RESULTS_DIR="${RESULTS_DIR:-/scratch/dblackle/results}"
TEXT_FORMAT="${TEXT_FORMAT:-bow}"   # bow | seq
TEXT_ORDER="${TEXT_ORDER:-3}"       # 0-3

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

BS_LIST=(${BS_LIST:-1000 10000 100000 1000000})
DPB_LIST=(${DPB_LIST:-10 50 100 250 500 1000})
K_LIST=(${K_LIST:-10})

DATA_ARR=(); K_ARR=(); BS_ARR=(); DPB_ARR=()
BINS_DATASETS="${BINS_DATASETS:-msmarco scifact}"
for d in $BINS_DATASETS; do
    for k in "${K_LIST[@]}"; do
        for bs in "${BS_LIST[@]}"; do
            [[ "$d" == msmarco-v2* && "$bs" -lt 100000 ]] && continue
            for dpb in "${DPB_LIST[@]}"; do
                DATA_ARR+=("$d"); K_ARR+=("$k"); BS_ARR+=("$bs"); DPB_ARR+=("$dpb")
            done
        done
    done
done

if [[ "${1:-}" == "--count" ]]; then
    echo "${#DATA_ARR[@]} configs (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"
    exit 0
fi

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

task_id=${SLURM_ARRAY_TASK_ID:?run with sbatch --array, or pass --count}
dataset=${DATA_ARR[$task_id]}
k_val=${K_ARR[$task_id]}
bin_size=${BS_ARR[$task_id]}
docs_val=${DPB_ARR[$task_id]}
DB_SIZE=$(db_size_of "$dataset")

SUFFIX=""
[[ "$TEXT_FORMAT" != "bow" || "$TEXT_ORDER" != "3" ]] && SUFFIX="_${TEXT_FORMAT}o${TEXT_ORDER}"
OUT_DIR="${RESULTS_DIR}/bins_text_${dataset}_k${k_val}_bs${bin_size}_dpb${docs_val}${SUFFIX}"
mkdir -p "$OUT_DIR"
echo "Task $task_id: $dataset k=$k_val bs=$bin_size dpb=$docs_val format=$TEXT_FORMAT order=$TEXT_ORDER -> $OUT_DIR" | tee "${OUT_DIR}/slurm_name.txt"

srun ./pir_app \
    -n "$DB_SIZE" \
    -t bins \
    -text \
    -text-format "$TEXT_FORMAT" \
    -text-order "$TEXT_ORDER" \
    -name "$dataset" \
    -k "$k_val" \
    -binSize "$bin_size" \
    -docsPerBin "$docs_val" \
    -outDir "$OUT_DIR" \
    -outFile "results" \
    -dataset "$DATASET_DIR" \
    -debug 1

mv "logs/BinsText_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
mv "logs/BinsText_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true
