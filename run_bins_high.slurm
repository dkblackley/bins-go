#!/bin/bash -l
#SBATCH --job-name=Bins_High
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1500G
#SBATCH --time=0-07:30:00
#SBATCH --array=0-351
#SBATCH --output=logs/Bins_%A_%a.out
#SBATCH --error=logs/Bins_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

set -euo pipefail
# cd "$SLURM_SUBMIT_DIR"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
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

BS_LIST=(100 500 1000 5000 10000 50000 100000 500000 1000000 5000000 10000000)
# BS_LIST=(100 500 1000 5000 10000 50000 100000 500000 1000000)

DATA_ARR=()
K_ARR=()
BS_ARR=()
DPB_ARR=()
VEC_ARR=()

TIER="high"

TIER="${TIER:?set TIER to low or high}"


tier_of() {
        local d=$1 k=$2 bs=$3 dpb=$4 vec=$5
    if [[ "$d" == msmarco-v2* || "$d" == miracl-en ]]; then
        echo "high"   # 25M-138M docs: everything needs bigmem
    elif [[ "$d" =~ ^(msmarco|nq|hotpotqa|fever|dbpedia-entity)$ && "$dpb" -ge 1000 ]]; then
        echo "high"
    elif [[ "$vec" == "true" && "$dpb" -ge 250 ]]; then
        echo "high"
    else
        echo "low"
    fi
}

# Helper function to generate the 32 ablations per dataset
add_bins_ablations() {
    local d=$1

    for vec in true false; do
        for k in 10 100; do
            for bs in "${BS_LIST[@]}"; do
                for dpb in 10 25 50 75 100 250 500 1000 1500 2000 2500; do
                    [[ "$(tier_of "$d" "$k" "$bs" "$dpb" "$vec")" == "$TIER" ]] || continue
                    # 25M-138M docs: fewer than 100k bins can only hold a tiny fraction of the corpus, skip them
                    [[ ( "$d" == msmarco-v2* || "$d" == miracl-en ) && "$bs" -lt 100000 ]] && continue
                    DATA_ARR+=("$d"); K_ARR+=("$k"); BS_ARR+=("$bs"); DPB_ARR+=("$dpb"); VEC_ARR+=("$vec")
                done
            done
        done
    done

}

# Build the setups for all datasets. Override to add the new ones, e.g.
#   BINS_DATASETS="nq hotpotqa fever" bash run_bins_high.slurm --count     # prints the --array range to use
#   BINS_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_bins_high.slurm
# Big corpora: BINS_DATASETS="msmarco-v2-25m msmarco-v2-50m msmarco-v2-100m msmarco-v2"
BINS_DATASETS="${BINS_DATASETS:-msmarco scifact}"
for d in $BINS_DATASETS; do
    add_bins_ablations "$d"
done

if [[ "${1:-}" == "--count" ]]; then
    echo "${#DATA_ARR[@]} configs in tier ${TIER} (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"
    exit 0
fi

FAIL_FILE="${SLURM_SUBMIT_DIR:-$PWD}/bins_high_fails"
finished=0

on_exit() {
    local rc=$?
    if [[ "$finished" -ne 1 ]]; then
        echo "${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} rc=${rc} $(date '+%F %T')" >> "$FAIL_FILE"
    fi
}
trap on_exit EXIT
trap 'kill "${srun_pid:-}" 2>/dev/null; exit 143' TERM INT

task_id=$SLURM_ARRAY_TASK_ID

dataset=${DATA_ARR[$task_id]}
k_val=${K_ARR[$task_id]}
bin_size=${BS_ARR[$task_id]}
docs_val=${DPB_ARR[$task_id]}
vec_val=${VEC_ARR[$task_id]}
method="bins"

DB_SIZE=$(db_size_of "$dataset")

if [[ "$vec_val" == "true" ]]; then
    VEC_TAG="_vec1"
else
    VEC_TAG="_vec0"
fi
OUT_DIR="${RESULTS_DIR:-/scratch/$USER/results}/${method}${VEC_TAG}_${dataset}_k${k_val}_bs${bin_size}_dpb${docs_val}"
# OUT_DIR="../results/${method}${VEC_TAG}_${dataset}_k${k_val}_bs${bin_size}_dpb${docs_val}"
mkdir -p "$OUT_DIR"

cat <<EOF > "${OUT_DIR}/slurm_name.txt"
========================================
Array Task ID: $SLURM_ARRAY_TASK_ID
Method:        $method
Dataset:       $dataset
K Value:       $k_val
Bin Size:      $bin_size
Docs/Bin:      $docs_val
Output Dir:    $OUT_DIR
========================================
EOF

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Method:        $method"
echo "Dataset:       $dataset"
echo "K Value:       $k_val"
echo "Bin Size:      $bin_size"
echo "Docs/Bin:      $docs_val"
echo "Output Dir:    $OUT_DIR"
echo "========================================"

srun ./pir_app \
    -n "$DB_SIZE" \
    -t "$method" \
    -name "$dataset" \
    -k "$k_val" \
    -binSize "$bin_size" \
    -docsPerBin "$docs_val" \
    -outDir "$OUT_DIR" \
    -outFile "results" \
    -dataset "$DATASET_DIR" \
    -debug 1 \
    -vectors="$vec_val" \
    -dim 192 &

srun_pid=$!
wait "$srun_pid"

mv "logs/Bins_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
mv "logs/Bins_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true

finished=1
echo "All done for this node."