#!/bin/bash -l
#SBATCH --job-name=Bins_Ablation
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=0-02:30:00

# Total Jobs: 13 (Indices 0 through 12)
#SBATCH --array=0-95
# SBATCH --output=logs/Bins_%A_%a.out
#SBATCH --error=logs/Bins_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

DATA_ARR=()
K_ARR=()
BS_ARR=()
DPB_ARR=()

# Helper function to generate the 32 ablations per dataset
add_bins_ablations() {
    local d=$1

    # 1. VARY K (BS=1.0, DPB=1000) -> 6 combinations
    for k in 10 50 100 500 1000 5000; do
        DATA_ARR+=("$d"); K_ARR+=("$k"); BS_ARR+=("1.0"); DPB_ARR+=("1000")
    done

    # 2. VARY BIN SIZE across K=10 and K=100 (DPB=1000) -> 14 combinations
    for k in 10 100; do
        for bs in 2.0 1.5 1.0 0.5 0.1 0.05 0.01; do
            DATA_ARR+=("$d"); K_ARR+=("$k"); BS_ARR+=("$bs"); DPB_ARR+=("1000")
        done
    done

    # 3. VARY DOCS PER BIN across K=10 and K=100 (BS=1.0) -> 12 combinations
    for k in 10 100; do
        for dpb in 10 50 100 500 1000 5000; do
            DATA_ARR+=("$d"); K_ARR+=("$k"); BS_ARR+=("1.0"); DPB_ARR+=("$dpb")
        done
    done
}

# Build the setups for all datasets
add_bins_ablations "msmarco"
add_bins_ablations "scifact"
add_bins_ablations "trec-covid"

task_id=$SLURM_ARRAY_TASK_ID

dataset=${DATA_ARR[$task_id]}
k_val=${K_ARR[$task_id]}
bin_size=${BS_ARR[$task_id]}
docs_val=${DPB_ARR[$task_id]}
method="bins"

if [ "$dataset" == "msmarco" ]; then
    DB_SIZE=8841823
elif [ "$dataset" == "scifact" ]; then
    DB_SIZE=5183
elif [ "$dataset" == "trec-covid" ]; then
    DB_SIZE=171332
else
    echo "Unknown dataset: $dataset"
    exit 1
fi

OUT_DIR="../results/${method}_${dataset}_k${k_val}_bs${bin_size}_dpb${docs_val}"
mkdir -p "$OUT_DIR"

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
    -dataset "../datasets" \
    -debug 1 \
    -dim 192


cp "logs/Bins_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
cp "logs/Bins_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true

echo "All done for this node."