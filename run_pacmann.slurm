#!/bin/bash -l
#SBATCH --job-name=Pacmann_Ablation
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=750G
#SBATCH --time=2-06:30:00

# Total Jobs: 6 (Indices 0 through 5)
#SBATCH --array=0-5
#SBATCH --output=logs/Pacmann_%A_%a.out
#SBATCH --error=logs/Pacmann_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

# --- PARALLEL ARRAYS FOR ABLATION STUDY ---
# Index:        0       1       2       3        4          5
K_ARR=(         50      100     500     1000     50         50         )
DATA_ARR=(      msmarco msmarco msmarco msmarco  scifact    trec-covid )

task_id=$SLURM_ARRAY_TASK_ID

k_val=${K_ARR[$task_id]}
dataset=${DATA_ARR[$task_id]}
method="pacmann"

# Set specific variables based on chosen dataset
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

OUT_DIR="../results/${method}_${dataset}_k${k_val}"
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Method:        $method"
echo "Dataset:       $dataset"
echo "K Value:       $k_val"
echo "Output Dir:    $OUT_DIR"
echo "========================================"

# Execute binary (no bin flags needed)
srun ./pir_app \
    -n "$DB_SIZE" \
    -t "$method" \
    -name "$dataset" \
    -k "$k_val" \
    -outDir "$OUT_DIR" \
    -outFile "results" \
    -dataset "../datasets" \
    -debug 1 \
    -save

cp "logs/Pacmann_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
cp "logs/Pacmann_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true

echo "All done for this node."