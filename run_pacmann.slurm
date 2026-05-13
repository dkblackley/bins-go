#!/bin/bash -l
#SBATCH --job-name=Pacmann_Ablation
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=0-07:30:00

# Total Jobs: 18 (Indices 0 through 17)
#SBATCH --array=0-17
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
# Index:         0       | 1       2       3       | 4          5          | 6       7       8       9       10      11      | 12      13      14      15      16      17
# Test:          BASE    | VARY K                  | VARY DATA             | VARY STEPS                                      | VARY NEIGHBOUR
K_ARR=(          50        100     500     1000      50         50           50      50      50      50      50      50        50      50      50      50      50      50   )
DATA_ARR=(       msmarco   msmarco msmarco msmarco   scifact    trec-covid   msmarco msmarco msmarco msmarco msmarco msmarco   msmarco msmarco msmarco msmarco msmarco msmarco )
STEPS_ARR=(      15        15      15      15        15         15           5       10      15      20      25      30        15      15      15      15      15      15   )
NEIGHB_ARR=(     32        32      32      32        32         32           32      32      32      32      32      32        8       16      24      32      40      48   )

task_id=$SLURM_ARRAY_TASK_ID

k_val=${K_ARR[$task_id]}
dataset=${DATA_ARR[$task_id]}
steps_val=${STEPS_ARR[$task_id]}
neighb_val=${NEIGHB_ARR[$task_id]}
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

OUT_DIR="../results/${method}_${dataset}_k${k_val}_steps${steps_val}_neighb${neighb_val}"
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Method:        $method"
echo "Dataset:       $dataset"
echo "K Value:       $k_val"
echo "Steps:         $steps_val"
echo "Neighbours:    $neighb_val"
echo "Output Dir:    $OUT_DIR"
echo "========================================"

# Execute binary
srun ./pir_app \
    -n "$DB_SIZE" \
    -t "$method" \
    -name "$dataset" \
    -k "$k_val" \
    -steps "$steps_val" \
    -neighb "$neighb_val" \
    -outDir "$OUT_DIR" \
    -outFile "results" \
    -dataset "../datasets" \
    -debug 1 \
    -dim 192

cp "logs/Pacmann_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
cp "logs/Pacmann_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true

echo "All done for this node."