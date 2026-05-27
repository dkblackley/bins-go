#!/bin/bash -l
#SBATCH --job-name=Bins_Ablation
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=0-02:30:00

# Total Jobs: 13 (Indices 0 through 12)
#SBATCH --array=0-33
#SBATCH --output=logs/Bins_%A_%a.out
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

# --- PARALLEL ARRAYS FOR ABLATION STUDY ---
# The columns line up exactly to define 13 specific job configurations.
# Variable being tested is grouped.

# 1. BASE, VARY K, and VARY DATA
# Index:    0       | 1       2       3       4       5       | 6          7
# Test:     BASE    | VARY K                                  | VARY DATA
K_ARR=(     50        10      100     500     1000    5000      100         100           )
DATA_ARR=(  msmarco   msmarco msmarco msmarco msmarco msmarco   scifact    trec-covid   )
BS_ARR=(    0.1       0.1     0.1     0.1     0.1     0.1       0.1        0.1          )
DPB_ARR=(   1000      1000    1000    1000    1000    1000      1000       1000         )

# 2. VARY BIN SIZE (Expanding BS=1.0 and 0.001 across K = 10, 50, 100, 500, 1000, 5000)
# Index:    8       9       | 10      11      | 12      13      | 14      15      | 16      17      | 18      19
# K Value:  K=10            | K=50            | K=100           | K=500           | K=1000          | K=5000
K_ARR+=(    10      10        50      50        100     100       500     500       1000    1000      5000    5000     )
DATA_ARR+=( msmarco msmarco   msmarco msmarco   msmarco msmarco   msmarco msmarco   msmarco msmarco   msmarco msmarco  )
BS_ARR+=(   1.0     0.001     1.0     0.001     1.0     0.001     1.0     0.001     1.0     0.001     1.0     0.001    )
DPB_ARR+=(  1000    1000      1000    1000      1000    1000      1000    1000      1000    1000      1000    1000     )

# 3. VARY DOCS PER BIN (Expanding DPB=10 and 50 across K variations) + (Original DPB=100, 500, 5000 for K=50)
# Index:    20      21      | 22      23      | 24      25      | 26      27      | 28      29      | 30      31      | 32      33      34
# K Value:  K=10            | K=50            | K=100           | K=500           | K=1000          | K=5000          | Orig DPB (K=50)
K_ARR+=(    10      10        50      50        100     100       500     500       1000    1000      5000    5000      50      50      50   )
DATA_ARR+=( msmarco msmarco   msmarco msmarco   msmarco msmarco   msmarco msmarco   msmarco msmarco   msmarco msmarco   msmarco msmarco msmarco )
BS_ARR+=(   0.1     0.1       0.1     0.1       0.1     0.1       0.1     0.1       0.1     0.1       0.1     0.1       0.1     0.1     0.1  )
DPB_ARR+=(  10      50        10      50        10      50        10      50        10      50        10      50        100     500     5000 )


task_id=$SLURM_ARRAY_TASK_ID

k_val=${K_ARR[$task_id]}
dataset=${DATA_ARR[$task_id]}
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