 #!/bin/bash -l
#SBATCH --job-name=Bins_HighK_MSMARCO
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/Bins_HighK_%A_%a.out
#SBATCH --error=logs/Bins_HighK_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

K_ARR=(500 1000 5000)

task_id=$SLURM_ARRAY_TASK_ID
k_val=${K_ARR[$task_id]}

dataset="msmarco"
bin_size="1.0"
docs_val="1000"
method="bins"
DB_SIZE=8841823

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

cp "logs/Bins_HighK_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
cp "logs/Bins_HighK_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true

echo "All done for this node."