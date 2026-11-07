#!/bin/bash -l
#SBATCH --job-name=Coeus_Baseline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-08:00:00
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --array=0-47
#SBATCH --output=logs/Coeus_%A_%a.out
#SBATCH --error=logs/Coeus_%A_%a.err
#SBATCH --constraint=amd

# Coeus baseline (see baselines/README.md): exact TF-IDF quality on the real corpus + server time from
# Coeus' own matmult/opt2 benchmark (calibrated once, cached next to the binary) + communication from
# the artifact's encryption parameters.
# Needs `baselines/setup_baselines.sh coeus` first.
#
# Sweep: keyword dictionary size F (multiples of 8192; the paper used 65,536) x tf weighting x k
#   2 datasets x 6 F x 2 tf x 2 k = 48 tasks. Configs with F past the vocabulary exit without writing anything
#   (they'd duplicate a smaller F with extra upload), so scifact produces fewer results.
#
# NOTE: the first task to run calibrates opt2 (4 short runs, the 2x2 one needs ~7 GB RAM). Run task 0 alone
# first (sbatch --array=0 run_coeus.slurm) so the others reuse the cache instead of all calibrating at once.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main

DATA_ARR=(); F_ARR=(); TF_ARR=(); K_ARR=()
for d in scifact msmarco; do
    for f in 8192 16384 32768 65536 131072 262144; do
        for tf in raw log; do
            for k in 10 100; do
                DATA_ARR+=("$d"); F_ARR+=("$f"); TF_ARR+=("$tf"); K_ARR+=("$k")
            done
        done
    done
done

task_id=$SLURM_ARRAY_TASK_ID
dataset=${DATA_ARR[$task_id]}
features=${F_ARR[$task_id]}
tf=${TF_ARR[$task_id]}
k_val=${K_ARR[$task_id]}

OPT2="$HOME/opt/src/Coeus_artifact/matmult/bin/opt2"

echo "Coeus: dataset=$dataset F=$features tf=$tf k=$k_val"
python baselines/coeus/run_coeus.py \
    --dataset ../../datasets \
    --name "$dataset" \
    --k "$k_val" \
    --features "$features" \
    --tf "$tf" \
    --results-root /scratch/dblackle/results \
    --opt2-bin "$OPT2" \
    --parallelism "${SLURM_CPUS_PER_TASK:-1}"
