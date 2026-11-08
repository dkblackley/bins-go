#!/bin/bash
#SBATCH --job-name=bins_index
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/bins_index_%j.out
#SBATCH --error=logs/bins_index_%j.err
#
# Build the Bluge index the bins method needs (<DATASETS_ROOT>/index_<name>) from corpus.jsonl. Run once per
# dataset, after prep and before any bins job:
#   DATASET=nq sbatch datasets/bins_index.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
ROOT="${DATASETS_ROOT:-../datasets}"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-8}"
./pir_app -t index -name "${DATASET:?set DATASET}" -dataset "$ROOT"
