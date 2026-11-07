#!/bin/bash
#SBATCH --job-name=tree_artifacts
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/tree_artifacts_%j.out
#SBATCH --error=logs/tree_artifacts_%j.err
#
# Lucene index + all tree artifacts for one dataset (see datasets/prep_tree.sh).
#   DATASET=nq sbatch datasets/tree_artifacts.slurm
# MS MARCO v2 (138M): pyserini returns each term's postings as a Python list, so a very common term can take
# ~10 GB per worker. Give it bigmem and fewer workers:
#   DATASET=msmarco-v2 WORKERS=12 sbatch --partition=bigmem --mem=1000G datasets/tree_artifacts.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source datasets/env.sh   # conda env + HF cache (override CONDA_BASE / CONDA_ENV / HF_HOME)

bash datasets/prep_tree.sh "${DATASET:?set DATASET}"
