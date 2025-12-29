#!/bin/bash
#SBATCH --job-name=post_process
#SBATCH --partition=bigmem
#SBATCH --output=post.out
#SBATCH --error=post.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=0-0:30:00



set -e

source /home/dblackle/miniconda3/etc/profile.d/conda.sh

export CONDA_ENVS_PATH=/scratch/dblackle/conda/envs
export CONDA_PKGS_DIRS=/scratch/dblackle/conda/pkgs

conda activate /scratch/dblackle/conda/envs/main



# conda install -c conda-forge sentencepiece -y

echo "CONDA DONE"


python3 re_rank.py --qrels ../datasets/msmarco/qrels/dev.tsv --step3-output bins_out.tsv --queries ../datasets/msmarco/queries.dev.small.tsv --documents ../datasets/msmarco/collection.tsv --k 100
