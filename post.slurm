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

# conda create -y -p /scratch/dblackle/conda/envs/rag_eval python=3.10 pip
conda activate /scratch/dblackle/conda/envs/rag_eval

# Example installs
# python -m pip install --upgrade pip
# pip install -r requirements.txt


# wget -c -O collection.tar.gz "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
# tar -xzf collection.tar.gz

# # 2) Dev-small queries (qid \t text)
# wget -c -O queries.dev.small.tsv \
#   "https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/queries.dev.small.tsv"


# wget -c -O qrels.dev.tsv \
    #   "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv"

# wget -c -O qrels.dev.tsv "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv"


# wget -c -O qrels.dev.small.tsv \
#   "https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/qrels.dev.small.tsv"

# conda install -c conda-forge sentencepiece -y

echo "CONDA DONE"


python3 re_rank.py --qrels ../datasets/msmarco/qrels/dev.tsv --step3-output bins_out.tsv --queries ../datasets/msmarco/queries.dev.small.tsv --documents ../datasets/msmarco/collection.tsv --k 100
