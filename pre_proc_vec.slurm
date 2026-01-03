#!/bin/bash
#SBATCH --job-name=rag_eval
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=1-6:30:00



set -e

source /home/dblackle/miniconda3/etc/profile.d/conda.sh

#export CONDA_ENVS_PATH=/scratch/dblackle/conda/envs
#export CONDA_PKGS_DIRS=/scratch/dblackle/conda/pkgs
#
#
## # If things go wrong with conda:
## conda deactivate 2>/dev/null || true
## conda deactivate
## rm -rf /scratch/dblackle/conda/envs/rag_eval
## rm -f  /scratch/dblackle/conda/pkgs/*
#
#
## conda create -y -p /scratch/dblackle/conda/envs/rag_eval python=3.10 pip
#conda activate /scratch/dblackle/conda/envs/rag_eval

# Example installs
#python -m pip install --upgrade pip
#pip install -r requirements.txt


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

conda activate main
echo "CONDA DONE"

# Optional but recommended: keep HF/transformers cache off $HOME
export HF_HOME=/scratch/dblackle/hf
export TRANSFORMERS_CACHE=/scratch/dblackle/hf/transformers
export HF_DATASETS_CACHE=/scratch/dblackle/hf/datasets
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"



python collection_preproc.py \
    --documents "../datasets/msmarco/collection.tsv" \
    --output "../datasets/Son/local_collection.npy"

python query_preproc.py \
    --documents "../datasets/msmarco/query.dev.small.tsv" \
    --output "../datasets/Son/local_query.npy"


echo "DONE"