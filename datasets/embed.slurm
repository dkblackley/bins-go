#!/bin/bash
#SBATCH --job-name=rag_embed
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/embed_%x_%A_%a.out
#SBATCH --error=logs/embed_%x_%A_%a.err
#
# e5-base-v2 -> PCA 192 embeddings for a dataset prepared by prep_datasets.slurm (same model/PCA as msmarco/scifact).
#
# Small/medium corpora (NQ 2.7M, HotpotQA 5.2M, FEVER 5.4M: roughly 2-6h on one A100), one job does everything:
#   DATASET=nq sbatch datasets/embed.slurm
# Big corpora (MS MARCO v2, 138M) are sharded; use datasets/embed_big.sh which chains
#   MODE=pca -> MODE=shard (array, one shard per task) -> MODE=finish (merge shards + embed queries)
#
# Output: ../datasets/<name>/<name>_corpus_embed_f32.npy(.ids), <name>_query_embed_f32.npy, pca_e5_768_to_192.faiss

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

DATASET="${DATASET:?set DATASET}"
MODE="${MODE:-all}"
NUM_SHARDS="${NUM_SHARDS:-1}"
ROOT="${DATASETS_ROOT:-../datasets}"
D="$ROOT/$DATASET"

# main.go's file names: trec-covid predates the generic <name>_ prefix
PREFIX="$DATASET"
[[ "$DATASET" == "trec-covid" ]] && PREFIX="trec"
CORPUS_OUT="$D/${PREFIX}_corpus_embed_f32.npy"
QUERY_OUT="$D/${PREFIX}_query_embed_f32.npy"
PCA="$D/pca_e5_768_to_192.faiss"
NUM_DOCS="$(cat "$D/num_docs.txt")"

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main
export HF_HOME=/scratch/dblackle/hf
export TRANSFORMERS_CACHE=/scratch/dblackle/hf/transformers
export HF_DATASETS_CACHE=/scratch/dblackle/hf/datasets
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# fp16 only for the 100M+ corpus: halves encode time, the PCA'd vectors are essentially unchanged
FP16=""
[[ "$NUM_SHARDS" -gt 1 ]] && FP16="--fp16"

case "$MODE" in
  all)
    python collection_preproc.py --documents "$D/corpus.jsonl" --output "$CORPUS_OUT" --pca "$PCA" \
        --num-docs "$NUM_DOCS" --encode-batch 256
    python query_preproc.py --queries "$D/queries.jsonl" --output "$QUERY_OUT" --pca "$PCA"
    ;;
  pca)
    python collection_preproc.py --documents "$D/corpus.jsonl" --output "$CORPUS_OUT" --pca "$PCA" --pca-only
    ;;
  shard)
    python collection_preproc.py --documents "$D/corpus.jsonl" --output "$CORPUS_OUT" --pca "$PCA" --load-pca \
        --num-docs "$NUM_DOCS" --num-shards "$NUM_SHARDS" --shard-id "${SLURM_ARRAY_TASK_ID:?run as an array}" \
        --encode-batch 512 $FP16
    ;;
  finish)
    python datasets/merge_embed_shards.py --output "$CORPUS_OUT" --num-shards "$NUM_SHARDS" --delete-shards
    python query_preproc.py --queries "$D/queries.jsonl" --output "$QUERY_OUT" --pca "$PCA"
    ;;
  *)
    echo "Unknown MODE $MODE"; exit 1 ;;
esac
echo "DONE $MODE for $DATASET"
