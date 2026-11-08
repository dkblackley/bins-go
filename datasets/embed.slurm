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
#
# The header asks for a DGX A100.40gb (16 GPUs, often busy for days). To queue the same jobs on other GPUs instead
# (A100.80gb on gpuq+contrib-gpuq, MIG slices, H100, B200) use GPU=<profile> with run_all.sh / build_all.sh; see
# datasets/README.md "Racing two GPU queues". Jobs are safe to race: the first to finish an output writes <out>.done,
# any other copy is thrown away and its job exits 3 so its dependants never start. Interrupted jobs resume.
#   EMBED_FP16=1   encode in fp16 (default only for the sharded 100M+ corpora; the GPU= profiles turn it on)
#   OVERWRITE=1    redo an output that is already done
#   EMBED_TAG=x    resume key for the partial file (run_all.sh sets it per series; default the job id)

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

source datasets/env.sh   # conda env + HF cache (override CONDA_BASE / CONDA_ENV / HF_HOME)

# fp16 by default only for the 100M+ corpus (EMBED_FP16=1 for all): ~2x faster than TF32, the PCA'd vectors are
# essentially unchanged
FP16=""
[[ "$NUM_SHARDS" -gt 1 || "${EMBED_FP16:-0}" == 1 ]] && FP16="--fp16"
OVR=""
[[ -n "${OVERWRITE:-}" ]] && OVR="--overwrite"
echo "embed: $DATASET mode=$MODE host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) ${FP16:-fp32/tf32} tag=${EMBED_TAG:-job}" >&2

case "$MODE" in
  all)
    python collection_preproc.py --documents "$D/corpus.jsonl" --output "$CORPUS_OUT" --pca "$PCA" \
        --num-docs "$NUM_DOCS" --encode-batch 256 $FP16 $OVR
    python query_preproc.py --queries "$D/queries.jsonl" --output "$QUERY_OUT" --pca "$PCA"
    ;;
  pca)
    python collection_preproc.py --documents "$D/corpus.jsonl" --output "$CORPUS_OUT" --pca "$PCA" --pca-only $OVR
    ;;
  shard)
    python collection_preproc.py --documents "$D/corpus.jsonl" --output "$CORPUS_OUT" --pca "$PCA" --load-pca \
        --num-docs "$NUM_DOCS" --num-shards "$NUM_SHARDS" --shard-id "${SLURM_ARRAY_TASK_ID:?run as an array}" \
        --encode-batch 512 $FP16 $OVR
    ;;
  finish)
    python datasets/merge_embed_shards.py --output "$CORPUS_OUT" --num-shards "$NUM_SHARDS" --delete-shards
    python query_preproc.py --queries "$D/queries.jsonl" --output "$QUERY_OUT" --pca "$PCA"
    ;;
  *)
    echo "Unknown MODE $MODE"; exit 1 ;;
esac
echo "DONE $MODE for $DATASET"
