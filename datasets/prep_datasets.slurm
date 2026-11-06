#!/bin/bash
#SBATCH --job-name=rag_prep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/prep_%x_%j.out
#SBATCH --error=logs/prep_%x_%j.err
#
# Download + convert one dataset into ../datasets/<name> (corpus.jsonl, queries.jsonl, qrels/test.tsv, num_docs.txt).
# Submit from the repo root:
#   DATASET=nq          sbatch datasets/prep_datasets.slurm
#   DATASET=hotpotqa    sbatch datasets/prep_datasets.slurm
#   DATASET=fever       sbatch datasets/prep_datasets.slurm
#   DATASET=msmarco-v2  sbatch datasets/prep_datasets.slurm          # ~2-4h, two passes over 138M passages
# Subsamples of MS MARCO v2 (run AFTER msmarco-v2 is prepared AND embedded, they copy its embedding rows):
#   DATASET=msmarco-v2-100m sbatch datasets/prep_datasets.slurm   (also -25m, -50m)
# If compute nodes have no internet, run `bash datasets/download.sh <name> ../datasets/raw` on the login node first.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

DATASET="${DATASET:?set DATASET}"
ROOT="${DATASETS_ROOT:-../datasets}"   # same folder you pass to pir_app -dataset
RAW="${RAW_ROOT:-$ROOT/raw}"
MAX_QUERIES="${MAX_QUERIES:-0}"        # 0 = keep every query that has qrels

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main

case "$DATASET" in
  nq|hotpotqa|fever|trec-covid)
    [[ -d "$RAW/$DATASET" ]] || bash datasets/download.sh "$DATASET" "$RAW"
    python3 datasets/prepare_dataset.py --format beir --src "$RAW/$DATASET" --out "$ROOT/$DATASET" \
        --split test --max-queries "$MAX_QUERIES"
    ;;
  msmarco-v2)
    [[ -d "$RAW/msmarco-v2/msmarco_v2_passage" ]] || bash datasets/download.sh msmarco-v2 "$RAW"
    # dev = passv2 dev1 (3,903 queries). Use SPLIT=dev2 for the other dev set.
    python3 datasets/prepare_dataset.py --format msmarco-v2 --src "$RAW/msmarco-v2" --out "$ROOT/msmarco-v2" \
        --split "${SPLIT:-dev}" --max-queries "$MAX_QUERIES"
    ;;
  msmarco-v2-25m|msmarco-v2-50m|msmarco-v2-100m)
    n="${DATASET##*-}"; n="${n%m}"
    FULL="$ROOT/msmarco-v2"
    python3 datasets/prepare_dataset.py --format beir --src "$FULL" --out "$ROOT/$DATASET" \
        --max-docs "$(( n * 1000000 ))" --max-queries "$MAX_QUERIES" \
        --src-embed "$FULL/msmarco-v2_corpus_embed_f32.npy" \
        --src-query-embed "$FULL/msmarco-v2_query_embed_f32.npy"
    ;;
  *)
    echo "Unknown DATASET $DATASET"; exit 1 ;;
esac

echo "DONE: $ROOT/$DATASET (-n $(cat "$ROOT/$DATASET/num_docs.txt"))"
