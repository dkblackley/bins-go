#!/bin/bash
# Download a raw dataset into <raw_dir>/<name>. Run on a node with internet access (login node is fine).
#
#   bash datasets/download.sh nq         ../datasets/raw
#   bash datasets/download.sh hotpotqa   ../datasets/raw
#   bash datasets/download.sh fever      ../datasets/raw
#   bash datasets/download.sh trec-covid ../datasets/raw
#   bash datasets/download.sh dbpedia-entity ../datasets/raw
#   bash datasets/download.sh msmarco-v2 ../datasets/raw     # ~20 GB tar (138,364,198 passages)
#   bash datasets/download.sh miracl-en  ../datasets/raw     # 32.9M Wikipedia passages, needs huggingface-cli
#
# Then convert with datasets/prepare_dataset.py (datasets/prep_datasets.slurm does both).
set -euo pipefail

name="${1:?dataset name}"
raw="${2:-../datasets/raw}"
mkdir -p "$raw"
cd "$raw"

BEIR_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
MARCO_URL="https://msmarco.z22.web.core.windows.net/msmarcoranking"

case "$name" in
  nq|hotpotqa|fever|trec-covid|dbpedia-entity|climate-fever|fiqa|nfcorpus|scifact)
    # BEIR zip -> <name>/corpus.jsonl, queries.jsonl, qrels/{train,dev,test}.tsv
    # If the TU Darmstadt mirror is down, the same files are on HF: huggingface-cli download BeIR/<name> (and BeIR/<name>-qrels)
    wget -c -O "${name}.zip" "${BEIR_URL}/${name}.zip"
    unzip -n -q "${name}.zip"
    ;;
  msmarco-v2)
    mkdir -p msmarco-v2 && cd msmarco-v2
    wget -c -O msmarco_v2_passage.tar "${MARCO_URL}/msmarco_v2_passage.tar"
    tar -xf msmarco_v2_passage.tar          # -> msmarco_v2_passage/msmarco_passage_{00..69}.gz (keep them gzipped)
    for f in passv2_dev_queries.tsv passv2_dev_qrels.tsv passv2_dev2_queries.tsv passv2_dev2_qrels.tsv; do
      wget -c -O "$f" "${MARCO_URL}/$f"
    done
    ;;
  miracl-en)
    # English MIRACL: Wikipedia passages + human-judged dev queries (https://huggingface.co/datasets/miracl/miracl)
    huggingface-cli download miracl/miracl-corpus --repo-type dataset --include "miracl-corpus-v1.0-en/*" \
        --local-dir miracl-en
    huggingface-cli download miracl/miracl --repo-type dataset \
        --include "miracl-v1.0-en/topics/*" "miracl-v1.0-en/qrels/*" --local-dir miracl-en
    ;;
  *)
    echo "Unknown dataset $name"; exit 1 ;;
esac
echo "Downloaded $name into $raw"
