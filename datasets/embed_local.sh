#!/bin/bash
# Download, prepare and embed datasets on your own machine (no SLURM), storing the corpus embeddings as float16 to
# halve the disk. Then copy them to the server and widen them to float32 there (datasets/import_local_embeds.sh).
# Run from the repo root:
#   bash datasets/embed_local.sh nq hotpotqa fever dbpedia-entity       # ~1h each on a 16 GB RTX card
#   bash datasets/embed_local.sh msmarco-v2                             # 138M passages: ~1.5-2 days, resumable
#   NUM_SHARDS=24 SHARDS="20 21 22 23" bash datasets/embed_local.sh msmarco-v2   # only some shards (see README)
#   STAGES=embed bash datasets/embed_local.sh nq                        # skip the download/prepare
#
# Uses ../../../datasets (DATASETS_ROOT=... to change). Everything resumes: re-run the same command after a crash,
# reboot or Ctrl-C and it carries on from the last checkpoint (~every 100k docs). FP16=0 encodes in fp32 (slower,
# same file). Only the corpus file is float16; queries stay float32 (tiny). Writes per dataset <name>/:
#   <name>_corpus_embed_f16.npy(.ids)   <name>_query_embed_f32.npy   pca_e5_768_to_192.faiss
# and prints the rsync command that copies exactly those to the server.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${DATASETS_ROOT:-../../../datasets}"
RAW="${RAW_ROOT:-$ROOT/raw}"
STAGES="${STAGES:-prep embed}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
SERVER_ROOT="${SERVER_ROOT:-hopper:<scratch repo>/../datasets}"   # only used to print the rsync line (set it once)
export DELETE_TAR=1        # drop the downloaded .tar/.zip once extracted (msmarco-v2: -20 GB)
export EMBED_TAG=local     # resume key for the partial files
[[ $# -gt 0 ]] || { sed -n 2,15p "$0"; exit 1; }
python=python3; command -v python3 >/dev/null || python=python

for d in "$@"; do
  D="$ROOT/$d"
  echo "======== $d ($D)"

  if has prep && [[ ! -f "$D/num_docs.txt" ]]; then
    case "$d" in
      nq|hotpotqa|fever|trec-covid|dbpedia-entity)
        [[ -d "$RAW/$d" ]] || bash datasets/download.sh "$d" "$RAW"
        $python datasets/prepare_dataset.py --format beir --src "$RAW/$d" --out "$D" --split test ;;
      msmarco-v2)   # same split/arguments as prep_datasets.slurm, so the corpus is row-for-row the server's
        [[ -f "$RAW/msmarco-v2/msmarco_v2_passage/msmarco_passage_69.gz" ]] || bash datasets/download.sh msmarco-v2 "$RAW"
        $python datasets/prepare_dataset.py --format msmarco-v2 --src "$RAW/msmarco-v2" --out "$D" --split "${SPLIT:-dev}" ;;
      miracl-en)
        [[ -d "$RAW/miracl-en" ]] || bash datasets/download.sh miracl-en "$RAW"
        $python datasets/prepare_dataset.py --format miracl --src "$RAW/miracl-en" --out "$D" --split "${SPLIT:-dev}" ;;
      *) echo "embed_local.sh doesn't prepare $d (subsamples like msmarco-v2-25m are made on the server)"; exit 1 ;;
    esac
  fi
  N=$(cat "$D/num_docs.txt")
  prefix="$d"; [[ "$d" == trec-covid ]] && prefix=trec
  OUT="$D/${prefix}_corpus_embed_f16.npy"
  PCA="$D/pca_e5_768_to_192.faiss"
  common=(--documents "$D/corpus.jsonl" --output "$OUT" --pca "$PCA" --num-docs "$N" --dtype float16
          --encode-batch "${ENCODE_BATCH:-128}" --skip-if-done)
  [[ "${FP16:-1}" != 0 ]] && common+=(--fp16)

  if has embed; then
    if [[ -n "${SHARDS:-}" ]]; then
      [[ -f "$PCA" ]] || { echo "Shards must use the server's PCA: copy <server>/datasets/$d/pca_e5_768_to_192.faiss to $D/ first"; exit 1; }
      for i in $SHARDS; do
        $python collection_preproc.py "${common[@]}" --load-pca --num-shards "${NUM_SHARDS:?set NUM_SHARDS (server uses 24 for msmarco-v2)}" --shard-id "$i"
      done
      files="$OUT.shard*  $PCA"
    else
      $python collection_preproc.py "${common[@]}"
      [[ -f "$D/${prefix}_query_embed_f32.npy" && -z "${FORCE:-}" ]] || \
        $python query_preproc.py --queries "$D/queries.jsonl" --output "$D/${prefix}_query_embed_f32.npy" --pca "$PCA"
      files="$OUT $OUT.ids $D/${prefix}_query_embed_f32.npy $PCA"
    fi
    echo "-- $d done. Copy to the server, then run there: bash datasets/import_local_embeds.sh $d"
    echo "   rsync -avP --exclude='*.done' $files $SERVER_ROOT/$d/local_import/"
  fi
done
