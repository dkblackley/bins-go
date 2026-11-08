#!/bin/bash
# Run the whole pipeline locally (no SLURM) for the smaller new datasets, to check and debug before the server.
# Run from the bins-go repo root:
#   bash datasets/run_local.sh nq dbpedia-entity
#   STAGES="embed tree-db run" bash datasets/run_local.sh nq      # skip stages that are already done
#
# Stages (default: all): prep (download + convert), embed (e5 + PCA on the local GPU), tree-db (Lucene index +
# tree artifacts for one B/r/s), run (one bins, one pacmann and one tree config -> results_local/<name>/).
# Each stage is skipped automatically if its output already exists (FORCE=1 redoes it).
#
# Needs: python with requirements.txt (+ faiss, pyserini), Java 21, Go + the NGT/hnsw libs (see run_scifact_local.sh).
# A 16 GB GPU is plenty for e5-base; FP16=1 roughly halves embedding time (the server runs fp32 for these sets).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${DATASETS_ROOT:-../../../datasets}"
RAW="${RAW_ROOT:-$ROOT/raw}"
OUT="${OUT_ROOT:-results_local}"
STAGES="${STAGES:-prep embed tree-db run}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
# one tree shape for the local check (the server builds the whole grid)
B="${TREE_B:-32}"; R="${TREE_R:-8}"; S="${TREE_S:-4}"
K="${K:-10}"

[[ $# -gt 0 ]] || { sed -n 2,13p "$0"; exit 1; }

if has run; then
  [[ -f "$HOME/opt/pacmann_env.sh" ]] && source "$HOME/opt/pacmann_env.sh"
  go build -o pir_app ./main.go
fi

for d in "$@"; do
  D="$ROOT/$d"
  echo "======== $d ($D)"

  # ---------------- prep
  if has prep && { [[ -n "${FORCE:-}" ]] || [[ ! -f "$D/num_docs.txt" ]]; }; then
    case "$d" in
      nq|hotpotqa|fever|trec-covid|dbpedia-entity)
        [[ -d "$RAW/$d" ]] || bash datasets/download.sh "$d" "$RAW"
        python3 datasets/prepare_dataset.py --format beir --src "$RAW/$d" --out "$D" --split test ;;
      *) echo "run_local.sh only handles the BEIR sets; use the slurm scripts for $d"; exit 1 ;;
    esac
  fi
  python3 datasets/query_stats.py "$D"
  N=$(cat "$D/num_docs.txt")

  # ---------------- embed
  PCA="$D/pca_e5_768_to_192.faiss"
  if has embed && { [[ -n "${FORCE:-}" ]] || [[ ! -f "$D/${d}_query_embed_f32.npy" ]]; }; then
    python3 collection_preproc.py --documents "$D/corpus.jsonl" --output "$D/${d}_corpus_embed_f32.npy" --pca "$PCA" \
        --num-docs "$N" --encode-batch "${ENCODE_BATCH:-128}" ${FP16:+--fp16}
    python3 query_preproc.py --queries "$D/queries.jsonl" --output "$D/${d}_query_embed_f32.npy" --pca "$PCA"
  fi

  # ---------------- tree artifacts (one B/r/s)
  T="data/${d}_tree"; A="$T/B${B}_r${R}_s${S}"
  if has tree-db && { [[ -n "${FORCE:-}" ]] || [[ ! -f "$A/stage2/idmap.bin" ]]; }; then
    DATASETS_ROOT="$ROOT" TREE_B="$B" TREE_R="$R" TREE_S="$S" WORKERS="${WORKERS:-$(nproc)}" \
        bash datasets/prep_tree.sh "$d"
  fi

  # ---------------- one config per method
  if has run; then
    mkdir -p "$OUT/$d/bins" "$OUT/$d/pacmann" "$OUT/$d/tree"
    if [[ ! -d "$ROOT/index_${d}" || -n "${FORCE:-}" ]]; then
      echo "-- bins index"
      ./pir_app -t index -name "$d" -dataset "$ROOT"
    fi
    echo "-- bins"
    ./pir_app -n "$N" -t bins -name "$d" -k "$K" -binSize "${BINS_BS:-100000}" -docsPerBin "${BINS_DPB:-100}" \
        -vectors=true -dim 192 -dataset "$ROOT" -outDir "$OUT/$d/bins" -outFile results -debug 1
    echo "-- pacmann"
    ./pir_app -n "$N" -t pacmann -name "$d" -k "$K" -steps 15 -neighb 32 -dim 192 \
        -dataset "$ROOT" -outDir "$OUT/$d/pacmann" -outFile results -debug 1
    echo "-- tree"
    ./pir_app -n "$N" -t tree -name "$d" -k "$K" -dim 192 -dataset "$ROOT" -outDir "$OUT/$d/tree" -outFile results \
        -debug 0 -B "$B" -r "$R" -s "$S" -L "${TREE_L:-100}" -k_candidates 128 -query-words "${QUERY_WORDS:-8}" \
        -max-queries 100000 -k1 0.9 -b 0.4 -scale 10 -precompute -pir-batch-size 128 -pir-threads "$(nproc)" \
        -cuckoo-load-factor 0.4 -cuckoo-max-kicks 1000 -cuckoo-build-attempts 8 \
        -index "$T/index_${d}" -layout-path "$A/layout.json" -vocab "$A/stage1_vocab.json" \
        -stage1-data-bin "$A/stage1_data.bin" -stage1-idmap-bin "$A/stage1_idmap.bin" \
        -stage2-data-bin "$A/stage2/data.bin" -stage2-idmap-bin "$A/stage2/idmap.bin" \
        -queries "$T/queries.qrels.tsv" -qrels "$T/qrels.trec" \
        -doc-embed "$T/stage3_doc.npy" -doc-id-map "$T/stage3_doc.npy.ids" \
        -query-embed "$D/${d}_query_embed_f32.npy" -embed-dim 192
    echo "Results: $OUT/$d/{bins,pacmann,tree}/metadata.json"
  fi
done
