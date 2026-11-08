#!/bin/bash
# Run the whole pipeline locally (no SLURM) for the smaller new datasets, to check and debug before the server.
# Run from the bins-go repo root:
#   bash datasets/run_local.sh nq dbpedia-entity
#   STAGES="embed tree-db run" bash datasets/run_local.sh nq      # skip stages that are already done
#
# Stages (default: all): prep (download + convert), embed (e5 + PCA on the local GPU), tree-db (Lucene index +
# tree artifacts for one B/r/s), run (one bins, one pacmann and one tree config, written to $RESULTS_DIR, default
# <datasets>/results, with the same folder names as the slurm runs), baselines (one Tiptoe and one Coeus config,
# same folder names; Tiptoe is real if ./pir_app_tiptoe is built, otherwise simulated).
# Each stage is skipped automatically if its output already exists (FORCE=1 redoes it).
#
# Needs: python with requirements.txt (+ faiss, pyserini), Java 21, Go + the NGT/hnsw libs (see run_scifact_local.sh).
# A 16 GB GPU is plenty for e5-base; FP16=1 roughly halves embedding time (the server runs fp32 for these sets).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${DATASETS_ROOT:-../../../datasets}"
RAW="${RAW_ROOT:-$ROOT/raw}"
# Same folder + naming as the slurm runs, so plotting/load_results.py picks local results up too
OUT="${RESULTS_DIR:-$ROOT/results}"
STAGES="${STAGES:-prep embed tree-db run baselines}"
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
        --num-docs "$N" --encode-batch "${ENCODE_BATCH:-128}" ${FP16:+--fp16} --skip-if-done ${FORCE:+--overwrite}
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
    BS="${BINS_BS:-100000}"; DPB="${BINS_DPB:-100}"; L="${TREE_L:-100}"; Q="${QUERY_WORDS:-8}"
    OB="$OUT/bins_vec1_${d}_k${K}_bs${BS}_dpb${DPB}"
    OP="$OUT/pacmann_${d}_k${K}_steps15_neighb32"
    OT="$OUT/tree_grid_${d}_b${B}_r${R}_s${S}_L${L}_q${Q}_k${K}"
    mkdir -p "$OB" "$OP" "$OT"
    if [[ ! -d "$ROOT/index_${d}" || -n "${FORCE:-}" ]]; then
      echo "-- bins index"
      ./pir_app -t index -name "$d" -dataset "$ROOT"
    fi
    echo "-- bins"
    ./pir_app -n "$N" -t bins -name "$d" -k "$K" -binSize "$BS" -docsPerBin "$DPB" \
        -vectors=true -dim 192 -dataset "$ROOT" -outDir "$OB" -outFile results -debug 1
    echo "-- pacmann"
    ./pir_app -n "$N" -t pacmann -name "$d" -k "$K" -steps 15 -neighb 32 -dim 192 \
        -dataset "$ROOT" -outDir "$OP" -outFile results -debug 1
    echo "-- tree"
    ./pir_app -n "$N" -t tree -name "$d" -k "$K" -dim 192 -dataset "$ROOT" -outDir "$OT" -outFile results \
        -debug 0 -B "$B" -r "$R" -s "$S" -L "$L" -k_candidates 128 -query-words "$Q" \
        -max-queries 100000 -k1 0.9 -b 0.4 -scale 10 -precompute -pir-batch-size 128 -pir-threads "$(nproc)" \
        -cuckoo-load-factor 0.4 -cuckoo-max-kicks 1000 -cuckoo-build-attempts 8 \
        -index "$T/index_${d}" -layout-path "$A/layout.json" -vocab "$A/stage1_vocab.json" \
        -stage1-data-bin "$A/stage1_data.bin" -stage1-idmap-bin "$A/stage1_idmap.bin" \
        -stage2-data-bin "$A/stage2/data.bin" -stage2-idmap-bin "$A/stage2/idmap.bin" \
        -queries "$T/queries.qrels.tsv" -qrels "$T/qrels.trec" \
        -doc-embed "$T/stage3_doc.npy" -doc-id-map "$T/stage3_doc.npy.ids" \
        -query-embed "$D/${d}_query_embed_f32.npy" -embed-dim 192
    echo "Results: $OB, $OP, $OT (metadata.json in each)"
  fi

  # ---------------- Tiptoe + Coeus (baselines/README.md), Tiptoe's and Coeus' paper defaults
  if has baselines; then
    DPC="${TT_DPC:-2500}"
    if [[ -x ./pir_app_tiptoe ]]; then FMT=txt; else FMT=npy; fi
    TT="$ROOT/tiptoe/${d}_dpc${DPC}_m2_b160_${FMT}"
    if [[ ! -f "$TT/tiptoe_meta.json" || -n "${FORCE:-}" ]]; then
      echo "-- tiptoe preprocessing"
      python3 baselines/tiptoe/preprocess_tiptoe.py --dataset "$ROOT" --name "$d" --docs-per-cluster "$DPC" \
          --multi-assign 2 --bundle-size 160 --format "$FMT" --out "$TT"
    fi
    if [[ "$FMT" == txt ]]; then
      echo "-- tiptoe (real)"
      read -r C MA < <(python3 -c "import json;m=json.load(open('$TT/tiptoe_meta.json'));print(m['num_clusters'],m['multi_assign'])")
      OTT="$OUT/tiptoe_${d}_c${C}_m${MA}_b160_h500_k${K}"
      mkdir -p "$OTT"
      ./pir_app_tiptoe -n "$N" -t tiptoe -name "$d" -k "$K" -dim 192 -dataset "$ROOT" -tiptoe-dir "$TT" \
          -tiptoe-emb-hint-mb 500 -tiptoe-max-queries "${TT_MAX_QUERIES:-1000}" -outDir "$OTT" -outFile results -debug 0
    else
      echo "-- tiptoe (simulated: no ./pir_app_tiptoe)"
      go build -o simplepir_bench ./cmd/simplepir_bench
      python3 baselines/tiptoe/simulate_tiptoe.py --dataset "$ROOT" --name "$d" --tiptoe-dir "$TT" --k "$K" \
          --emb-hint-mb 500 --results-root "$OUT" --bench-bin ./simplepir_bench --parallelism "$(nproc)"
    fi
    echo "-- coeus"
    OPT2="$HOME/opt/src/Coeus_artifact/matmult/bin/opt2"
    OPT2_ARG=(); [[ -x "$OPT2" ]] && OPT2_ARG=(--opt2-bin "$OPT2" --parallelism "$(nproc)")
    python3 baselines/coeus/run_coeus.py --dataset "$ROOT" --name "$d" --k "$K" --features "${COEUS_F:-65536}" \
        --tf raw --results-root "$OUT" ${OPT2_ARG[@]+"${OPT2_ARG[@]}"}
  fi
done
