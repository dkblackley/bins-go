#!/bin/bash
# Build the tree method's artifacts for a dataset.
#   bash datasets/prep_tree.sh nq                   # every (B, r, s) the tree sweep needs (tree_sweep_grid.sh, SWEEP=all)
#   SWEEP=oat bash datasets/prep_tree.sh nq         # only what SWEEP=oat needs
#   TREE_B="16 32" TREE_R="8 32" TREE_S="4 8" bash datasets/prep_tree.sh nq   # explicit B x r x s product instead
#   (or submit it: DATASET=nq sbatch datasets/tree_artifacts.slurm)
# New datasets (prepared by prep_datasets.slurm + embed.slurm), the layout run_tree_pir_old.slurm expects:
#   data/<name>_tree/index_<name>/            single-threaded pyserini index (+ docmap.bin)
#   data/<name>_tree/B<B>_r<r>_s<s>/          layout.json, stage1_*.bin, stage1_vocab.json, stage2/{data,idmap}.bin
#   data/<name>_tree/stage3_doc.npy(.ids)     doc embeddings in index order
#   data/<name>_tree/queries.qrels.tsv, queries.qrels.analyzed.json, qrels.trec
# msmarco / scifact keep their original index and stage-3 files; only the missing B<B>_r<r>_s<s>/ dirs are added
# to data/<name>/pir/ (--db-only).
# Needs Java 21 + pyserini. Combos that already exist are skipped (pass FORCE=1 to rebuild).
set -euo pipefail
name="${1:?name}"
ROOT="${DATASETS_ROOT:-../datasets}"
D="$ROOT/$name"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

jv="$(java -version 2>&1 | grep -oE 'version "[0-9]+' | grep -oE '[0-9]+$' || echo 0)"
[[ "$jv" -ge 21 ]] || { echo "pyserini needs Java 21+, found '$jv' ($(command -v java)). In the conda env: conda install -c conda-forge openjdk=21" >&2; exit 1; }

# Which (B, r, s) to build
if [[ -n "${TREE_B:-}${TREE_R:-}${TREE_S:-}" ]]; then
  SHAPE_ARGS=(--B ${TREE_B:-16 32 64} --r ${TREE_R:-8 32 128} --s ${TREE_S:-4 8 16})
else
  mapfile -t combos < <(DATASET="$name" SWEEP="${SWEEP:-all}" bash tree_sweep_grid.sh --shapes | awk '{print $1","$2","$3}')
  [[ ${#combos[@]} -gt 0 ]] || { echo "tree_sweep_grid.sh has no shapes for $name" >&2; exit 1; }
  SHAPE_ARGS=(--combos "${combos[@]}")
  echo "SWEEP=${SWEEP:-all}: ${#combos[@]} (B, r, s) combos for $name"
fi

if [[ "$name" == "msmarco" || "$name" == "scifact" ]]; then
  if [[ "$name" == "msmarco" ]]; then IDX=data/msmarco_reordered; else IDX=indexes/scifact; fi
  T="${TREE_OUT:-data/${name}/pir}"
  python3 tree/build_tree_artifacts.py --dataset-dir "$D" --out "$T" --name "$name" --index "$IDX" --db-only \
    "${SHAPE_ARGS[@]}" --workers "$WORKERS" ${FORCE:+--force}
  echo "Tree artifacts ready in $T"
  exit 0
fi

T="${TREE_OUT:-data/${name}_tree}"
mkdir -p "$T"
if [[ ! -d "$T/index_${name}" ]]; then
  mkdir -p "$T/coll"
  # pyserini JsonCollection input, title + text like the e5 embeddings
  python3 -c "
import json
with open('$T/coll/docs.jsonl', 'w') as o:
    for l in open('$D/corpus.jsonl'):
        d = json.loads(l); o.write(json.dumps({'id': d['_id'], 'contents': (d.get('title', '') + ' ' + d.get('text', '')).strip()}) + '\n')"
  # --threads 1 keeps Lucene internal docids in corpus order (the tree's blocks are runs of consecutive docids)
  python3 -m pyserini.index.lucene --collection JsonCollection --input "$T/coll" --index "$T/index_${name}" \
    --generator DefaultLuceneDocumentGenerator --threads 1
  rm -rf "$T/coll"
fi

python3 tree/build_tree_artifacts.py --dataset-dir "$D" --out "$T" --name "$name" \
  "${SHAPE_ARGS[@]}" --workers "$WORKERS" ${FORCE:+--force}
echo "Tree artifacts ready in $T"
