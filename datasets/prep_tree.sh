#!/bin/bash
# Build the tree method's artifacts for a dataset prepared by prep_datasets.slurm + embed.slurm.
#   bash datasets/prep_tree.sh nq                       # every (B, r, s) in the default grid, one postings pass
#   TREE_B="16 32" TREE_R="8 32" TREE_S="4 8" bash datasets/prep_tree.sh nq
#   (or submit it: DATASET=nq sbatch datasets/tree_artifacts.slurm)
# Output layout (what run_tree_pir_old.slurm expects for the new datasets):
#   data/<name>_tree/index_<name>/            single-threaded pyserini index (+ docmap.bin)
#   data/<name>_tree/B<B>_r<r>_s<s>/          layout.json, stage1_*.bin, stage1_vocab.json, stage2/{data,idmap}.bin
#   data/<name>_tree/stage3_doc.npy(.ids)     doc embeddings in index order
#   data/<name>_tree/queries.qrels.tsv, queries.qrels.analyzed.json, qrels.trec
# Needs Java 21 + pyserini. Combos that already exist are skipped (pass FORCE=1 to rebuild).
set -euo pipefail
name="${1:?name}"
ROOT="${DATASETS_ROOT:-../datasets}"
D="$ROOT/$name"
T="${TREE_OUT:-data/${name}_tree}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
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
  --B ${TREE_B:-16 32 64} --r ${TREE_R:-8 32 128} --s ${TREE_S:-4 8 16} \
  --workers "$WORKERS" ${FORCE:+--force}
echo "Tree artifacts ready in $T"
