#!/bin/bash
# Build the tree method's artifacts for a dataset prepared by prep_datasets.slurm + embed.slurm.
#   bash datasets/prep_tree.sh nq 32 8 4          # name B r s  -> data/nq_tree/B32_r8_s4/...
# Run once per (B, r, s) in the tree grid; the Lucene index is built only the first time.
# Output layout (what run_tree_pir_old.slurm expects for the new datasets):
#   data/<name>_tree/index_<name>/            single-threaded pyserini index (+ docmap.bin)
#   data/<name>_tree/B<B>_r<r>_s<s>/          layout.json, stage1_*.bin, stage1_vocab.json, stage2/{data,idmap}.bin
#   data/<name>_tree/stage3_doc.npy(.ids)     doc embeddings in index order
#   data/<name>_tree/queries.qrels.tsv, queries.qrels.analyzed.json, qrels.trec
# Needs Java 21 + pyserini. build_small_test_artifacts.py is pure Python, so expect ~hours for the 5M-doc sets;
# for MS MARCO v2 (138M) use the collaborator's optimised prepare_data.py / prepare_data_stage2.py instead.
set -euo pipefail
name="${1:?name}"; B="${2:-32}"; R="${3:-8}"; S="${4:-4}"
ROOT="${DATASETS_ROOT:-../datasets}"
D="$ROOT/$name"
T="data/${name}_tree"
mkdir -p "$T/coll"

if [[ ! -d "$T/index_${name}" ]]; then
  # pyserini JsonCollection input, title + text like the e5 embeddings
  python3 -c "
import json, sys
with open('$T/coll/docs.jsonl', 'w') as o:
    for l in open('$D/corpus.jsonl'):
        d = json.loads(l); o.write(json.dumps({'id': d['_id'], 'contents': (d.get('title', '') + ' ' + d.get('text', '')).strip()}) + '\n')"
  # --threads 1 keeps Lucene internal docids in corpus order
  python3 -m pyserini.index.lucene --collection JsonCollection --input "$T/coll" --index "$T/index_${name}" \
    --generator DefaultLuceneDocumentGenerator --threads 1 --storeDocvectors
  rm -rf "$T/coll"
fi

if [[ ! -f "$T/B${B}_r${R}_s${S}/stage1_data.bin" ]]; then
  python3 tree/build_small_test_artifacts.py "$D" "$T" "$B" "$R" "$S" "$name"
fi
echo "Tree artifacts ready in $T/B${B}_r${R}_s${S}"
