#!/bin/bash
# Run bins, pacmann and tree on scifact locally. Run from the bins-go repo root.
# Needs: Go + the NGT/hnsw libs you already build with (see build.sh), Java 21, and
#   pip install pyserini numpy   (only for the one-off tree artifact build)
set -euo pipefail

DATASETS=../../../datasets          # contains scifact/, index_scifact/ (bluge index for bins)
D=$DATASETS/scifact
T=data/scifact_tree                 # tree artifacts get built here
B=8; R=4; S=4                       # tree shape (any of your slurm grid values work, e.g. B=16 R=8 S=4)
A=$T/B${B}_r${R}_s${S}
OUT=results_local

source $HOME/opt/pacmann_env.sh
go build -o pir_app ./main.go
mkdir -p $OUT/bins $OUT/pacmann $OUT/tree

# ---------- bins ----------
./pir_app -n 5183 -t bins -name scifact -k 10 -binSize 1000 -docsPerBin 100 -vectors=true -dim 192 \
  -dataset $DATASETS -outDir $OUT/bins -outFile results -debug 1

# ---------- pacmann ----------
./pir_app -n 5183 -t pacmann -name scifact -k 10 -steps 15 -neighb 32 -dim 192 \
  -dataset $DATASETS -outDir $OUT/pacmann -outFile results -debug 1

# ---------- tree: one-off artifact build (skip if $A already exists) ----------
if [ ! -f "$A/stage1_data.bin" ]; then
  mkdir -p $T/coll
  # pyserini JsonCollection input, title + text like the e5 embeddings
  python3 -c "
import json
with open('$T/coll/docs.jsonl','w') as o:
    for l in open('$D/corpus.jsonl'):
        d=json.loads(l); o.write(json.dumps({'id':d['_id'],'contents':(d.get('title','')+' '+d.get('text','')).strip()})+'\n')"
  # --threads 1 keeps Lucene internal docids in corpus order
  python3 -m pyserini.index.lucene --collection JsonCollection --input $T/coll --index $T/index_scifact \
    --generator DefaultLuceneDocumentGenerator --threads 1 --storeDocvectors
  # stage1/stage2 DBs, vocab, layout, docmap.bin, stage3 doc vectors, queries.qrels.{tsv,analyzed.json}, qrels.trec
  python3 tree/build_small_test_artifacts.py $D $T $B $R $S
fi

# ---------- tree ----------
./pir_app -n 5183 -t tree -name scifact -k 10 -dim 192 -dataset $DATASETS -outDir $OUT/tree -outFile results -debug 0 \
  -B $B -r $R -s $S -L 20 -k_candidates 20 -query-words 8 -max-queries 300 -k1 0.9 -b 0.4 -scale 10 \
  -precompute -pir-batch-size 128 -pir-threads 8 -cuckoo-load-factor 0.4 -cuckoo-max-kicks 1000 -cuckoo-build-attempts 8 \
  -index $T/index_scifact -layout-path $A/layout.json -vocab $A/stage1_vocab.json \
  -stage1-data-bin $A/stage1_data.bin -stage1-idmap-bin $A/stage1_idmap.bin \
  -stage2-data-bin $A/stage2/data.bin -stage2-idmap-bin $A/stage2/idmap.bin \
  -queries $T/queries.qrels.tsv -qrels $T/qrels.trec \
  -doc-embed $T/stage3_doc.npy -doc-id-map $T/stage3_doc.npy.ids
# -query-embed defaults to $D/scifact_query_embed_f32.npy (rows matched to queries.jsonl order automatically)

echo "Done: see $OUT/{bins,pacmann,tree}/metadata.json and results*.json"
