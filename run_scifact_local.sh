#!/bin/bash
# Run bins, pacmann, tree, tiptoe and coeus on scifact locally. Run from the bins-go repo root.
# Needs: Go + the NGT/hnsw libs you already build with (see build.sh), Java 21, and
#   pip install pyserini numpy   (only for the one-off tree artifact build; tiptoe/coeus need numpy)
# Tiptoe runs the real protocol if ./pir_app_tiptoe exists (./baselines/setup_baselines.sh tiptoe), otherwise the
# SEAL-free simulator. Coeus uses opt2 for server time if it's built (./baselines/setup_baselines.sh coeus).
set -euo pipefail

DATASETS=../../../datasets          # contains scifact/, index_scifact/ (bluge index for bins)
D=$DATASETS/scifact
T=data/scifact_tree                 # tree artifacts get built here
B=8; R=4; S=4                       # tree shape (any of your slurm grid values work, e.g. B=16 R=8 S=4)
A=$T/B${B}_r${R}_s${S}
OUT=results_local

source $HOME/opt/pacmann_env.sh
go build -o pir_app ./main.go
mkdir -p $OUT/bins $OUT/pacmann $OUT/tree $OUT/tiptoe $OUT/coeus

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

# ---------- tiptoe ----------
TT_DPC=${TT_DPC:-1000}              # docs per k-means cluster (Tiptoe used ~2500 on MS MARCO; 1000 = 6 clusters here)
TT=$DATASETS/tiptoe/scifact_dpc${TT_DPC}_m2_b160
if [ ! -f "$TT/tiptoe_meta.json" ]; then
  python3 baselines/tiptoe/preprocess_tiptoe.py --dataset $DATASETS --name scifact --docs-per-cluster $TT_DPC \
    --multi-assign 2 --bundle-size 160 --out $TT
fi
if [ -x ./pir_app_tiptoe ]; then
  C=$(python3 -c "import json;print(json.load(open('$TT/tiptoe_meta.json'))['num_clusters'])")
  TT_OUT=$OUT/tiptoe/tiptoe_scifact_c${C}_m2_b160_h500_k10
  mkdir -p $TT_OUT
  ./pir_app_tiptoe -n 5183 -t tiptoe -name scifact -k 10 -dim 192 -dataset $DATASETS -tiptoe-dir $TT \
    -tiptoe-emb-hint-mb 500 -outDir $TT_OUT -outFile results -debug 0
else
  echo "no ./pir_app_tiptoe, running the Tiptoe simulator instead"
  go build -o simplepir_bench ./cmd/simplepir_bench
  python3 baselines/tiptoe/simulate_tiptoe.py --dataset $DATASETS --name scifact --tiptoe-dir $TT --k 10 \
    --emb-hint-mb 500 --results-root $OUT/tiptoe --bench-bin ./simplepir_bench --parallelism "$(nproc)"
fi

# ---------- coeus ----------
OPT2=$HOME/opt/src/Coeus_artifact/matmult/bin/opt2
OPT2_ARG=(); [ -x "$OPT2" ] && OPT2_ARG=(--opt2-bin "$OPT2" --parallelism "$(nproc)")
# F values past scifact's vocabulary are skipped by the script (they'd only add empty ciphertexts)
for F in 8192 16384 32768 65536; do
  python3 baselines/coeus/run_coeus.py --dataset $DATASETS --name scifact --k 10 --features $F --tf raw \
    --results-root $OUT/coeus ${OPT2_ARG[@]+"${OPT2_ARG[@]}"}
done

echo "Done: see $OUT/{bins,pacmann,tree}/metadata.json, $OUT/{tiptoe,coeus}/*/metadata.json and results*.json"
