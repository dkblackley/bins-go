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
OUT=../../../datasets/

source $HOME/opt/pacmann_env.sh
go build -o pir_app ./main.go
mkdir -p $OUT/bins $OUT/pacmann $OUT/tree $OUT/tiptoe $OUT/coeus

## ---------- bins ----------
#./pir_app -n 5183 -t bins -name scifact -k 10 -binSize 1000 -docsPerBin 100 -vectors=true -dim 192 \
#  -dataset $DATASETS -outDir $OUT/bins -outFile results -debug 1
#
## ---------- pacmann ----------
#./pir_app -n 5183 -t pacmann -name scifact -k 10 -steps 15 -neighb 32 -dim 192 \
#  -dataset $DATASETS -outDir $OUT/pacmann -outFile results -debug 1
#
## ---------- tree: one-off artifact build (skip if $A already exists) ----------
#if [ ! -f "$A/stage1_data.bin" ]; then
#  mkdir -p $T/coll
#  # pyserini JsonCollection input, title + text like the e5 embeddings
#  python3 -c "
#import json
#with open('$T/coll/docs.jsonl','w') as o:
#    for l in open('$D/corpus.jsonl'):
#        d=json.loads(l); o.write(json.dumps({'id':d['_id'],'contents':(d.get('title','')+' '+d.get('text','')).strip()})+'\n')"
#  # --threads 1 keeps Lucene internal docids in corpus order
#  python3 -m pyserini.index.lucene --collection JsonCollection --input $T/coll --index $T/index_scifact \
#    --generator DefaultLuceneDocumentGenerator --threads 1 --storeDocvectors
#  # stage1/stage2 DBs, vocab, layout, docmap.bin, stage3 doc vectors, queries.qrels.{tsv,analyzed.json}, qrels.trec
#  python3 tree/build_small_test_artifacts.py $D $T $B $R $S
#fi
#
## ---------- tree ----------
#./pir_app -n 5183 -t tree -name scifact -k 10 -dim 192 -dataset $DATASETS -outDir $OUT/tree -outFile results -debug 0 \
#  -B $B -r $R -s $S -L 20 -k_candidates 20 -query-words 8 -max-queries 300 -k1 0.9 -b 0.4 -scale 10 \
#  -precompute -pir-batch-size 128 -pir-threads 8 -cuckoo-load-factor 0.4 -cuckoo-max-kicks 1000 -cuckoo-build-attempts 8 \
#  -index $T/index_scifact -layout-path $A/layout.json -vocab $A/stage1_vocab.json \
#  -stage1-data-bin $A/stage1_data.bin -stage1-idmap-bin $A/stage1_idmap.bin \
#  -stage2-data-bin $A/stage2/data.bin -stage2-idmap-bin $A/stage2/idmap.bin \
#  -queries $T/queries.qrels.tsv -qrels $T/qrels.trec \
#  -doc-embed $T/stage3_doc.npy -doc-id-map $T/stage3_doc.npy.ids
## -query-embed defaults to $D/scifact_query_embed_f32.npy (rows matched to queries.jsonl order automatically)

# Tiptoe/Coeus: same sweep as run_tiptoe.slurm / run_coeus.slurm, scifact only, JOBS configs at a time.
#   JOBS=8 (default)  concurrent configs; each gets GOMAXPROCS=nproc/JOBS threads.
#                     Quality/byte numbers don't depend on JOBS. Measured server times DO (the PIR scan is memory-
#                     bandwidth bound, so neighbours slow each other down): use JOBS=1 FORCE=1 for final timings.
#   FORCE=1           rerun configs whose output dir already has metadata.json (default: skip them, like slurm).
FORCE=${FORCE:-0}
NCPU=$(nproc)
JOBS=${JOBS:-8}
THREADS=$(( NCPU / JOBS > 0 ? NCPU / JOBS : 1 ))
LOGS=$OUT/baseline_logs; mkdir -p $LOGS
JOBFILE=$(mktemp); trap 'rm -f "$JOBFILE"' EXIT

# Runs the job lines in $JOBFILE (one shell command each), JOBS at a time. A failing config doesn't stop the rest.
run_jobs() {
  local n; n=$(wc -l < "$JOBFILE")
  echo "running $n $1 configs, $JOBS at a time ($THREADS threads each); per-config logs in $LOGS"
  [ "$n" -gt 0 ] || return 0
  if ! xargs -d '\n' -n 1 -P "$JOBS" bash -c < "$JOBFILE"; then
    echo "some $1 configs failed: grep -l -i error $LOGS/$1_*.log"
  fi
  : > "$JOBFILE"
}

# ---------- tiptoe ----------
# DPC x MA x BUNDLE x HINT x k. On scifact (5183 docs) DPC>=5000 all give 1 cluster, so several (DPC, MA)
# settings collapse to the same output dir; the first one wins, as in the slurm script. Deduping happens here,
# before anything runs, so two processes never write the same dir.
if [ -x ./pir_app_tiptoe ]; then
  TT_MODE=real
else
  TT_MODE=sim
  echo "no ./pir_app_tiptoe, running the Tiptoe simulator instead"
fi
declare -A SEEN=()
for dpc in 250 500 1000 2500 5000 10000 40000; do
  for ma in 1 2; do
    for bundle in 40 80 160 320; do
      # preprocessing is ~1s and shared by every (hint, k), so it runs serially here: no races on $TT
      TT=$DATASETS/tiptoe/scifact_dpc${dpc}_m${ma}_b${bundle}
      if [ ! -f "$TT/tiptoe_meta.json" ]; then
        python3 baselines/tiptoe/preprocess_tiptoe.py --dataset $DATASETS --name scifact --docs-per-cluster $dpc \
          --multi-assign $ma --bundle-size $bundle --out $TT > $LOGS/preprocess_dpc${dpc}_m${ma}_b${bundle}.log 2>&1
      fi
      # multi-assign as applied: it's capped at the number of clusters, and the simulator names its folder with it
      read -r C MA_USED < <(python3 -c "import json;m=json.load(open('$TT/tiptoe_meta.json'));print(m['num_clusters'],m['multi_assign'])")
      for hint in 125 500; do
        for k in 10 100; do
          name=tiptoe_scifact_c${C}_m${MA_USED}_b${bundle}_h${hint}_k${k}
          TT_OUT=$OUT/tiptoe/$name
          [ -n "${SEEN[$name]:-}" ] && continue
          SEEN[$name]=1
          if [ "$FORCE" != 1 ] && [ -f "$TT_OUT/metadata.json" ]; then continue; fi
          if [ "$TT_MODE" = real ]; then
            mkdir -p $TT_OUT
            cmd="./pir_app_tiptoe -n 5183 -t tiptoe -name scifact -k $k -dim 192 -dataset $DATASETS -tiptoe-dir $TT \
-tiptoe-emb-hint-mb $hint -outDir $TT_OUT -outFile results -debug 0"
          else
            cmd="python3 baselines/tiptoe/simulate_tiptoe.py --dataset $DATASETS --name scifact --tiptoe-dir $TT --k $k \
--emb-hint-mb $hint --results-root $OUT/tiptoe --parallelism $NCPU"
          fi
          echo "echo '=== $name (dpc=$dpc)'; GOMAXPROCS=$THREADS OMP_NUM_THREADS=$THREADS $cmd > $LOGS/$name.log 2>&1 || echo 'FAILED $name'" >> "$JOBFILE"
        done
      done
    done
  done
done
run_jobs tiptoe

# ---------- coeus ----------
# F x tf x k. F values past scifact's vocabulary (~28.9k terms, so F>32768) are skipped by run_coeus.py.
# Quality + communication only; the opt2 server-time benchmark is never run.
OPT2_ARG=""
for F in 8192 16384 32768 65536 131072 262144; do
  warmed=0
  for tf in raw log; do
    for k in 10 100; do
      name=coeus_scifact_k${k}_f${F}_tf${tf}
      if [ "$FORCE" != 1 ] && [ -f "$OUT/coeus/$name/metadata.json" ]; then continue; fi
      cmd="python3 baselines/coeus/run_coeus.py --dataset $DATASETS --name scifact --k $k --features $F --tf $tf \
--results-root $OUT/coeus $OPT2_ARG"
      if [ $warmed = 0 ]; then
        warmed=1
        echo "=== $name (serial)"
        $cmd > $LOGS/$name.log 2>&1 || echo "FAILED $name"
      else
        echo "echo '=== $name'; OMP_NUM_THREADS=$THREADS $cmd > $LOGS/$name.log 2>&1 || echo 'FAILED $name'" >> "$JOBFILE"
      fi
    done
  done
done
run_jobs coeus

echo "Done: see $OUT/{bins,pacmann,tree}/metadata.json, $OUT/{tiptoe,coeus}/*/metadata.json and results*.json"
