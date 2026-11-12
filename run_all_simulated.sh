#!/bin/bash
# Tiptoe + Coeus sweeps (the simulated / Python baselines) over every dataset, locally. Same grids and output folder
# names as run_scifact_local.sh, which this generalises. Run from the bins-go repo root:
#   bash run_all_simulated.sh                          # all of $DEFAULT_SETS below
#   bash run_all_simulated.sh scifact nq               # only some
#   STAGES=coeus bash run_all_simulated.sh nq          # only one baseline (default: "tiptoe coeus")
#
# Datasets use the layout from datasets/prepare_dataset.py + embed (see datasets/README.md); scifact and msmarco
# (v1) use their old layouts. A dataset that isn't prepared and embedded yet is skipped with a warning.
# Needs: python with numpy (+ faiss or sklearn for Tiptoe's k-means), Go to build simplepir_bench.
# Both baselines are always simulated: Tiptoe uses the SEAL-free simulator (baselines/tiptoe/simulate_tiptoe.py) even
# if ./pir_app_tiptoe exists, since the real protocol takes hours locally. Coeus uses opt2 for server time if it's
# built (./baselines/setup_baselines.sh coeus).
#
#   JOBS=8 (default)  concurrent configs on datasets with <= 1M docs; each gets GOMAXPROCS=nproc/JOBS threads.
#   JOBS_BIG=2        same for bigger datasets: every config holds the whole corpus text (and the Coeus TF-IDF) in
#                     RAM, so 8 copies of MS MARCO at once doesn't fit. Use 1 for msmarco-v2.
#                     Quality/byte numbers don't depend on JOBS. Measured server times DO (the PIR scan is memory-
#                     bandwidth bound, so neighbours slow each other down): use JOBS=1 JOBS_BIG=1 FORCE=1 for final timings.
#   BIG_MA="2" BIG_BUNDLES="160"   Tiptoe grid on datasets with > 1M docs: only Tiptoe's own defaults (multi-assign 2,
#                     160 docs per bundle), so 7 preprocessing runs instead of 56. The DPC sweep (the quality/cost curve)
#                     is kept in full. BIG_MA="1 2" BIG_BUNDLES="40 80 160 320" runs the full grid like small datasets.
#   FORCE=1           rerun configs whose output dir already has metadata.json (default: skip them, like slurm).
set -euo pipefail

DATASETS=${DATASETS_ROOT:-../../../datasets}
OUT=${RESULTS_DIR:-../../../datasets}
DEFAULT_SETS="scifact nq fever msmarco msmarco-v2-50m msmarco-v2"
STAGES=${STAGES:-tiptoe coeus}
has() { [[ " $STAGES " == *" $1 "* ]]; }
SETS=("$@"); [ ${#SETS[@]} -gt 0 ] || read -r -a SETS <<< "$DEFAULT_SETS"

FORCE=${FORCE:-0}
NCPU=$(nproc)
LOGS=$OUT/baseline_logs
mkdir -p $OUT/tiptoe $OUT/coeus $LOGS
JOBFILE=$(mktemp); trap 'rm -f "$JOBFILE"' EXIT
TIPTOE_MAX_CLUSTERS=25196   # preprocess_tiptoe.py caps the cluster count here

# Quality + communication only: no server-time benchmarks (simplepir_bench / opt2) are run.
OPT2_ARG=""

# Runs the job lines in $JOBFILE (one shell command each), $jobs at a time. A failing config doesn't stop the rest.
run_jobs() {
  local n; n=$(wc -l < "$JOBFILE")
  echo "running $n $1 configs, $jobs at a time ($threads threads each); per-config logs in $LOGS"
  [ "$n" -gt 0 ] || return 0
  if ! xargs -d '\n' -n 1 -P "$jobs" bash -c < "$JOBFILE"; then
    echo "some $1 configs failed: grep -l -i error $LOGS/$1_*.log"
  fi
  : > "$JOBFILE"
}

for name in "${SETS[@]}"; do
  echo "================ $name"
  # Paths exactly as the baselines resolve them (baselines/common.py), and N from the embedding matrix itself
  if ! N=$(python3 - "$DATASETS" "$name" <<'EOF'
import os, sys
sys.path.insert(0, "baselines")
import common
import numpy as np
p = common.dataset_paths(sys.argv[1], sys.argv[2])
for k in ("queries", "qrels", "corpus_vec", "query_vec"):
    if not os.path.exists(p[k]):
        sys.exit(f"{p[k]} not found")
print(np.load(p["corpus_vec"], mmap_mode="r").shape[0])
EOF
  ); then
    echo "SKIPPING $name: not prepared/embedded yet (see datasets/README.md)"
    continue
  fi
  if [ "$N" -gt 1000000 ]; then jobs=${JOBS_BIG:-2}; else jobs=${JOBS:-8}; fi
  threads=$(( NCPU / jobs > 0 ? NCPU / jobs : 1 ))
  echo "N=$N docs"

  # ---------- tiptoe ----------
  # DPC x MA x BUNDLE x HINT x k. Several DPC values give the same cluster count (all of them => 1 cluster on scifact,
  # several hit the 25196 cap on msmarco-v2), so preprocessing is deduped on the planned cluster count: the first DPC
  # that gives it is built and reused (same seed + same C = same output), and the runs are deduped on their output
  # name, before anything runs, so two processes never write the same dir.
  if has tiptoe; then
    declare -A SEEN=() PRE=() CPLAN=()
    DPCS="250 2500 40000"
    for dpc in $DPCS; do CPLAN[$dpc]=$(python3 -c "print(min(max(1, round($N / $dpc)), $TIPTOE_MAX_CLUSTERS))"); done
    if [ "$N" -gt 1000000 ]; then MAS=${BIG_MA:-2}; BUNDLES=${BIG_BUNDLES:-160}
    else MAS="1 2"; BUNDLES="40 320"; fi
    echo "tiptoe grid: dpc={$DPCS} ma={$MAS} bundle={$BUNDLES}"
    n_ma=$(wc -w <<< "$MAS"); n_b=$(wc -w <<< "$BUNDLES")
    n_pre=$(( $(printf '%s\n' "${CPLAN[@]}" | sort -u | wc -l) * n_ma * n_b ))   # distinct C x ma x bundle
    i_pre=0
    for dpc in $DPCS; do
      Cplan=${CPLAN[$dpc]}
      for ma in $MAS; do
        for bundle in $BUNDLES; do
          key=c${Cplan}_m${ma}_b${bundle}
          if [ -z "${PRE[$key]:-}" ]; then
            i_pre=$((i_pre + 1))
            # preprocessing is shared by every (hint, k), so it runs serially here: no races on $TT
            TT=$DATASETS/tiptoe/${name}_dpc${dpc}_m${ma}_b${bundle}
            if [ ! -f "$TT/tiptoe_meta.json" ]; then
              echo "preprocess [$i_pre/$n_pre] $name dpc=$dpc ma=$ma bundle=$bundle (C~$Cplan)"
              # progress bars (stderr) go to the terminal and the log; the rest (stdout) only to the log.
              # stdbuf: faiss' C++ k-means prints would otherwise sit in a buffer until it finishes.
              PLOG=$LOGS/preprocess_${name}_dpc${dpc}_m${ma}_b${bundle}.log
              : > "$PLOG"
              t0=$SECONDS
              # --kmeans-cache: the big k-means depends only on C, so every (ma, bundle) after the first reuses it
              if ! stdbuf -oL -eL python3 -u baselines/tiptoe/preprocess_tiptoe.py --dataset $DATASETS --name $name \
                  --docs-per-cluster $dpc --multi-assign $ma --bundle-size $bundle --out $TT \
                  --kmeans-cache $DATASETS/tiptoe/kmeans_cache >> "$PLOG" 2> >(tee -a "$PLOG" >&2); then
                echo "FAILED preprocess $name dpc=$dpc ma=$ma bundle=$bundle"
                continue
              fi
              echo "  done in $(( (SECONDS - t0) / 60 ))m$(( (SECONDS - t0) % 60 ))s"
            fi
            PRE[$key]=$TT
          fi
          TT=${PRE[$key]}
          C=$(python3 -c "import json;print(json.load(open('$TT/tiptoe_meta.json'))['num_clusters'])")
          for hint in 125 500; do
            for k in 10 100; do
              run=tiptoe_${name}_c${C}_m${ma}_b${bundle}_h${hint}_k${k}
              TT_OUT=$OUT/tiptoe/$run
              [ -n "${SEEN[$run]:-}" ] && continue
              SEEN[$run]=1
              if [ "$FORCE" != 1 ] && [ -f "$TT_OUT/metadata.json" ]; then continue; fi
              cmd="python3 baselines/tiptoe/simulate_tiptoe.py --dataset $DATASETS --name $name --tiptoe-dir $TT --k $k \
--emb-hint-mb $hint --results-root $OUT/tiptoe --parallelism $NCPU"
              echo "echo '=== $run'; GOMAXPROCS=$threads OMP_NUM_THREADS=$threads $cmd > $LOGS/$run.log 2>&1 || echo 'FAILED $run'" >> "$JOBFILE"
            done
          done
        done
      done
    done
    unset SEEN PRE CPLAN MAS BUNDLES
    run_jobs tiptoe
  fi

  # ---------- coeus ----------
  # F x tf x k. F values past the dataset's vocabulary are skipped by run_coeus.py (e.g. F>32768 on scifact).
  if has coeus; then
    for F in 8192 32768 262144; do
      warmed=0
      for tf in raw log; do
        for k in 10 100; do
          run=coeus_${name}_k${k}_f${F}_tf${tf}
          if [ "$FORCE" != 1 ] && [ -f "$OUT/coeus/$run/metadata.json" ]; then continue; fi
          cmd="python3 baselines/coeus/run_coeus.py --dataset $DATASETS --name $name --k $k --features $F --tf $tf \
--results-root $OUT/coeus $OPT2_ARG"
          if [ $warmed = 0 ]; then
            warmed=1
            echo "=== $run (serial)"
            $cmd > $LOGS/$run.log 2>&1 || echo "FAILED $run"
          else
            echo "echo '=== $run'; OMP_NUM_THREADS=$threads $cmd > $LOGS/$run.log 2>&1 || echo 'FAILED $run'" >> "$JOBFILE"
          fi
        done
      done
    done
    run_jobs coeus
  fi
done

echo "Done: see $OUT/{tiptoe,coeus}/*/metadata.json (logs in $LOGS)"
