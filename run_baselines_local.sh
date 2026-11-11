#!/bin/bash
# run_baselines_local.sh: the simulated Tiptoe and Coeus runs of one dataset on this machine, WITH server-time
# estimates, into the folders plotting reads for them (plotting/globals.py EXTRA_RESULTS_DIRS). Existing results of
# that dataset are overwritten, so they gain a TotalAnswerTime.
#
#   bash run_baselines_local.sh                      # scifact, both baselines
#   bash run_baselines_local.sh scifact tiptoe       # only Tiptoe (or coeus)
#   DRY_RUN=1 bash run_baselines_local.sh            # list the configs, run nothing
#
# Server time:
#   Tiptoe  ./simplepir_bench (baselines/setup_baselines.sh sim) times SimplePIR's Server.Answer at each config's
#           real DB shape (simulate_tiptoe.py --bench-bin), divided by --parallelism (Tiptoe shards freely).
#   Coeus   Coeus' matmult/bin/opt2 (baselines/setup_baselines.sh coeus: SEAL 3.5 + the artifact, ~10 min to build)
#           times its matrix-vector product (run_coeus.py --opt2-bin; calibrated once, cached next to the binary).
#           Without it Coeus still runs (quality + bytes) and the script says so.
# Grids: the cluster's (run_tiptoe.slurm DPC/MA/BUNDLE lists x emb hint 125/500 x k 10/100; run_coeus.slurm
# F 8192/32768/262144 x tf raw/log x k 10/100). Tiptoe needs its preprocessed DBs in <datasets>/tiptoe/<d>_dpc*_m*_b*
# (preprocess_tiptoe.py; SciFact's are there); a missing one is skipped with a message.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

d=${1:-scifact}
which=${2:-tiptoe coeus}
DATASETS_ROOT=${DATASETS_ROOT:-$HOME/Nextcloud/10TB-STHDD/datasets}
TIPTOE_OUT=${TIPTOE_OUT:-$DATASETS_ROOT/tiptoe}
COEUS_OUT=${COEUS_OUT:-$DATASETS_ROOT/coeus}
PAR=${PARALLELISM:-16}   # what the cluster runs pass (SLURM_CPUS_PER_TASK=16), so the times compare
BENCH=${BENCH_BIN:-$PWD/simplepir_bench}
OPT2=${OPT2_BIN:-$HOME/opt/src/Coeus_artifact/matmult/bin/opt2}
SECONDS=0

if [[ " $which " == *" tiptoe "* ]]; then
    eval "$(grep -E '^(DPC_LIST|MA_LIST|BUNDLE_LIST)=' run_tiptoe.slurm)"   # the cluster's grid
    bench=(); if [[ -x "$BENCH" ]]; then bench=(--bench-bin "$BENCH"); else echo "tiptoe: no $BENCH (baselines/setup_baselines.sh sim): no server time"; fi
    n=0
    for dpc in "${DPC_LIST[@]}"; do for ma in "${MA_LIST[@]}"; do for b in "${BUNDLE_LIST[@]}"; do
        tt="$DATASETS_ROOT/tiptoe/${d}_dpc${dpc}_m${ma}_b${b}"
        [[ -f "$tt/tiptoe_meta.json" ]] || { echo "tiptoe: skip dpc$dpc m$ma b$b: no $tt (preprocess_tiptoe.py)"; continue; }
        for hint in 125 500; do for k in 10 100; do
            n=$((n + 1))
            if [[ -n "${DRY_RUN:-}" ]]; then echo "tiptoe $d dpc$dpc m$ma b$b h$hint k$k"; continue; fi
            python baselines/tiptoe/simulate_tiptoe.py --dataset "$DATASETS_ROOT" --name "$d" --tiptoe-dir "$tt" \
                --k "$k" --emb-hint-mb "$hint" --results-root "$TIPTOE_OUT" --parallelism "$PAR" "${bench[@]}" \
                > /tmp/tiptoe_local_$$.log 2>&1 \
                && echo "tiptoe ok   dpc$dpc m$ma b$b h$hint k$k" \
                || { echo "tiptoe FAIL dpc$dpc m$ma b$b h$hint k$k:"; tail -3 /tmp/tiptoe_local_$$.log; }
        done; done
    done; done; done
    rm -f /tmp/tiptoe_local_$$.log
    echo "tiptoe: $n configs ($(( SECONDS / 60 )) min so far)"
fi

if [[ " $which " == *" coeus "* ]]; then
    opt2=(); if [[ -x "$OPT2" ]]; then opt2=(--opt2-bin "$OPT2")
    else echo "coeus: no $OPT2 (baselines/setup_baselines.sh coeus, or OPT2_BIN=...): quality + bytes only, no server time"; fi
    n=0
    for f in 8192 32768 262144; do for tf in raw log; do for k in 10 100; do
        n=$((n + 1))
        if [[ -n "${DRY_RUN:-}" ]]; then echo "coeus $d F$f tf$tf k$k"; continue; fi
        python baselines/coeus/run_coeus.py --dataset "$DATASETS_ROOT" --name "$d" --k "$k" --features "$f" --tf "$tf" \
            --results-root "$COEUS_OUT" --token-cache "$DATASETS_ROOT/coeus_cache" --parallelism "$PAR" "${opt2[@]}" \
            > /tmp/coeus_local_$$.log 2>&1 \
            && echo "coeus  ok   F$f tf$tf k$k" \
            || { echo "coeus  FAIL F$f tf$tf k$k:"; tail -3 /tmp/coeus_local_$$.log; }
    done; done; done
    rm -f /tmp/coeus_local_$$.log
    echo "coeus: $n configs"
fi
echo "done in $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s"
