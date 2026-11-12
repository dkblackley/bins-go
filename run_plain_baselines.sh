#!/bin/bash
# Submit one plain_baselines.slurm job per dataset (BM25, exact dense, hybrid -> baselines.json). Run from the repo root.
#
#   bash run_plain_baselines.sh                         # scifact msmarco nq fever dbpedia-entity msmarco-v2-25m
#                                                       # msmarco-v2-50m msmarco-v2
#   bash run_plain_baselines.sh nq fever                # only these
#   DRY_RUN=1 bash run_plain_baselines.sh               # print the sbatch commands only
#   LOCAL=1 bash run_plain_baselines.sh scifact nq      # no slurm: run them here, one after another
#   DATASETS_ROOT=/path/to/datasets BASELINES_OUT=x.json EXTRA_ARGS="-alpha 0.3" bash run_plain_baselines.sh
#
# Every job adds its dataset to baselines.json under a file lock, so they can all run at once; re-running a
# dataset overwrites only its own entry.
#
# Sizes (32 threads): memory is the query terms' postings (~6 B x ~35 per doc) + doc lengths + the dense top-1000
# heaps; the corpus embeddings are mmap'd (page cache, reclaimable). Dense is ~130 ns per (query, doc) per thread:
# msmarco-v2 (138M docs, ~4k queries) is roughly 1 h of dense + the time to read ~110 GB of embeddings and the corpus.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs bin

DATASETS="${*:-scifact msmarco nq fever dbpedia-entity msmarco-v2-25m msmarco-v2-50m msmarco-v2}"
export DATASETS_ROOT="${DATASETS_ROOT:-../datasets}"

# Build a fresh binary with its own name: replacing a file running jobs use kills them on NFS (see run_me.sh)
BIN="bin/plain_baselines.$(date +%Y%m%d_%H%M%S)"
if [[ -n "${DRY_RUN:-}" ]]; then
    echo "would build $BIN"
else
    ( source ./go_env.sh 2>/dev/null; CGO_ENABLED=0 go build -o "$BIN" ./cmd/plain_baselines ) \
        || { echo "build failed"; exit 1; }
    ln -sfn "$(basename "$BIN")" bin/plain_baselines.latest
    echo "built $BIN"
fi

for d in $DATASETS; do
    # partition mem time
    case "$d" in
        msmarco-v2)      res=(bigmem 192G 2-00:00:00) ;;
        msmarco-v2-50m)  res=(normal 96G 1-00:00:00) ;;
        msmarco-v2-25m)  res=(normal 64G 0-16:00:00) ;;
        msmarco|fever|dbpedia-entity|hotpotqa) res=(normal 48G 0-06:00:00) ;;
        *)               res=(normal 32G 0-03:00:00) ;;
    esac
    if [[ ! -f "$DATASETS_ROOT/$d/corpus.jsonl" ]]; then
        echo "skip $d: no $DATASETS_ROOT/$d/corpus.jsonl"; continue
    fi
    if [[ -n "${LOCAL:-}" ]]; then
        echo "== $d (local)"
        DATASET="$d" BIN="$BIN" bash plain_baselines.slurm 2>&1 | tee "logs/PlainBaselines_${d}_local.log"
        continue
    fi
    cmd=(sbatch --parsable --job-name="PB_$d" --partition="${res[0]}" --mem="${res[1]}" --time="${res[2]}"
         --export=ALL,DATASET="$d",BIN="$BIN" plain_baselines.slurm)
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "${cmd[*]}"
    else
        echo "$d: job $("${cmd[@]}")  (${res[*]})"
    fi
done
[[ -n "${DRY_RUN:-}${LOCAL:-}" ]] || echo "Watch: squeue -u \$USER -n $(for d in $DATASETS; do printf 'PB_%s,' "$d"; done | sed 's/,$//')   logs: logs/PlainBaselines_PB_*"
