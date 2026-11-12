#!/bin/bash
#SBATCH --job-name=tree_artifacts
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/tree_artifacts_%j.out
#SBATCH --error=logs/tree_artifacts_%j.err
#
# Lucene index + all tree artifacts for one dataset (see datasets/prep_tree.sh).
#   DATASET=nq sbatch datasets/tree_artifacts.slurm
# datasets/run_all.sh sizes it (treedb_model / measured builds in the usage log, slurm_common.sh). Postings are
# streamed into numpy now (build_tree_artifacts.py), so msmarco-v2 no longer needs the old 1000 GB.
#
# Logging, for sizing the next build: the summed heap (RssAnon) of every process of this job, sampled every 20 s,
# goes to logs/treedb_peak_<job>.txt and to the usage log as method "treedb" (also when it runs out of memory or
# time); the builder's own timings and per-worker peaks go to <out>/build_stats.json and the log.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./slurm_common.sh
DATASET="${DATASET:?set DATASET}"
CFG="W${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
PEAK="logs/treedb_peak_${SLURM_JOB_ID:-$$}.txt"
TFILE="logs/treedb_time_${SLURM_JOB_ID:-$$}.txt"   # record_usage format, filled with the heap peak

job_anon_kb() {   # summed RssAnon of this job's processes on this node
    local pid p e sum=0 kb
    for pid in $(pgrep -u "$(id -u)"); do
        p=/proc/$pid
        e=$( { tr '\0' '\n' < "$p/environ"; } 2>/dev/null | grep -m1 '^SLURM_JOB_ID=') || continue
        [[ "$e" == "SLURM_JOB_ID=${SLURM_JOB_ID:-none}" ]] || continue
        kb=$(awk '/^RssAnon:/ {print $2}' "$p/status" 2>/dev/null) && sum=$(( sum + ${kb:-0} ))
    done
    echo "$sum"
}
( peak=0; while sleep 20; do
      a=$(job_anon_kb); (( a > peak )) && peak=$a
      printf '%d %d %s\n' "$peak" "$SECONDS" "$(date +%FT%T)" > "$PEAK"
  done ) &
sampler=$!
write_tfile() { echo "Maximum resident set size (kbytes): $(awk '{print $1}' "$PEAK" 2>/dev/null || echo 0)" > "$TFILE"; }
source datasets/env.sh   # conda env + HF cache (override CONDA_BASE / CONDA_ENV / HF_HOME)

# in the background + wait: bash runs a trap only after a foreground command ends, i.e. never before the time limit
rc=0
bash datasets/prep_tree.sh "$DATASET" &
build=$!
trap 'kill -TERM "$build" 2>/dev/null; kill $sampler 2>/dev/null; write_tfile
      record_failure treedb "$DATASET" "$CFG" "$TFILE" 143 "$SECONDS" timeout; exit 143' TERM
wait "$build" || rc=$?
kill "$sampler" 2>/dev/null || true
a=$(job_anon_kb); [[ -s "$PEAK" ]] && (( a < $(awk '{print $1}' "$PEAK") )) || printf '%d %d %s\n' "$a" "$SECONDS" "$(date +%FT%T)" > "$PEAK"
write_tfile
echo "treedb heap peak (all processes): $(awk '{printf "%.1f GB", $1 / 1048576}' "$PEAK"), elapsed ${SECONDS}s"
if (( rc != 0 )); then
    record_failure treedb "$DATASET" "$CFG" "$TFILE" "$rc" "$SECONDS" || true
    exit "$rc"
fi
record_usage treedb "$DATASET" "$CFG" "$TFILE" "$SECONDS" || true
