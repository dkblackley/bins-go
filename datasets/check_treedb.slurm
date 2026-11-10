#!/bin/bash
#SBATCH --job-name=treedb_check
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:30:00
#SBATCH --output=logs/treedb_check.out
#SBATCH --error=logs/treedb_check.out
#
# Checks the streaming postings reader of tree/build_tree_artifacts.py (2026-11-10) before the msmarco-v2 tree
# artifacts are built with it: rebuilds one nq shape (B32 r32 s8) from the existing nq index into a scratch folder
# and compares every file byte for byte with data/nq_tree/, which the old reader (pyserini get_postings_list) built.
#
#   sbatch datasets/check_treedb.slurm        (from the repo root)
#   cat logs/treedb_check.out                 the verdict and, if it passed, what to run next
#
# The log has no job id in its name, so clean_logs.sh leaves it alone; each run overwrites it.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="/scratch/$USER/treedb_check_nq"
REF="data/nq_tree"
SHAPE="32,32,8"
FILES="stage1_vocab.json stage1/B32_r32/data.bin stage1/B32_r32/idmap.bin B32_r32_s8/stage2/data.bin B32_r32_s8/stage2/idmap.bin"

echo "treedb check started $(date '+%F %T') on $(hostname), job ${SLURM_JOB_ID:-local}, code $(git log -1 --format='%h %s' 2>/dev/null || echo "(no git on this node)")"
echo

source datasets/env.sh
rm -rf "$OUT"   # the builder skips shapes that already exist, so start empty
SECONDS=0
python3 tree/build_tree_artifacts.py --dataset-dir ../datasets/nq --out "$OUT" --name nq --index "$REF/index_nq" \
    --combos "$SHAPE" --db-only --workers 8
rc=$?
echo
echo "build exit code $rc after ${SECONDS}s"

same=0 differ=0
if (( rc == 0 )); then
    for f in $FILES; do
        if cmp -s "$REF/$f" "$OUT/$f"; then echo "  same:   $f"; same=$((same + 1))
        else echo "  DIFFER: $f ($(stat -Lc %s "$REF/$f" 2>/dev/null || echo missing) vs $(stat -Lc %s "$OUT/$f" 2>/dev/null || echo missing) bytes)"
             differ=$((differ + 1)); fi
    done
fi

echo
echo "========================================================================================================"
if (( rc == 0 && differ == 0 && same == 5 )); then
    cat <<'EOF'
PASSED: the new tree artifact builder writes exactly the same files as the old one (5/5 identical).

Next steps (from the repo root on hopper):

  # 1. Replace the old 1000 GB msmarco-v2 treedb jobs, and the v2 tree jobs waiting on them (they would wait
  #    forever on a cancelled job). The msmarco (non-v2) tree jobs are not touched.
  scancel -u $USER -t PENDING -n treedb_msmarco-v2_sim,treedb_msmarco-v2-25m_sim,treedb_msmarco-v2-50m_sim
  names=$(squeue -u $USER -h -t PD -o %j | grep -E '^tree_msmarco-v2' | sort -u | paste -sd,)
  echo "$names"                                   # should list only tree_msmarco-v2*_sim names
  scancel -u $USER -t PENDING -n "$names"

  # 2. Resubmit them (simulated low-RAM by default): treedb at ~32-67 GB on the normal partition, then the trees
  bash run_me.sh

  # 3. Tidy the logs (dry run first), then check
  bash clean_logs.sh
  DO_IT=1 bash clean_logs.sh
  bash check_runs.sh

  # Later, to see what the treedb builds really used (for the next estimates):
  #   cat logs/treedb_peak_<jobid>.txt     (heap peak in KB, seconds, time)
  #   cat data/msmarco-v2*_tree/build_stats.json
  rm -rf /scratch/$USER/treedb_check_nq     # this check's scratch output, no longer needed
EOF
else
    cat <<EOF
FAILED: do NOT replace the msmarco-v2 treedb jobs yet (leave the queued 1000 GB ones as they are).
  build exit code: $rc, identical files: $same/5, different: $differ
  The build log is above. If a worker stopped with "postings of ... differ from pyserini's get_postings_list",
  the streaming reader disagrees with the old one; send this log to Claude.
  Scratch output kept in $OUT for comparison.
EOF
fi
echo "========================================================================================================"
echo "finished $(date '+%F %T')"
