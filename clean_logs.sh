#!/bin/bash
# clean_logs.sh: move old logs out of the way, so check_runs.sh and a quick `ls logs/` only show what is current.
#
#   bash clean_logs.sh            # dry run: what it would move (changes nothing)
#   DO_IT=1 bash clean_logs.sh    # move them to logs/old/<now>/ (archives older than KEEP_DAYS=7 are deleted)
#   DELETE=1 DO_IT=1 bash ...     # delete instead of archiving
#
# Never touches anything a queued (pending or running) job still uses:
#   * logs/<anything>_<jobid>[_<task>].{out,err,txt} of a job still in squeue stays
#   * logs/lists/<run>/ folders that a queued array reads its configs from stay, and so do their lines in
#     logs/lists/submitted.tsv (check_runs.sh / run_me.sh use them to see what is queued); other lines are dropped
#   * results folders are never touched (check_runs.sh's "crashed" comes from those, not from logs/)
# Moves:
#   * job logs (Tree_/Bins_/Pacmann_/Tiptoe*/Coeus_/tree_artifacts_/treedb_*/... in logs/) of jobs no longer queued
#   * logs/check/<run>/ and logs/run_me/ except the newest KEEP_RUNS=3
#   * logs/lists/<run>/ folders no queued job uses
#   * the fail files in the repo root (bins_low_fails, bins_high_fails, pacman_fails, tiptoe_fails: append-only
#     lists of tasks that exited early; check_runs.sh section 6 counts them)
set -uo pipefail
cd "$(dirname "$0")"
[[ -d logs ]] || { echo "no logs/ here"; exit 0; }
DO="${DO_IT:-}"; KEEP_DAYS="${KEEP_DAYS:-7}"; KEEP_RUNS="${KEEP_RUNS:-3}"
DEST="logs/old/$(date +%Y%m%d_%H%M%S)"
n_moved=0
move() {   # move <path> [subdir in the archive]
    local dst="$DEST/${2:-}"
    n_moved=$((n_moved + 1))
    [[ -n "$DO" ]] || return 0
    if [[ -n "${DELETE:-}" ]]; then rm -rf -- "$1"; else mkdir -p "$dst" && mv -- "$1" "$dst/"; fi
}

# job ids still in the queue: array parents (%F) and plain job ids (%A, %i)
declare -A active=()
while read -r a b c; do active[$a]=1; active[$b]=1; active[${c%%_*}]=1; done \
    < <(squeue -u "$USER" -h -o "%F %A %i" 2>/dev/null)
echo "jobs in the queue: ${#active[@]} ids; their logs, list files and submitted.tsv lines are kept"

# ---- job logs in logs/
n_jobs=0 n_kept=0
for f in logs/*; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f")
    [[ "$base" =~ _([0-9]+)(_[0-9]+)?\.(out|err|txt|log)$ ]] || continue   # only files named after a job id
    if [[ -n "${active[${BASH_REMATCH[1]}]:-}" ]]; then n_kept=$((n_kept + 1)); continue; fi
    move "$f" jobs; n_jobs=$((n_jobs + 1))
done
echo "job logs: $n_jobs to archive, $n_kept kept (job still queued)"

# ---- logs/check and logs/run_me: keep the newest runs
for d in logs/check logs/run_me; do
    [[ -d "$d" ]] || continue
    mapfile -t runs < <(ls -1t "$d" 2>/dev/null | grep -v '^since$' | sed -E 's/(_queued\.tsv|\.log)$//' | awk '!seen[$0]++')
    n=0
    for r in "${runs[@]:$KEEP_RUNS}"; do
        for f in "$d/$r" "$d/$r.log" "$d/${r}_queued.tsv"; do
            [[ -e "$f" ]] && { move "$f" "$(basename "$d")"; n=$((n + 1)); }
        done
    done
    echo "$d: $n old entries to archive (newest $KEEP_RUNS runs kept)"
done

# ---- logs/lists: folders no queued array reads; submitted.tsv trimmed to queued jobs
sub=logs/lists/submitted.tsv
declare -A used=()
if [[ -f "$sub" ]]; then
    while IFS=$'\t' read -r jid list _off; do
        [[ -n "${active[$jid]:-}" ]] && used[$(dirname "$list")]=1
    done < "$sub"
fi
n=0
for d in logs/lists/*/; do
    d=${d%/}
    [[ -n "${used[$d]:-}" ]] && continue
    move "$d" lists; n=$((n + 1))
done
echo "logs/lists: $n folders to archive, ${#used[@]} kept (queued arrays read them)"
if [[ -f "$sub" ]]; then
    keep=$(awk -F'\t' 'NR == FNR { a[$1] = 1; next } a[$1]' <(printf '%s\n' "${!active[@]}") "$sub" | wc -l)
    echo "$sub: $(grep -c . "$sub") lines, $keep belong to queued jobs (the rest are dropped)"
    if [[ -n "$DO" ]]; then
        [[ -z "${DELETE:-}" ]] && { mkdir -p "$DEST/lists"; cp "$sub" "$DEST/lists/submitted.tsv"; }
        awk -F'\t' 'NR == FNR { a[$1] = 1; next } a[$1]' <(printf '%s\n' "${!active[@]}") "$sub" > "$sub.tmp" && mv "$sub.tmp" "$sub"
    fi
fi

# ---- fail files in the repo root
for f in bins_low_fails bins_high_fails pacman_fails tiptoe_fails; do
    [[ -s "$f" ]] && { echo "$f: to archive ($(grep -c . "$f") lines)"; move "$f" fail_files; }
done

# ---- old archives
if [[ -d logs/old ]]; then
    mapfile -t stale < <(find logs/old -mindepth 1 -maxdepth 1 -type d -mtime +"$KEEP_DAYS" 2>/dev/null)
    (( ${#stale[@]} )) && echo "logs/old: ${#stale[@]} archives older than $KEEP_DAYS days to delete"
    if [[ -n "$DO" ]]; then for d in "${stale[@]}"; do rm -rf -- "$d"; done; fi
fi

if [[ -n "$DO" ]]; then
    echo "done: $n_moved items $([[ -n "${DELETE:-}" ]] && echo deleted || echo "moved to $DEST")"
else
    echo "dry run ($n_moved items): DO_IT=1 bash clean_logs.sh to $([[ -n "${DELETE:-}" ]] && echo delete || echo archive) them"
fi
