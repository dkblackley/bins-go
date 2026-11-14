#!/bin/bash
# shrink_time.sh: lower the --time of queued (pending or held) arrays whose estimate is known to be too long.
# Shorter limits get backfilled sooner. Only pending tasks change; running tasks keep their limit.
# Measured on 2026-11-09 (a one-off sacct check): bins runs on nq / fever / msmarco took 6-32% of the time they asked for.
#
#   bash shrink_time.sh --dry-run      # show old -> new limits, change nothing
#   bash shrink_time.sh                # new limit = old x 0.5, rounded up to TIME_BUCKETS, at least 60 min
#   bash shrink_time.sh --factor 0.4   # other factor
#   bash shrink_time.sh --name 'tree_' # other job names (regex; default: bins on the 5 small datasets)
# A run that hits the new limit is logged as a timeout and comes back with 2x time on the next run_me.sh.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./slurm_common.sh
factor=0.5; dry=0; name_re='^bins_(scifact|nq|fever|msmarco|dbpedia-entity)_[0-9]+G'; floor=60
while [[ $# -gt 0 ]]; do
    case "$1" in
        --factor)  factor="$2"; shift 2 ;;
        --name)    name_re="$2"; shift 2 ;;
        --min)     floor="$2"; shift 2 ;;
        --dry-run|-n) dry=1; shift ;;
        -h|--help) sed -n 2,10p "$0"; exit 0 ;;
        *) echo "unknown flag: $1"; exit 1 ;;
    esac
done
mins() { awk -v t="$1" 'BEGIN { d = 0; if (t ~ /-/) { split(t, a, "-"); d = a[1]; t = a[2] }
    n = split(t, a, ":"); printf "%d", d * 1440 + (n == 3 ? a[1] * 60 + a[2] : a[1]) }'; }
printf "%-10s %-34s %12s -> %-12s %s\n" JOBID NAME OLD NEW STATE
n=0
while read -r id name lim reason; do
    [[ "$name" =~ $name_re ]] || continue
    old=$(mins "$lim")
    new=$(awk -v o="$old" -v f="$factor" -v m="$floor" 'BEGIN { x = o * f; if (x < m) x = m; printf "%d", x }')
    new=$(time_bucket "$new"); (( new < old )) || continue
    printf "%-10s %-34s %12s -> %-12s %s\n" "$id" "$name" "$lim" "$(slurm_time "$new")" "$reason"
    n=$((n + 1))
    [[ $dry == 1 ]] || scontrol update JobId="$id" TimeLimit="$(slurm_time "$new")" 2>&1 | grep -v MAX_JOB_ID
done < <(squeue -u "${USER:-$(id -un)}" -h -t PD -o '%F %j %l %r' | sort -u)
echo "$n arrays$([[ $dry == 1 ]] && echo " would change (dry run)" || echo " changed")"
