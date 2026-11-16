#!/bin/bash
# hold_jobs.sh: put your big pending jobs on hold so the smaller ones get through, and release them later.
# Holding is reversible: nothing is cancelled, and held jobs keep their place (AGE keeps growing).
#
#   bash hold_jobs.sh                   # hold every pending job that asks for >= 500G (same as --mem 500G)
#   bash hold_jobs.sh --mem 250G        # hold everything pending at >= 250G
#   bash hold_jobs.sh --dry-run         # only show what it would hold
#   bash hold_jobs.sh --name msmarco-v2 # only jobs whose name matches this regex (combine with --mem)
#   bash hold_jobs.sh --except treedb   # never touch jobs whose name matches this regex
#   bash hold_jobs.sh --list            # show what is held now
#   bash hold_jobs.sh --release         # release everything you held (or only >= --mem / --name if given)
#
# Notes:
#  - Array jobs are held/released as a whole array (all of its pending tasks), and one array always has one --mem.
#  - Held jobs still count as queued for run_me.sh / run_all.sh, so they are not submitted a second time.
#  - A job that others depend on (prep / treedb / binsidx) blocks its dependents while held; the table shows names,
#    so check it with --dry-run first, and keep such jobs with --except.
set -uo pipefail

mem_gb=500; mem_given=0; release=0; dry=0; list=0; name_re="."; except_re=""
usage() { sed -n 2,18p "$0"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mem)      mem_gb="$2"; mem_given=1; shift 2 ;;
        --mem=*)    mem_gb="${1#*=}"; mem_given=1; shift ;;
        --release)  release=1; shift ;;
        --dry-run|-n) dry=1; shift ;;
        --list)     list=1; shift ;;
        --name)     name_re="$2"; shift 2 ;;
        --except)   except_re="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *) echo "unknown flag: $1"; usage 1 ;;
    esac
done
# "500G", "500", "1T", "64000M" -> GB
to_gb() { awk -v m="$1" 'BEGIN { u = toupper(substr(m, length(m))); v = m + 0
    if (u == "T") v *= 1024; else if (u == "M") v /= 1024; else if (u == "K") v /= 1048576; printf "%d", v }'; }
min=$(to_gb "$mem_gb")
# --release with no --mem releases every held job, whatever its size
[[ $release == 1 && $mem_given == 0 ]] && min=0

user="${USER:-$(id -un)}"
if [[ $list == 1 ]]; then
    echo "held jobs (reason JobHeldUser):"
    squeue -u "$user" -h -r -t PD -o '%F|%m|%P|%r|%j' | awk -F'|' '$4 ~ /Held/' \
        | awk -F'|' '{ c[$1"|"$2"|"$3"|"$5]++ } END { for (k in c) { split(k, a, "|"); printf "  %-10s %6s %-7s %4d tasks  %s\n", a[1], a[2], a[3], c[k], a[4] } }' | sort -k5
    exit 0
fi

# one line per array job / job: id | mem | partition | reason | name, with task counts
mapfile -t rows < <(squeue -u "$user" -h -r -t PD -o '%F|%m|%P|%r|%j' 2>/dev/null \
    | awk -F'|' -v min="$min" -v rel="$release" -v nre="$name_re" -v xre="$except_re" '
        function gb(m,  u, v) { u = toupper(substr(m, length(m))); v = m + 0
            if (u == "T") v *= 1024; else if (u == "M") v /= 1024; else if (u == "K") v /= 1048576; return v }
        { held = ($4 ~ /Held/) }
        rel == 1 && !held { next }
        rel == 0 && held  { next }
        gb($2) < min || $5 !~ nre || (xre != "" && $5 ~ xre) { next }
        { k = $1"|"$2"|"$3"|"$5; n[k]++; if (!(k in r)) r[k] = $4 }
        END { for (k in n) print k "|" n[k] "|" r[k] }' | sort -t'|' -k2,2 -k4,4)

verb=$([[ $release == 1 ]] && echo release || echo hold)
if [[ ${#rows[@]} -eq 0 ]]; then echo "nothing to $verb (pending, >= ${min}G, name ~ /$name_re/${except_re:+, not /$except_re/})"; exit 0; fi

printf "%-10s %6s %-7s %6s  %-12s %s\n" JOBID MEM PART TASKS REASON NAME
ids=(); total=0
for row in "${rows[@]}"; do
    IFS='|' read -r id mem part name n reason <<< "$row"
    printf "%-10s %6s %-7s %6s  %-12s %s\n" "$id" "$mem" "$part" "$n" "$reason" "$name"
    ids+=("$id"); total=$((total + n))
done
echo "$verb: ${#ids[@]} jobs/arrays, $total pending tasks"
[[ $dry == 1 ]] && { echo "(dry run: nothing changed)"; exit 0; }

fail=0
for id in "${ids[@]}"; do scontrol "$verb" "$id" || fail=1; done
mkdir -p logs/check
echo "$(date +%FT%T) $verb ${ids[*]}" >> logs/check/hold_jobs.log
[[ $fail == 0 ]] && echo "done. Undo: bash hold_jobs.sh $([[ $release == 1 ]] && echo "--mem ${mem_gb}" || echo --release)" \
                 || { echo "some scontrol calls failed (see above)"; exit 1; }
