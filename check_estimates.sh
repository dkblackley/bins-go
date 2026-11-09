#!/bin/bash
# check_estimates.sh: are the memory / time requests close to what the runs actually use?
#
#   bash check_estimates.sh              # finished jobs of the last 7 days + the estimates behind every pending array
#   DAYS=2 bash check_estimates.sh       # look back 2 days instead
#   bash check_estimates.sh nq fever     # only job names containing these words
#
# 1. finished jobs (sacct), per job name (= method_dataset_bucket):
#      asked     the --mem / --time the array got
#      used      peak RSS / elapsed: median and max over its tasks, and as % of what was asked
#    Low % = over-asking (a smaller bucket would have started sooner). OOM / TIMEOUT counts = under-asking.
# 2. pending / held arrays (logs/lists/submitted.tsv + the config list each one runs):
#      est GB    what estimate_resources (slurm_common.sh) gave its configs: min / median / max, before rounding up
#      bucket    what was asked after rounding up to MEM_BUCKETS / TIME_BUCKETS
#      source    where the estimate came from: measured (this dataset ran before), scaled-from-X (another dataset ran
#                that config), failed-bumped (it OOMed / timed out before), model (no run yet: a formula guess)
#    "model" estimates are the ones to distrust; the rest come from resource_usage.tsv.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
user="${USER:-$(id -un)}"
filt="${*:-.}"; filt="${filt// /|}"

echo "==== 1. finished jobs since $(date -d "-${DAYS:-7} days" +%F) (asked vs used)"
printf "  %-34s %5s %7s %7s %7s %5s   %6s %6s %6s %5s  %s\n" "job name" runs askGB medGB maxGB max% askMin medMin maxMin max% OOM/TO
sacct -u "$user" -S "$(date -d "-${DAYS:-7} days" +%F)" -n -P -o JobID,JobName,ReqMem,Timelimit,Elapsed,MaxRSS,State 2>/dev/null \
| awk -F'|' -v filt="$filt" '
    function gb(x,  u, v) { v = x + 0; u = toupper(x); gsub(/[^KMGT]/, "", u); u = substr(u, 1, 1)
        if (u == "K") v /= 1048576; else if (u == "M") v /= 1024; else if (u == "T") v *= 1024; else if (u == "") v /= 1073741824
        return v }
    function mins(t,  d, a, n) { d = 0; if (t ~ /-/) { split(t, a, "-"); d = a[1]; t = a[2] }
        n = split(t, a, ":"); return d * 1440 + (n == 3 ? a[1] * 60 + a[2] + a[3] / 60 : a[1] + a[2] / 60) }
    function med(s,  a, n, i, j, x) { n = split(s, a, " "); for (i = 2; i <= n; i++) { x = a[i]; for (j = i - 1; j >= 1 && a[j] + 0 > x + 0; j--) a[j+1] = a[j]; a[j+1] = x }
        return n ? a[int((n + 1) / 2)] : 0 }
    $1 !~ /\./ {   # the job / array task line: name, request, state
        if ($7 ~ /PENDING|RUNNING|CANCELLED by/) next
        name[$1] = $2; req[$1] = gb($3); lim[$1] = mins($4); el[$1] = mins($5); st[$1] = $7; next }
    $1 ~ /\.batch$/ { id = $1; sub(/\.batch$/, "", id); rss[id] = gb($6) }
    END {
        for (id in name) {
            n = name[id]; if (n !~ filt || !(id in rss)) continue
            cnt[n]++; R[n] = req[id]; L[n] = lim[id]; U[n] = U[n] " " rss[id]; E[n] = E[n] " " el[id]
            if (rss[id] > MU[n]) MU[n] = rss[id]; if (el[id] > ME[n]) ME[n] = el[id]
            if (st[id] ~ /OUT_OF_ME/) oom[n]++; if (st[id] ~ /TIMEOUT/) to[n]++
        }
        for (n in cnt) printf "  %-34s %5d %7.0f %7.1f %7.1f %4.0f%%   %6.0f %6.0f %6.0f %4.0f%%  %s\n", n, cnt[n], R[n], med(U[n]), MU[n], R[n] ? 100 * MU[n] / R[n] : 0,
            L[n], med(E[n]), ME[n], L[n] ? 100 * ME[n] / L[n] : 0, (oom[n] + to[n]) ? (oom[n] + 0) "/" (to[n] + 0) : "-"
    }' | sort -k1,1
echo "  (max% under ~50% for a whole array = it could have gone one bucket lower; OOM/TO = it asked too little)"

echo
echo "==== 2. pending / held arrays: estimate behind each request"
sub="${SUBMITTED_FILE:-logs/lists/submitted.tsv}"
[[ -f "$sub" ]] || { echo "  no $sub (only arrays submitted by run_all.sh are listed)"; exit 0; }
declare -A pend=()
while read -r j n m l r; do pend[$j]="$n|$m|$l|$r"; done < <(squeue -u "$user" -h -t PD -o '%F %j %m %l %r' 2>/dev/null | sort -u)
printf "  %-10s %-32s %6s %5s  %6s %6s %6s  %8s %6s  %s\n" JOBID NAME ASKGB TASKS minGB medGB maxGB ASKTIME maxMin "SOURCE (configs)"
while IFS=$'\t' read -r jid list off; do
    [[ -n "${pend[$jid]:-}" && -f "$list" ]] || continue
    IFS='|' read -r name mem lim reason <<< "${pend[$jid]}"
    [[ "$name" =~ $filt ]] || continue
    # list lines: config fields ... GB minutes source; an array runs lines off+1 .. off+1000
    tail -n +"$((off + 1))" "$list" | head -n "${MAX_ARRAY:-1000}" | awk -v j="$jid" -v nm="$name" -v mem="$mem" -v lim="$lim" -v why="$reason" '
        { g = $(NF-2) + 0; t = $(NF-1) + 0; s = $NF; sub(/-.*/, "", s); src[s]++; G[NR] = g; if (t > mt) mt = t }
        END { n = NR; for (i = 2; i <= n; i++) { x = G[i]; for (k = i - 1; k >= 1 && G[k] > x; k--) G[k+1] = G[k]; G[k+1] = x }
              printf "  %-10s %-32s %6s %5d  %6.0f %6.0f %6.0f  %8s %6d  ", j, nm, mem, n, G[1], G[int((n + 1) / 2)], G[n], lim, mt
              for (s in src) printf "%s %d  ", s, src[s]; printf "%s\n", (why ~ /Held/ ? "[held]" : "") }'
done < "$sub" | sort -k2,2
echo "  (medGB far below ASKGB = bucket rounding: e.g. 260G estimates land in the 500G bucket)"
echo "  The estimates and their sources: slurm_common.sh (estimate_resources, bins_model) and tree_sweep_grid.sh (tree_mem_gb)."
echo "  Measured usage they learn from: ${USAGE_FILE:-${RESULTS_DIR:-/scratch/$user/results}/resource_usage.tsv}"
