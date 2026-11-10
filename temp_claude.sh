#!/bin/bash
# (2026-11-09) estimate-accuracy check for Daniel; delete once the estimates are settled.
# temp_claude.sh: one-off check, safe to run (reads only). Compares the estimates run_all.sh submitted with
# (logs/lists/*/*.tsv) against what the same configs really used (resource_usage.tsv). Output also saved in logs/.
cd "$(dirname "$(readlink -f "$0")")"
source ./slurm_common.sh
out="logs/temp_claude_$(date +%Y%m%d_%H%M%S).txt"; mkdir -p logs
{
echo "usage log: $USAGE_FILE ($(wc -l < "$USAGE_FILE" 2>/dev/null || echo 0) lines)"
echo
echo "==== A. what the runs really used (resource_usage.tsv; latest line per config), per method + dataset"
printf "  %-8s %-22s %6s %8s %8s %8s %8s %6s\n" method dataset runs medGB p90GB maxGB maxMin failed
awk -F'\t' 'NF >= 5 && $4 != "max_rss_gb" { k = $1 "|" $2 "|" $3; g[k] = $4; s[k] = $5; f[k] = ($7 ~ /failed/) }
  END { for (k in g) { split(k, a, "|"); m = a[1] "|" a[2]; if (f[k]) { nf[m]++; continue }
          n[m]++; G[m] = G[m] " " g[k]; if (s[k] / 60 > T[m]) T[m] = s[k] / 60 }
        for (m in G) { c = split(substr(G[m], 2), x, " "); for (i = 2; i <= c; i++) { v = x[i]; for (j = i - 1; j >= 1 && x[j] + 0 > v + 0; j--) x[j+1] = x[j]; x[j+1] = v }
          split(m, a, "|"); printf "  %-8s %-22s %6d %8.1f %8.1f %8.1f %8.0f %6d\n", a[1], a[2], c, x[int((c + 1) / 2)], x[int(c * 0.9 + 0.99)], x[c], T[m], nf[m] }
        for (m in nf) if (!(m in G)) { split(m, a, "|"); printf "  %-8s %-22s %6d %8s %8s %8s %8s %6d\n", a[1], a[2], 0, "-", "-", "-", "-", nf[m] } }' "$USAGE_FILE" | sort

echo
echo "==== B. estimate (before rounding) vs real use, same config, for configs that have run"
echo "  ratio = used / estimated: 1.0 = spot on, 0.3 = asked ~3x too much, >1 = would OOM. Only model / scaled estimates"
echo "  (the ones made before the config had run). 'bucket%' = used / what the job actually asked for."
printf "  %-8s %-18s %-8s %5s  %9s %9s %9s  %8s  %9s\n" method dataset source n minRatio medRatio maxRatio bucket% timeRatio
for f in $(ls -d logs/lists/*/ 2>/dev/null | sort); do
    for l in "$f"*.tsv; do [[ -f "$l" ]] && echo "$l"; done
done | while read -r l; do
    b=$(basename "$l"); method=${b%%_*}; bucket=$(sed -E 's/.*_([0-9]+)G_[0-9]+min\.tsv$/\1/' <<< "$b")
    case $method in bins|tree) ;; *) continue ;; esac
    while read -r -a F; do
        n=${#F[@]}; d=${F[0]}; gb=${F[n-3]}; mn=${F[n-2]}; src=${F[n-1]%%-*}
        [[ $src == model || $src == scaled ]] || continue
        if [[ $method == bins ]]; then key=$(bins_config_key "${F[4]}" "${F[1]}" "${F[2]}" "${F[3]}")
        else
            read -r h lf <<< "$(tree_cuckoo "$d")"; tag=""; [[ "$h" != 2 || "$lf" != 0.4 ]] && tag="_h${h}lf${lf}"
            key="b${F[1]}_r${F[2]}_s${F[3]}_L${F[4]}_q${F[5]}_c${F[6]}${tag}"
        fi
        echo "$method|$d|$key|$src|$gb|$mn|$bucket"
    done < "$l"
done | awk -F'|' -v uf="$USAGE_FILE" '
    BEGIN { while ((getline line < uf) > 0) { split(line, u, "\t"); if (u[4] == "max_rss_gb") continue
              k = u[1] "|" u[2] "|" u[3]; if (u[7] ~ /failed/) { fail[k] = u[4]; continue } ug[k] = u[4]; us[k] = u[5] } }
    { k = $1 "|" $2 "|" $3; if (seen[k]++) next                     # first estimate made for this config
      if (k in ug) { r = ug[k] / ($5 > 0 ? $5 : 1); t = (us[k] / 60) / ($6 > 0 ? $6 : 1); g = $1 "|" $2 "|" $4
                     R[g] = R[g] " " r; TR[g] = TR[g] " " t; B[g] = B[g] " " ug[k] / $7 }
      else if (k in fail) { g = $1 "|" $2 "|" $4; nf[g]++ } }
    function med(s,  x, c, i, j, v) { c = split(substr(s, 2), x, " "); for (i = 2; i <= c; i++) { v = x[i]; for (j = i - 1; j >= 1 && x[j] + 0 > v + 0; j--) x[j+1] = x[j]; x[j+1] = v }
        lo = x[1]; hi = x[c]; cnt = c; return x[int((c + 1) / 2)] }
    END { for (g in R) { split(g, a, "|"); m = med(R[g]); l = lo; h = hi; c = cnt; b = med(B[g]); t = med(TR[g])
            printf "  %-8s %-18s %-8s %5d  %9.2f %9.2f %9.2f  %7.0f%%  %9.2f%s\n", a[1], a[2], a[3], c, l, m, h, 100 * b, t, (nf[g] ? "   (+" nf[g] " OOM/timeout)" : "") }
          for (g in nf) if (!(g in R)) { split(g, a, "|"); printf "  %-8s %-18s %-8s %5d  only failed runs: %d\n", a[1], a[2], a[3], 0, nf[g] } }' | sort
echo
echo "==== C. failed lines in the usage log (OOM / timeout), latest 15"
grep -P '\tfailed' "$USAGE_FILE" 2>/dev/null | tail -15 | sed 's/^/  /'
} 2>&1 | tee "$out"
echo "(saved to $out)"
