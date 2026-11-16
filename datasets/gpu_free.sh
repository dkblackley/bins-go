#!/bin/bash
# Which GPU types are free right now, and which ones pending jobs are queuing for.
#   bash datasets/gpu_free.sh            # gpuq
#   bash datasets/gpu_free.sh contrib-gpuq
p="${1:-gpuq}"

echo "== free / total GPUs per type on $p (drained/down nodes skipped)"
sinfo -p "$p" -N -h -O "nodelist:30,statecompact:12,gres:300,gresused:300" |
while read -r node state gres used; do
    [[ "$state" =~ ^(drain|drng|down|maint|fail) ]] && continue
    for kv in $(grep -oE 'gpu:[^:,()]+:[0-9]+' <<<"$gres"); do t=${kv#gpu:}; echo "T ${t%:*} ${t##*:} $node"; done
    for kv in $(grep -oE 'gpu:[^:,()]+:[0-9]+' <<<"$used"); do t=${kv#gpu:}; echo "U ${t%:*} ${t##*:} $node"; done
done | awk '$1=="T"{tot[$2]+=$3; cap[$2" "$4]=$3} $1=="U"{use[$2]+=$3; u[$2" "$4]=$3}
    END{for (t in tot) printf "  %-12s free %3d / %3d\n", t, tot[t]-use[t], tot[t]
        print "== nodes with a free GPU (type node free)"
        for (k in cap) if (cap[k] > u[k]) printf "  %s %d\n", k, cap[k]-u[k]}'

echo "== GPUs asked for by pending jobs (count, request)"
squeue -p "$p" -t PD -h -O "tres-per-node:60" | sed 's#gres/##g' | sort | uniq -c | sort -rn
