#!/bin/bash
# prune_vec0_sweep.sh <dataset> [results dir]: move the dataset's vec0 batch/rounds sweep results that are no longer
# in the sweep grid (run_bins_low/high.slurm add_vec0_sweep) out of the results folder, into
# <results dir>_vec0_sweep_old/, so plotting stops loading them. Nothing is deleted: move them back to restore.
#   bash prune_vec0_sweep.sh scifact                                   # hopper: /scratch/$USER/results_simulated
#   bash prune_vec0_sweep.sh scifact ~/Nextcloud/10TB-STHDD/datasets/results_simulated
#   DRY_RUN=1 bash prune_vec0_sweep.sh scifact ...                     # only list them
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
d=${1:?usage: prune_vec0_sweep.sh <dataset> [results dir]}
res=${2:-${RESULTS_DIR:-/scratch/${USER:-$(id -un)}/results_simulated}}
res=${res%/}
[[ -d "$res" ]] || { echo "no $res" >&2; exit 1; }
module() { :; }; export -f module   # a cluster command the slurm scripts call; harmless to skip for --list-raw
keep=$(for t in low high; do
    BINS_DATASETS="$d" BINS_VECS=false SLURM_SUBMIT_DIR="$PWD" bash "run_bins_$t.slurm" --list-raw 2>/dev/null
done | awk '$5 ~ /^vb[0-9]+r[0-9]+i[0-9]+$/ { v = substr($5, 3); split(v, a, /[ri]/)
            printf "bins_vec0_%s_k%s_bs%s_dpb%s_vb%s_vr%s_ib%s\n", $1, $2, $3, $4, a[1], a[2], a[3] }' | sort -u)
[[ -n "$keep" ]] || { echo "the grid has no sweep configs for $d: refusing to move everything" >&2; exit 1; }
old="${res}_vec0_sweep_old"
n=0
for f in "$res"/bins_vec0_"$d"_k*_vb*_vr*_ib*; do
    [[ -d "$f" ]] || continue
    grep -qxF "${f##*/}" <<< "$keep" && continue
    n=$((n + 1))
    if [[ -n "${DRY_RUN:-}" ]]; then echo "would move ${f##*/}"; else mkdir -p "$old"; mv "$f" "$old/"; fi
done
echo "$d: $(grep -c . <<< "$keep") sweep configs in the grid; $n results not in it $([[ -n "${DRY_RUN:-}" ]] && echo "would be" || echo "were") moved to $old"
