#!/bin/bash
# Saves the configs the current bins / pacmann / tree grids produce (check_runs.sh EXPECTED_ONLY=1, i.e. what
# run_me.sh submits) to plotting/current_grid.txt. load_results.py only loads those configs by default
# (g.ONLY_CURRENT_GRID), so runs from older grids or one-off extras stay out of the plots.
#
#   bash plotting/save_current_grid.sh          # rewrite current_grid.txt from the grids as they are now
#   bash plotting/save_current_grid.sh nq fever # only these datasets (the others are dropped from the file)
#
# Re-run it when you WANT a grid change to show up in every plot. The fixed-MRR scripts (frozen_values.py,
# fixed_recall.py, plot_fixed_mrr.py) load every run anyway, so extra configs added only to
# fill MRR@10 values (e.g. tree_sweep_grid.sh SWEEP=lowmrr) reach them without being added here.
# Works off the cluster too: `module` is stubbed out.
set -euo pipefail
cd "$(dirname "$0")/.."
out=plotting/current_grid.txt
tmp=$(mktemp)
bash -c 'module() { :; }; export -f module
         EXPECTED_ONLY=1 METHODS="bins pacmann tree" RESULTS_DIR=RES bash check_runs.sh "$@" 2>/dev/null' _ "$@" |
    awk '$1 ~ /^(bins|pacmann|tree)$/ { sub(/^RES\//, "", $3); print $1 "\t" $2 "\t" $3 }' | sort -u > "$tmp"
[[ -s "$tmp" ]] || { echo "save_current_grid.sh: check_runs.sh listed no configs, $out left alone" >&2; rm -f "$tmp"; exit 1; }
{ echo "# method	dataset	folder (relative to the results dir). Written by plotting/save_current_grid.sh on $(date +%F)"
  echo "# from code $(git log -1 --format='%h %ci'); load_results.py only loads these (ONLY_CURRENT_GRID=0 loads everything)"
  cat "$tmp"; } > "$out"
rm -f "$tmp"
echo "wrote $out:"
awk -F'\t' '!/^#/ { n[$1 " " $2]++ } END { for (k in n) printf "  %-28s %5d\n", k, n[k] }' "$out" | sort
