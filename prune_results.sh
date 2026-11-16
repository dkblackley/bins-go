#!/bin/bash
# prune_results.sh: move every bins / pacmann / tree run folder that run_me.sh would NOT produce out of a results
# folder, into an archive folder next to it. "Would produce" = the grids check_runs.sh lists (EXPECTED_ONLY=1), for
# run_me.sh's default datasets. Nothing is deleted; a dry run unless APPLY=1.
#
#   bash prune_results.sh                     # list what would move (and how many per method / dataset)
#   APPLY=1 bash prune_results.sh             # move it
#   RESULTS_DIR=/scratch/$USER/results_simulated APPLY=1 bash prune_results.sh     # on hopper
#
# Env: DATASETS_ROOT  datasets folder (default: ~/Nextcloud/10TB-STHDD/datasets if it exists, else ../datasets)
#      RESULTS_DIR    folder to clean (default $DATASETS_ROOT/results_simulated)
#      ARCHIVE        where the extra folders go (default ${RESULTS_DIR}_old); sub-folders like h3_lf0.88/ are kept
#      DATASETS       default run_me.sh's list; a dataset NOT in it (e.g. trec-covid) has all of its runs moved
#      SWEEP          tree grid (default all, as run_me.sh)
# Only top-level folders named bins_* / pacmann_* / tree_* (and those inside h<H>_lf<LF>/ sub-folders) are looked at;
# Tiptoe / Coeus, plan.tsv, resource_usage.tsv etc. are never touched. A folder that already exists in ARCHIVE is
# left where it is and reported.
set -uo pipefail
cd "$(dirname "$0")"
if [[ -z "${DATASETS_ROOT:-}" ]]; then
    for c in "$HOME/Nextcloud/10TB-STHDD/datasets" ../datasets; do [[ -d "$c" ]] && { DATASETS_ROOT=$c; break; }; done
fi
DATASETS_ROOT=$(cd "${DATASETS_ROOT:?no datasets folder, set DATASETS_ROOT}" && pwd)
RES=$(cd "${RESULTS_DIR:-$DATASETS_ROOT/results_simulated}" && pwd) || exit 1
ARCHIVE="${ARCHIVE:-${RES}_old}"
DATASETS="${DATASETS:-scifact msmarco nq fever dbpedia-entity msmarco-v2-25m msmarco-v2-50m msmarco-v2}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# the bins slurm scripts call `module` (hopper's Lmod); off hopper make it a no-op so their --list-raw still works
if ! type module >/dev/null 2>&1; then module() { :; }; export -f module; fi

# the expected folders, relative to RES (e.g. bins_vec0_nq_k10_bs1000_dpb10, h3_lf0.88/tree_grid_msmarco-v2_...)
EXPECTED_ONLY=1 METHODS="bins pacmann tree" RESULTS_DIR="$RES" DATASETS_ROOT="$DATASETS_ROOT" \
    DATASET_DIR="$DATASETS_ROOT" SWEEP="${SWEEP:-all}" bash check_runs.sh $DATASETS 2>/dev/null |
  awk -v res="$RES/" '$1 ~ /^(bins|pacmann|tree)$/ && NF >= 3 { p = $3; sub("^" res, "", p); print p }' |
  sort -u > "$tmp/expected"
n_exp=$(wc -l < "$tmp/expected")
[[ $n_exp -gt 0 ]] || { echo "check_runs.sh listed no expected configs; nothing moved" >&2; exit 1; }

# the run folders that exist
( cd "$RES" && { find . -mindepth 1 -maxdepth 1 -type d \( -name 'bins_*' -o -name 'pacmann_*' -o -name 'tree_*' \)
                 find . -mindepth 2 -maxdepth 2 -type d -path './h*_lf*/*' \( -name 'bins_*' -o -name 'pacmann_*' -o -name 'tree_*' \); } |
  sed 's|^\./||' | sort -u ) > "$tmp/present"
comm -23 "$tmp/present" "$tmp/expected" > "$tmp/extra"

echo "results:  $RES"
echo "archive:  $ARCHIVE"
echo "expected: $n_exp configs (run_me.sh grids for: $DATASETS)"
echo "present:  $(wc -l < "$tmp/present") run folders, kept $(comm -12 "$tmp/present" "$tmp/expected" | wc -l), extra $(wc -l < "$tmp/extra")"
echo
echo "extra folders per method / dataset:"
sed -E 's|^h[^/]*/||; s/^(bins|pacmann|tree)_(vec[0-9]_|text_|grid_)?([a-z0-9-]+)_.*/\1 \3/' "$tmp/extra" | sort | uniq -c | sort -rn
[[ -s "$tmp/extra" ]] || { echo "  none"; exit 0; }

if [[ "${APPLY:-0}" != 1 ]]; then
    echo; echo "first 20:"; head -20 "$tmp/extra" | sed 's/^/  /'
    echo; echo "dry run: nothing moved. Full list: re-run with LIST=1, or APPLY=1 to move them into $ARCHIVE"
    [[ "${LIST:-0}" == 1 ]] && sed 's/^/  /' "$tmp/extra"
    exit 0
fi
moved=0; skipped=0
while read -r rel; do
    dest="$ARCHIVE/$rel"
    if [[ -e "$dest" ]]; then echo "  already in archive, left in place: $rel"; skipped=$((skipped + 1)); continue; fi
    mkdir -p "$(dirname "$dest")" && mv "$RES/$rel" "$dest" && moved=$((moved + 1))
done < "$tmp/extra"
echo "moved $moved folders to $ARCHIVE ($skipped already there, left in place)"
