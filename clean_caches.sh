#!/bin/bash
# clean_caches.sh: delete caches nothing will use again. run_me.sh runs it (DO_IT=1) after it has resubmitted.
#
#   bash clean_caches.sh                 # dry run: what it would delete and how much (changes nothing)
#   DO_IT=1 bash clean_caches.sh         # delete
#   REAL=1 ...                           # judge "finished" by the real runs' results (default: the simulated ones)
#
# 1. Tree Cuckoo cache, cuckoo_cache/<dataset>/<shape>/shared/ (one per B/r/s, every config of the shape mmaps it):
#    deleted once EVERY config of that shape in the tree grid (tree_sweep_grid.sh --list, SWEEP=all) has a
#    metadata.json in this mode's results folder and no pending/running job covers any of them. Shapes no longer in the
#    grid count as finished. A dataset with a queued tree array whose configs can't be worked out keeps everything.
#    msmarco_original/ (the collaborator's layout) is judged like msmarco. Other folders (scifact's ablation grid/,
#    old task_*/) are left alone: clean_cuckoo_cache.sh handles those.
# 2. PACMANN graph files, /scratch/$USER/pac/<dataset>_corpus_embed_f32_<N>_192_<m>_neigh<m>_step<s>_k<k>{.ngt,
#    _graph.npy}: pir_app never reads them back (it rebuilds the graph every run, Pacmann/private-search.go), so they
#    are deleted unless a pending/running job covers that config. While any pacmann job of the dataset is running,
#    files written in the last 24 h are kept as well.
# Pending and running jobs of both kinds (simulated and real) count as "in use".
set -uo pipefail
cd "$(dirname "$0")"
if [[ "${REAL:-0}" == 1 ]]; then export SIMULATE=0; else export SIMULATE=1; fi
source ./slurm_common.sh
RES="${RESULTS_DIR:-/scratch/$USER/results}"
DO="${DO_IT:-}"
PAC_DIR="${PACMANN_CACHE_DIR:-/scratch/$USER/pac}"
command -v squeue >/dev/null || { echo "   no squeue here, caches not checked"; exit 0; }

# folders queued jobs will write (both modes; compared by folder name) and method|dataset pairs we can't map
declare -A QNAME=() QBLOCK=() PACRUN=()
while IFS=$'\t' read -r a b c _; do
    if [[ "$a" == "?" ]]; then QBLOCK["$b|$c"]=1; elif [[ -n "$a" ]]; then QNAME["$(basename "$a")"]=1; fi
done < <( (SIMULATE=1 RESULTS_DIR=/x queued_dirs; SIMULATE=0 RESULTS_DIR=/x queued_dirs) 2>/dev/null )
while read -r name; do d=${name#pacmann_}; PACRUN["${d%%_*}"]=1; done \
    < <(squeue -u "$USER" -h -t RUNNING -o %j 2>/dev/null | grep '^pacmann_' | sort -u)

kb_total=0
del() {   # del <path>: count it and delete it (DO_IT=1)
    local kb; kb=$(du -sk "$1" 2>/dev/null | cut -f1); kb_total=$((kb_total + ${kb:-0}))
    [[ -n "$DO" ]] && rm -rf -- "$1"
    return 0
}

# ---- 1. tree Cuckoo cache
n_del=0 n_keep=0
for ddir in cuckoo_cache/*/; do
    [[ -d "$ddir" ]] || continue
    cname=$(basename "$ddir"); d=${cname%_original}
    if [[ -n "${QBLOCK["tree|$d"]:-}" ]]; then
        echo "   cuckoo_cache/$cname: a queued tree array of $d has configs we can't map, all kept"
        n_keep=$((n_keep + $(ls -d "$ddir"*/shared 2>/dev/null | wc -l))); continue
    fi
    # shape -> number of grid configs still to do (no metadata.json, or a queued job writes it)
    declare -A todo=()
    tag=$(tree_cuckoo_tag "$d")
    while read -r _ ds B r s L q c; do
        B=${B#B=}; r=${r#r=}; s=${s#s=}; L=${L#L=}; q=${q#q=}; c=${c#c=}
        name="tree_grid_${ds}_b${B}_r${r}_s${s}_L${L}_q${q}_c${c}_k10"
        key="B${B}_r${r}_s${s}"
        if [[ ! -s "$RES/${tag}${name}/metadata.json" || -n "${QNAME[$name]:-}" ]]; then
            todo[$key]=$(( ${todo[$key]:-0} + 1 ))
        else
            todo[$key]=${todo[$key]:-0}
        fi
    done < <(DATASET="$d" SWEEP=all bash tree_sweep_grid.sh --list 2>/dev/null)
    for sdir in "$ddir"*/; do
        sdir=${sdir%/}; shape=$(basename "$sdir")
        [[ -d "$sdir/shared" ]] || continue
        if (( ${todo[$shape]:-0} > 0 )); then n_keep=$((n_keep + 1)); continue; fi
        echo "   $sdir/shared: every config of the shape is finished$([[ -z "${todo[$shape]+x}" ]] && echo " (shape not in the grid)")"
        del "$sdir/shared"; n_del=$((n_del + 1))
        [[ -n "$DO" ]] && rmdir "$sdir" 2>/dev/null
    done
    unset todo
done
printf '   tree Cuckoo cache: %d shapes %s, %d kept (configs still to run)\n' "$n_del" \
    "$([[ -n "$DO" ]] && echo deleted || echo "would be deleted")" "$n_keep"

# ---- 2. PACMANN graph files
p_del=0 p_keep=0
if [[ -d "$PAC_DIR" ]]; then
    now=$(date +%s)
    for f in "$PAC_DIR"/*_neigh*_step*_k*; do
        [[ -e "$f" ]] || continue
        b=$(basename "$f")
        [[ "$b" =~ ^(.+)_corpus_embed_f32_[0-9]+_[0-9]+_[0-9]+_neigh([0-9]+)_step([0-9]+)_k([0-9]+)(\.ngt|_graph\.npy)$ ]] || continue
        d=${BASH_REMATCH[1]} m=${BASH_REMATCH[2]} st=${BASH_REMATCH[3]} k=${BASH_REMATCH[4]}
        if [[ -n "${QNAME["pacmann_${d}_k${k}_steps${st}_neighb${m}"]:-}" ]] ||
           { [[ -n "${PACRUN[$d]:-}" ]] && (( now - $(stat -c %Y "$f") < 86400 )); }; then
            p_keep=$((p_keep + 1)); continue
        fi
        del "$f"; p_del=$((p_del + 1))
    done
fi
printf '   PACMANN graph files (%s): %d %s, %d kept (their job is queued or running)\n' "$PAC_DIR" "$p_del" \
    "$([[ -n "$DO" ]] && echo deleted || echo "would be deleted")" "$p_keep"
printf '   %s %d GB\n' "$([[ -n "$DO" ]] && echo freed || echo "would free")" $(( kb_total / 1048576 ))
[[ -n "$DO" ]] || echo "   (dry run: DO_IT=1 bash clean_caches.sh deletes them)"
