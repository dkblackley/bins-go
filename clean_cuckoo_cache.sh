#!/bin/bash
# clean_cuckoo_cache.sh: turn the old per-task Cuckoo caches into the shared per-shape caches, then delete the rest.
#
#   bash clean_cuckoo_cache.sh            # dry run: shows what it would keep, move and delete (changes nothing)
#   DO_IT=1 bash clean_cuckoo_cache.sh    # do it
#
# Until 2026-11-10 every tree task cached its tables in its own cuckoo_cache/<dataset>/<shape>/task_<job>_<task>/
# and nothing ever read them again (3.8 TB). Now every config of a shape shares cuckoo_cache/<dataset>/<shape>/shared/
# (run_tree_pir_old.slurm). For each shape this script:
#   1. if shared/ has no Stage 2 table yet, moves the tables of one finished task folder there (a folder with a Stage 2
#      table is complete: Stage 2 is built after every Stage 1 level) and writes the "ready" marker, so the shape's
#      configs are sized as cache hits (tree_cache_ready). Same tables a new run would build (deterministic seeds); if
#      the tree artifacts changed since, pir_app notices (size/mtime tag) and rebuilds them.
#   2. deletes every task_* folder whose array job is no longer in the queue (pending or running jobs keep theirs).
# Never touches shared/, the scifact ablation's grid/ folders, or anything outside cuckoo_cache/.
set -uo pipefail
cd "$(dirname "$0")"
[[ -d cuckoo_cache ]] || { echo "no cuckoo_cache/ here"; exit 0; }
DO="${DO_IT:-}"
run() { if [[ -n "$DO" ]]; then "$@"; else echo "    would: $*"; fi; }

active=$(squeue -u "$USER" -h -o %F 2>/dev/null | sort -u | paste -sd'|')
[[ -n "$active" ]] || active="NONE"
echo "array jobs still in the queue (their task folders are kept): $(tr '|' ' ' <<< "$active")"
echo "before: $(du -sh cuckoo_cache 2>/dev/null | cut -f1)"

promoted=0 deleted=0 kept=0
for shape_dir in cuckoo_cache/*/*/; do
    shape_dir=${shape_dir%/}
    compgen -G "$shape_dir/task_*" > /dev/null || continue
    shared="$shape_dir/shared"
    if ! compgen -G "$shared/stage2_*.cuckoo" > /dev/null; then
        src=""
        for t in "$shape_dir"/task_*; do
            [[ "$t" =~ /task_(${active})_ ]] && continue          # a queued job may still be writing it
            if compgen -G "$t/stage2_*.cuckoo" > /dev/null; then src=$t; break; fi
        done
        if [[ -n "$src" ]]; then
            echo "  $shape_dir: shared cache from $(basename "$src")"
            run mkdir -p "$shared"
            for f in "$src"/*.cuckoo; do run mv -n "$f" "$shared/"; done
            # default layout (2 hashes, load factor 0.4) is what every pre-2026-11-10 run used
            if compgen -G "$src/stage2_*_alpha_0.4000.cuckoo" > /dev/null || compgen -G "$shared/stage2_*_alpha_0.4000.cuckoo" > /dev/null; then
                run touch "$shared/.ready_h2_lf0.4"
            fi
            promoted=$((promoted + 1))
        fi
    fi
    for t in "$shape_dir"/task_*; do
        if [[ "$t" =~ /task_(${active})_ ]]; then kept=$((kept + 1)); continue; fi
        run rm -rf "$t"
        deleted=$((deleted + 1))
    done
done
echo "shapes given a shared cache: $promoted; task folders deleted: $deleted; kept (job still queued): $kept"
if [[ -n "$DO" ]]; then echo "after: $(du -sh cuckoo_cache 2>/dev/null | cut -f1)"; else echo "dry run: DO_IT=1 bash clean_cuckoo_cache.sh to do it"; fi
