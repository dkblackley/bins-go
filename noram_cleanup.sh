#!/bin/bash
# noram_cleanup_job: delete the -noram run folders (temp_storage/<run id>/, diskstore.Init) that belong to this Slurm
# task. pir_app deletes its own folder when it finishes or gets SIGTERM, but not when it is SIGKILLed (OOM kill, the
# time limit's final kill, scancel's kill after the grace period) or crashes: those folders, often tens of GB each,
# piled up to 14 TB on 2026-11-10. The batch shell survives its step being killed, so run_bins_low/high.slurm and
# run_pacmann.slurm call this from their EXIT trap. A folder is this task's when its owner file names the task
# ("slurm_job=<array job>_<task> "); nothing else is touched.
noram_cleanup_job() {
    local base="${NORAM_DIR:-temp_storage}" tag="slurm_job=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} " owner i
    [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -d "$base" ]] || return 0
    # give a step that was just sent SIGTERM a moment to exit (and drop its mmaps) first
    for (( i = 0; i < 30; i++ )); do
        [[ -n "${srun_pid:-}" ]] && kill -0 "$srun_pid" 2>/dev/null || break
        sleep 1
    done
    for owner in "$base"/*/owner; do
        [[ -f "$owner" ]] && grep -qF "$tag" "$owner" 2>/dev/null || continue
        echo "noram cleanup: removing $(dirname "$owner") ($(du -sh "$(dirname "$owner")" 2>/dev/null | cut -f1))"
        rm -rf -- "$(dirname "$owner")"
    done
    return 0
}

# noram_sweep_dead [dry]: delete every run folder in temp_storage whose Slurm task is no longer running (run_me.sh
# calls it on every run, so leftovers can't pile up again). A folder is kept if its task is in squeue as RUNNING, if
# it has no owner file yet and is under an hour old (a run starting up), or if it is not from a Slurm array task
# (a local run) and under a day old. With "dry" it only reports. Prints what it removed and how much was freed.
noram_sweep_dead() {
    local dry="${1:-}" base="${NORAM_DIR:-temp_storage}" d owner job n_dead=0 n_live=0 kb kb_dead=0 now
    [[ -d "$base" ]] || { echo "   $base: none"; return 0; }
    command -v squeue >/dev/null || { echo "   no squeue here, temp_storage not checked"; return 0; }
    local running; running=" $(squeue -u "$USER" -h -r -t RUNNING -o %i 2>/dev/null | tr '\n' ' ') "
    now=$(date +%s)
    for d in "$base"/*/; do
        d=${d%/}; [[ -d "$d" ]] || continue
        owner="$d/owner"
        if [[ ! -f "$owner" ]]; then
            (( now - $(stat -c %Y "$d") > 3600 )) || { n_live=$((n_live + 1)); continue; }
        else
            job=$(grep -oE 'slurm_job=[0-9]*_[0-9]*' "$owner" | cut -d= -f2)
            if [[ "$job" == _ || -z "$job" ]]; then
                (( now - $(stat -c %Y "$owner") > 86400 )) || { n_live=$((n_live + 1)); continue; }
            elif [[ "$running" == *" $job "* ]]; then
                n_live=$((n_live + 1)); continue
            fi
        fi
        kb=$(du -sk "$d" 2>/dev/null | cut -f1); kb_dead=$((kb_dead + ${kb:-0})); n_dead=$((n_dead + 1))
        [[ -n "$dry" ]] || rm -rf -- "$d"
    done
    printf '   %s: %d folders of running jobs kept, %d dead %s (%s GB)\n' "$base" "$n_live" "$n_dead" \
        "$([[ -n "$dry" ]] && echo "found, would delete" || echo deleted)" "$(( kb_dead / 1048576 ))"
}
