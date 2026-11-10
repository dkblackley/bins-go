#!/bin/bash
# run_me.sh: run this after every fix. It never deletes results.
#
#   bash run_me.sh                       # scifact msmarco nq fever dbpedia-entity, bins + pacmann + tree
#   bash run_me.sh nq fever              # only these datasets
#   DRY_RUN=1 bash run_me.sh             # show what it would submit, change nothing
#   STAGES="bins tree" bash run_me.sh    # only some methods
#
# What it does, in order:
#   1. git pull --ff-only (NO_PULL=1 skips it); if this script changed, it restarts itself
#   2. makes the *.sh / *.slurm scripts executable again (a checkout from a filesystem without exec bits drops them;
#      that is what made every tree run exit with 126)
#   3. rebuilds ./pir_app when it is missing, older than the Go code, or can't find its libraries (libngt.so.2 ...),
#      or REBUILD=1. It builds to pir_app.new and renames it, so running jobs keep their binary. Unlike build.sh it
#      deletes nothing (build.sh wipes ../results/bins_* and ../results/pacmann_*).
#   4. resubmits every config that has no metadata.json and that no pending/running job covers ("crashed" and
#      "not started" in check_runs.sh), through datasets/run_all.sh, with fresh memory/time estimates: runs that
#      OOMed or hit the time limit are re-estimated bigger (resource_usage.tsv). Queued jobs are left alone.
#   5. moves check_runs.sh's "since" to now, so failures from before this fix stop being listed
#   6. prints a short check_runs.sh summary (section 3)
# Everything it prints is also saved in logs/run_me/<timestamp>.log.

main() {
    set -uo pipefail
    local self; self=$(readlink -f "$0")
    cd "$(dirname "$self")"

    # ---- 1. update (before the log starts, so a restart doesn't log twice)
    if [[ -z "${NO_PULL:-}" && -z "${DRY_RUN:-}" ]]; then
        local before f
        before=$(git hash-object "$self")
        echo "== git pull"
        # pir_app is built here, not taken from git; and files whose only change is the exec bit would block the pull
        if git ls-files --error-unmatch pir_app >/dev/null 2>&1; then git checkout -q -- pir_app; fi
        for f in $(git diff --name-only); do
            git -c core.fileMode=false diff --quiet -- "$f" && git checkout -q -- "$f"
        done
        git pull --ff-only || { echo "git pull failed (local changes? see git status). Fix it, or NO_PULL=1 bash run_me.sh"; exit 1; }
        if [[ "$before" != "$(git hash-object "$self")" ]]; then
            echo "run_me.sh changed, restarting it"
            NO_PULL=1 exec bash "$self" "$@"
        fi
    fi

    local ts; ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p logs/run_me logs/check
    exec > >(tee -a "logs/run_me/$ts.log") 2>&1
    local started; started=$(date +%FT%T)
    local datasets="${*:-scifact msmarco nq fever dbpedia-entity}"
    local stages="${STAGES:-bins pacmann tree}"
    echo "run_me.sh $started  datasets: $datasets  stages: $stages${DRY_RUN:+  (DRY_RUN: nothing is submitted)}"
    git log -1 --format='   code: %h %ci %s'

    # ---- 2. exec bits
    chmod u+x ./*.sh ./*.slurm datasets/*.sh datasets/*.slurm 2>/dev/null || true

    # ---- 3. pir_app
    echo "== pir_app"
    local why=""
    if [[ -n "${REBUILD:-}" ]]; then why="REBUILD=1"
    elif [[ ! -x ./pir_app ]]; then why="missing"
    elif ldd ./pir_app 2>/dev/null | grep -q 'not found'; then
        why="can't find $(ldd ./pir_app | awk '/not found/ {print $1}' | paste -sd' ')"
    elif [[ -n "$(find . -path ./third_party -prune -o \( -name '*.go' -o -name go.mod -o -name go.sum \) -newer ./pir_app -print -quit)" ]]; then
        why="older than the Go code"
    fi
    if [[ -z "$why" ]]; then
        echo "   up to date"
    elif [[ -n "${DRY_RUN:-}" ]]; then
        echo "   would rebuild ($why)"
    else
        echo "   rebuilding ($why)"
        # go_env.sh exits when it finds no Go, so build in a subshell
        if ( source ./go_env.sh && go build -o pir_app.new ./main.go ) && mv -f pir_app.new pir_app; then
            if ldd ./pir_app | grep 'not found'; then
                echo "   pir_app still can't find the libraries above: run setup_pacmann.sh (builds them into ~/opt), then run_me.sh again"
                exit 1
            fi
            echo "   built: $(ls -l --time-style=+%F_%T pir_app | awk '{print $6}')"
        else
            rm -f pir_app.new
            echo "   build FAILED, nothing submitted"; exit 1
        fi
    fi

    # ---- 4. resubmit
    # queued configs, worked out once and shared with run_all.sh and check_runs.sh
    source ./slurm_common.sh
    export QUEUED_FILE="logs/run_me/${ts}_queued.tsv"
    queued_dirs > "$QUEUED_FILE"
    echo "== resubmitting (jobs in the queue now cover $(grep -vc '^?' "$QUEUED_FILE") configs; those are skipped)"
    STAGES="$stages" bash datasets/run_all.sh $datasets 2>&1 | grep -vE '^Queued\. Watch'

    # ---- 5. since
    if [[ -z "${DRY_RUN:-}" ]]; then
        echo "$started" > logs/check/since
        echo "== check_runs.sh now lists failures after $started only (logs/check/since)"
    fi

    # ---- 6. summary
    echo
    env -u QUEUED_FILE METHODS="$stages" bash check_runs.sh $datasets 2>/dev/null | awk '/^==== 3\./ { p = 1 } /^==== 4\./ { p = 0 } p'
    echo
    echo "Saved to logs/run_me/$ts.log. Later: bash check_runs.sh"
}

# main is called on the last line, so bash has read the whole file before git pull can change it
main "$@"
