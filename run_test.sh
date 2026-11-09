#!/bin/bash
# Local (non-Slurm) runner for the "test" method: text-only PIR that fetches k random docs per query.
# Loops over every dataset x k combination, one run at a time.
#
# Usage:
#   ./run_test_local.sh            # run everything
#   ./run_test_local.sh --count    # just print the list of configs
#
# Override anything below from the shell, e.g.
#   RESULTS_ROOT=~/results K_LIST="10 50" DATASETS="scifact" ./run_test_local.sh

set -uo pipefail # no -e: one failed run shouldn't kill the rest of the sweep
cd "$(dirname "$0")"

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_CFLAGS="-I${NGT_PREFIX}/include"
export CGO_CXXFLAGS="-std=c++11"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
# ------------------------------------------------------------------ settings you'll want to edit
K_LIST=(${K_LIST:-10 50 100})
DATASETS=(${DATASETS:-scifact}) # add trec-covid if wanted
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/Nextcloud/10TB-STHDD/datasets/results}"
DATASET_DIR="${DATASET_DIR:-$HOME/Nextcloud/10TB-STHDD/datasets}"
APP="${APP:-./pir_app}"
BUILD="${BUILD:-1}" # 1 = go build before starting, 0 = use existing $APP
DEBUG="${DEBUG:-1}"
DIM="${DIM:-192}"
export GOMAXPROCS="${GOMAXPROCS:-$(nproc)}"

# Only needed if the binary links against NGT/HNSW (as on Hopper) - point these at your local installs.
# export NGT_PREFIX="$HOME/opt/ngt"
# export HNSW_PREFIX="$HOME/opt/hnsw"
# export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
# ------------------------------------------------------------------

method="test"

db_size_of() {
    case "$1" in
        msmarco) echo 8841823 ;;
        scifact) echo 5183 ;;
        trec-covid) echo 171332 ;;
        *) echo "" ;;
    esac
}

if [[ "${1:-}" == "--count" ]]; then
    for d in "${DATASETS[@]}"; do for k in "${K_LIST[@]}"; do echo "${method}_${d}_k${k}"; done; done
    echo "$(( ${#DATASETS[@]} * ${#K_LIST[@]} )) configs"
    exit 0
fi

if [[ "$BUILD" == "1" ]]; then
    echo "Building $APP ..."
    go build -o "$APP" . || { echo "Build failed"; exit 1; }
fi

FAIL_FILE="$PWD/${method}_local_fails"
current=""
trap 'echo; echo "Interrupted during ${current:-startup}"; kill "${app_pid:-}" 2>/dev/null; exit 130' INT TERM

total=$(( ${#DATASETS[@]} * ${#K_LIST[@]} ))
i=0
n_failed=0

for dataset in "${DATASETS[@]}"; do
    DB_SIZE="$(db_size_of "$dataset")"
    if [[ -z "$DB_SIZE" ]]; then
        echo "Unknown dataset: $dataset - skipping"
        continue
    fi

    for k_val in "${K_LIST[@]}"; do
        i=$((i + 1))
        current="${dataset} k=${k_val}"
        OUT_DIR="${RESULTS_ROOT}/${method}_${dataset}_k${k_val}"
        mkdir -p "$OUT_DIR"

        header=$(cat <<EOF
========================================
Run:           $i / $total
Method:        $method
Dataset:       $dataset
DB Size:       $DB_SIZE
K Value:       $k_val
Output Dir:    $OUT_DIR
Started:       $(date '+%F %T')
========================================
EOF
)
        echo "$header"
        echo "$header" > "${OUT_DIR}/run_info.txt"

        # stdout -> terminal + execution.log, stderr -> error.log (same split as the Slurm .out/.err files)
        "$APP" \
            -n "$DB_SIZE" \
            -t "$method" \
            -name "$dataset" \
            -k "$k_val" \
            -outDir "$OUT_DIR" \
            -outFile "results" \
            -dataset "$DATASET_DIR" \
            -debug "$DEBUG" \
            -dim "$DIM"  &
        app_pid=$!
        wait "$app_pid"
        rc=$?

        echo "Finished:      $(date '+%F %T') (rc=$rc)" >> "${OUT_DIR}/run_info.txt"
        if [[ $rc -ne 0 ]]; then
            n_failed=$((n_failed + 1))
            echo "${method}_${dataset}_k${k_val} rc=${rc} $(date '+%F %T')" >> "$FAIL_FILE"
            echo "!! Failed (rc=$rc) - see ${OUT_DIR}/error.log"
        fi
    done
done

echo "All done: $((total - n_failed)) / $total succeeded."
[[ $n_failed -gt 0 ]] && echo "Failures logged in $FAIL_FILE"
exit 0