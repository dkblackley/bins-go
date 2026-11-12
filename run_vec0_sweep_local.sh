#!/bin/bash
# run_vec0_sweep_local.sh: the bins vec0 batch / rounds sweep of one dataset, run on this machine (simulated PIR),
# into the simulated results folder with the same folder names the cluster uses, so plotting picks them up as is.
#
#   bash run_vec0_sweep_local.sh scifact            # SciFact's own grid (1,800 configs, seconds each)
#   bash run_vec0_sweep_local.sh trec-covid 4       # 4 runs at a time (default: JOBS or nproc / 2)
#   DRY_RUN=1 bash run_vec0_sweep_local.sh scifact  # list the configs, run nothing
#   FORCE=1 ...                                     # run configs that already have a metadata.json again
#
# The configs are exactly run_bins_low/high.slurm --list-raw's vb<B>r<R>i<I> entries (add_vec0_sweep), so every
# VEC0_SWEEP_* / VEC0_SCIFACT_* override works here too. Results: $SIM_RESULTS/bins_vec0_<dataset>_k<K>_bs<BS>_dpb<DPB>
# _vb<B>_vr<R>_ib<I>/ (metadata.json, results.json, results_reRank.json, log.txt). pir_app is built from the current
# code into bin/pir_app.local first (REBUILD=0 reuses it). A failed config is listed in $SIM_RESULTS/vec0_sweep_fails.txt.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

d=${1:?usage: run_vec0_sweep_local.sh <dataset> [parallel jobs]}
JOBS=${2:-${JOBS:-$(( $(nproc) / 2 ))}}
DATASETS_ROOT=${DATASETS_ROOT:-$HOME/Nextcloud/10TB-STHDD/datasets}
SIM_RESULTS=${SIM_RESULTS:-$DATASETS_ROOT/results_simulated}
BIN=${BIN:-$PWD/bin/pir_app.local}

case $d in
    msmarco) n=8841823 ;; scifact) n=5183 ;; trec-covid) n=171332 ;;
    *) n=$(cat "$DATASETS_ROOT/$d/num_docs.txt" 2>/dev/null) || { echo "no $DATASETS_ROOT/$d/num_docs.txt" >&2; exit 1; } ;;
esac

# the sweep's configs, from the same code the cluster uses (module is a cluster command: stubbed here)
module() { :; }; export -f module
list=$(for t in low high; do
    DATASET_DIR="$DATASETS_ROOT" BINS_DATASETS="$d" BINS_VECS=false SLURM_SUBMIT_DIR="$PWD" bash "run_bins_$t.slurm" --list-raw
done | awk '$5 ~ /^vb[0-9]+r[0-9]+i[0-9]+$/')
[[ -n "$list" ]] || { echo "no vec0 sweep configs for $d (VEC0_SWEEP_DATASETS?)" >&2; exit 1; }

todo=$(while read -r ds k bs dpb vec; do
    [[ $vec =~ ^vb([0-9]+)r([0-9]+)i([0-9]+)$ ]]
    out="$SIM_RESULTS/bins_vec0_${ds}_k${k}_bs${bs}_dpb${dpb}_vb${BASH_REMATCH[1]}_vr${BASH_REMATCH[2]}_ib${BASH_REMATCH[3]}"
    [[ -s "$out/metadata.json" && -z "${FORCE:-}" ]] && continue
    echo "$k $bs $dpb ${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]} $out"
done <<< "$list")
echo "$d: $(grep -c . <<< "$list") sweep configs, $(grep -c . <<< "$todo") to run, $JOBS at a time -> $SIM_RESULTS"
[[ -n "${DRY_RUN:-}" ]] && { echo "$todo" | awk '{print $NF}'; exit 0; }
[[ -n "$todo" ]] || exit 0

if [[ "${REBUILD:-1}" == 1 || ! -x "$BIN" ]]; then
    mkdir -p "$(dirname "$BIN")"
    ( source ./go_env.sh >/dev/null 2>&1; go build -o "$BIN" ./main.go ) || { echo "build failed" >&2; exit 1; }
fi

run_one() {   # k bs dpb vb vr ib out
    local k=$1 bs=$2 dpb=$3 vb=$4 vr=$5 ib=$6 out=$7
    mkdir -p "$out"
    if "$BIN" -n "$N_DOCS" -t bins -name "$DATASET" -k "$k" -binSize "$bs" -docsPerBin "$dpb" \
            -dataset "$DATASETS_ROOT" -debug 1 -vectors=false -dim 192 -simulate \
            -vecBatch "$vb" -vecRounds "$vr" -idBatch "$ib" -vecOrder rr \
            -outDir "$out" -outFile results > "$out/log.txt" 2>&1 && [[ -s "$out/metadata.json" ]]; then
        echo "ok   ${out##*/}"
    else
        echo "FAIL ${out##*/}"; echo "${out##*/}" >> "$SIM_RESULTS/vec0_sweep_fails.txt"
    fi
}
export -f run_one
export BIN DATASETS_ROOT SIM_RESULTS N_DOCS="$n" DATASET="$d"
SECONDS=0
echo "$todo" | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ | awk -v t="$(grep -c . <<< "$todo")" '{ printf "[%d/%d] %s\n", NR, t, $0; fflush() }'
echo "done in $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s"
