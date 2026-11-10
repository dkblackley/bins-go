#!/bin/bash -l
#SBATCH --job-name=Tiptoe_MSv2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --array=0-63%8
#SBATCH --output=logs/TiptoeV2_%A_%a.out
#SBATCH --error=logs/TiptoeV2_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

# Tiptoe on msmarco-v2 and its subsamples (simulated; see baselines/README.md). The other datasets use run_tiptoe.slurm.
#
# Same grid as run_tiptoe.slurm (DPC x MA x BUNDLE x HINT x k), but organised for 50M-138M docs:
#   * One task per preprocessed DB (dataset, clusters, MA, BUNDLE). It preprocesses once, then runs the 4 (HINT, k)
#     simulations itself, which take minutes. In run_tiptoe.slurm those 4 would be separate tasks, 3 of them sitting
#     on an allocation waiting for the preprocessing lock.
#   * DPC values that give the same number of clusters are merged before submitting. Tiptoe caps clusters at 25196,
#     so on msmarco-v2 DPC 250..5000 are all the same 25196-cluster DB.
#   * The DB is written as compact arrays (--format npy, ~200 B per doc copy, ~34 GB on msmarco-v2) and deleted after
#     its simulations unless KEEP_TT=1. Only the small files (meta, centroids, bundle sizes) are kept.
#   * k-means is shared across MA/BUNDLE through --kmeans-cache, so there's one k-means per (dataset, clusters).
#
# Memory: preprocessing reads the 106 GB embedding file memory-mapped. Peak is ~4 GB + <100 B per doc
# (measured on a 2M-doc set), so ~15 GB on msmarco-v2. Page cache helps speed but isn't required, so 32G is enough.
# Time: assigning 138M docs to 25196 centroids plus bundling is the slow part, estimated at 2-4 h on 16 cores.
#
# Usage:
#   bash run_tiptoe_msmarco_v2.slurm --count          # prints the --array range (64 for the default datasets)
#   sbatch run_tiptoe_msmarco_v2.slurm
#   TIPTOE_V2_DATASETS="msmarco-v2-25m msmarco-v2-100m" bash run_tiptoe_msmarco_v2.slurm --count

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATASET_DIR="${DATASET_DIR:-../../datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/dblackle/results}"
TIPTOE_V2_DATASETS="${TIPTOE_V2_DATASETS:-msmarco-v2-25m msmarco-v2-50m msmarco-v2}"

# One line per task: dataset N clusters dpc ma bundle (clusters = preprocess_tiptoe.py's own rule, deduplicated)
mapfile -t TASKS < <(python3 - "$DATASET_DIR" $TIPTOE_V2_DATASETS <<'EOF'
import os, re, sys
root, names = sys.argv[1], sys.argv[2:]
for d in names:
    f = os.path.join(root, d, "num_docs.txt")
    sub = re.fullmatch(r"msmarco-v2-(\d+)m", d)
    if os.path.exists(f):
        n = int(open(f).read().split()[0])
    elif sub:   # a subsample that isn't prepared yet (run_all.sh submits this behind its prep job): exactly N million
        n = int(sub.group(1)) * 1_000_000
    elif d == "msmarco-v2":
        n = 138_364_198
    else:
        sys.exit(f"No {f}, run datasets/prep_datasets.slurm for {d} first")
    seen = set()
    for dpc in (250, 2500, 40000):   # 2026-11-09, was 250 500 1000 2500 5000 10000 40000
        c = min(max(1, round(n / dpc)), 25196)        # TIPTOE_MAX_CLUSTERS
        for ma in (1, 2):
            for bundle in (40, 320):   # was 40 80 160 320
                if (c, ma, bundle) not in seen:
                    seen.add((c, ma, bundle))
                    print(d, n, c, dpc, ma, bundle)
EOF
)

if [[ "${1:-}" == "--count" ]]; then
    echo "${#TASKS[@]} tasks for: $TIPTOE_V2_DATASETS (use --array=0-$(( ${#TASKS[@]} - 1 ))%8)"
    printf '  %s\n' "${TASKS[@]}" | awk '{print "  " $1, "C=" $3, "(dpc " $4 ")", "ma=" $5, "bundle=" $6}' | sort -u | head -80
    exit 0
fi

read -r dataset DB_SIZE C dpc ma bundle <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"
source datasets/env.sh     # conda env main (same as run_tiptoe.slurm; the old hard-coded miniconda3 path doesn't exist)

FAIL_FILE="${SLURM_SUBMIT_DIR:-$PWD}/tiptoe_v2_fails"
finished=0
trap '[[ $finished -eq 1 ]] || echo "${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} $(date "+%F %T")" >> "$FAIL_FILE"' EXIT
trap 'kill "${child:-}" 2>/dev/null; exit 143' TERM INT

TT_DIR="${DATASET_DIR}/tiptoe/${dataset}_c${C}_m${ma}_b${bundle}_npy"
MA_PRED=$(( ma < C ? ma : C ))
cat <<EOF
========================================
Array Task ID:     ${SLURM_ARRAY_TASK_ID:-}
Dataset:           $dataset ($DB_SIZE docs)
Clusters:          $C (first DPC giving it: $dpc)
Multi-assign:      $ma
Bundle size:       $bundle
Tiptoe dir:        $TT_DIR
========================================
EOF

out_name() { echo "${RESULTS_ROOT}/tiptoe_${dataset}_c$1_m$2_b${bundle}_h$3_k$4"; }
all_done() {   # $1 = clusters, $2 = multi-assign as applied
    for hint in 125 500; do for k in 10 100; do
        [[ -f "$(out_name "$1" "$2" "$hint" "$k")/metadata.json" ]] || return 1
    done; done
}
if all_done "$C" "$MA_PRED"; then
    echo "all 4 results already exist, nothing to do"; finished=1; exit 0
fi

# 1) Preprocess (flock in case a resubmitted task overlaps a running one)
mkdir -p "$(dirname "$TT_DIR")"
(
    flock 9
    if [[ ! -f "${TT_DIR}/tiptoe_meta.json" ]]; then
        python baselines/tiptoe/preprocess_tiptoe.py --dataset "$DATASET_DIR" --name "$dataset" \
            --num-clusters "$C" --multi-assign "$ma" --bundle-size "$bundle" --format npy --out "$TT_DIR" \
            --kmeans-cache "${DATASET_DIR}/tiptoe/kmeans_cache"
    fi
) 9>"${TT_DIR}.lock"
# clusters / multi-assign as applied (empty clusters are dropped, multi-assign is capped at the cluster count)
read -r C_USED MA_USED < <(python -c "import json;m=json.load(open('${TT_DIR}/tiptoe_meta.json'));print(m['num_clusters'],m['multi_assign'])")

# 2) The 4 simulations of this DB
for hint in 125 500; do
    for k in 10 100; do
        OUT_DIR=$(out_name "$C_USED" "$MA_USED" "$hint" "$k")
        if [[ -f "${OUT_DIR}/metadata.json" ]]; then echo "skip $OUT_DIR (exists)"; continue; fi
        echo "=== hint=$hint k=$k -> $OUT_DIR"
        python baselines/tiptoe/simulate_tiptoe.py \
            --dataset "$DATASET_DIR" --name "$dataset" --tiptoe-dir "$TT_DIR" --k "$k" \
            --emb-hint-mb "$hint" --results-root "$RESULTS_ROOT" \
            --parallelism "${SLURM_CPUS_PER_TASK:-1}" &
        child=$!
        wait "$child"
    done
done

# 3) Free the disk: the arrays are only needed to (re)run these simulations
if [[ -z "${KEEP_TT:-}" ]]; then
    rm -f "${TT_DIR}/emb.npy" "${TT_DIR}/rows.npy" "${TT_DIR}/bundle.npy"
    echo "deleted the big arrays in $TT_DIR (KEEP_TT=1 keeps them)"
fi

LOG_DIR="${RESULTS_ROOT}/tiptoe_v2_logs"
mkdir -p "$LOG_DIR"
mv "logs/TiptoeV2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${LOG_DIR}/${dataset}_c${C}_m${ma}_b${bundle}.log" || true
mv "logs/TiptoeV2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${LOG_DIR}/${dataset}_c${C}_m${ma}_b${bundle}.err" || true
finished=1
echo "All done for this node."
