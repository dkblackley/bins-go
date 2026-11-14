#!/bin/bash -l
#SBATCH --job-name=Bins_Low
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-07:30:00
#SBATCH --array=0-615
#SBATCH --output=logs/Bins_%A_%a.out
#SBATCH --error=logs/Bins_%A_%a.err
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=amd

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

set -euo pipefail
# cd "$SLURM_SUBMIT_DIR"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"
# pir_app finds libngt/libhnsw through its rpath; this also covers a pir_app built without go_env.sh
export LD_LIBRARY_PATH="${NGT_PREFIX:-$HOME/opt/ngt-amd}/lib:${HNSW_PREFIX:-$HOME/opt/hnsw}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

DATASET_DIR="${DATASET_DIR:-../../datasets}"
# dataset sizes, input checks, RAM/time model and the usage log (shared with the tree runs and datasets/run_all.sh)
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_common.sh"
source "${SLURM_SUBMIT_DIR:-$PWD}/noram_cleanup.sh"
# -n for each dataset: hardcoded for the originals, read from num_docs.txt (written by datasets/prepare_dataset.py)
# for everything else (nq, hotpotqa, fever, msmarco-v2, msmarco-v2-100m, ...)
db_size_of() {
    case "$1" in
        msmarco)    echo 8841823 ;;
        scifact)    echo 5183 ;;
        trec-covid) echo 171332 ;;
        *)
            local f="${DATASET_DIR}/$1/num_docs.txt"
            [[ -f "$f" ]] || { echo "No $f, run datasets/prep_datasets.slurm for $1 first" >&2; exit 1; }
            cat "$f" ;;
    esac
}

# 2026-11-09: cut to 8 values (min and max kept); the full list was
# (100 500 1000 5000 10000 50000 100000 500000 1000000 5000000 10000000)
BS_LIST=(100 1000 10000 100000 500000 1000000 5000000 10000000)
DPB_LIST=(10 25 50 100 250 500 1000 2500)   # was 10 25 50 75 100 250 500 1000 1500 2000 2500
# 2026-11-11: msmarco-v2, msmarco-v2-25m and msmarco-v2-50m only: two bin sizes past BS_LIST's largest (10M)
BS_LIST_V2_EXTRA=(20000000 50000000)
# BS_LIST=(100 500 1000 5000 10000 50000 100000 500000 1000000)

DATA_ARR=()
K_ARR=()
BS_ARR=()
DPB_ARR=()
VEC_ARR=()

TIER="low"

TIER="${TIER:?set TIER to low or high}"

tier_of() {
        local d=$1 k=$2 bs=$3 dpb=$4 vec=$5
    if [[ "$d" == msmarco-v2* || "$d" == miracl-en ]]; then
        echo "high"   # 25M-138M docs: everything needs bigmem
    elif [[ "$d" =~ ^(msmarco|nq|hotpotqa|fever|dbpedia-entity)$ && "$dpb" -ge 1000 ]]; then
        echo "high"
    elif [[ "$vec" == "true" && "$dpb" -ge 250 ]]; then
        echo "high"
    elif [[ "$vec" == "text" && "$bs" -ge 1000000 && "$dpb" -ge 1000 ]]; then
        echo "high"   # >= 1e9 (bin, doc) slots of ~75 B
        echo "high"
    else
        echo "low"
    fi
}

# Helper function to generate the 32 ablations per dataset
add_bins_ablations() {
    local d=$1

    local sizes=("${BS_LIST[@]}")
    [[ "$d" =~ ^msmarco-v2(-25m|-50m)?$ ]] && sizes+=("${BS_LIST_V2_EXTRA[@]}")

    for vec in $BINS_VECS; do
        for k in 10 100; do
            for bs in "${sizes[@]}"; do
                for dpb in "${DPB_LIST[@]}"; do
                    [[ "$(tier_of "$d" "$k" "$bs" "$dpb" "$vec")" == "$TIER" ]] || continue
                    # 25M-138M docs: fewer than 100k bins can only hold a tiny fraction of the corpus, skip them
                    [[ ( "$d" == msmarco-v2* || "$d" == miracl-en ) && "$bs" -lt 100000 ]] && continue
                    DATA_ARR+=("$d"); K_ARR+=("$k"); BS_ARR+=("$bs"); DPB_ARR+=("$dpb"); VEC_ARR+=("$vec")
                done
            done
        done
    done

}

# Build the setups for all datasets. Override to add the new ones, e.g.
#   BINS_DATASETS="nq hotpotqa fever" bash run_bins_high.slurm --count     # prints the --array range to use
#   BINS_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_bins_high.slurm
# Big corpora: BINS_DATASETS="msmarco-v2-25m msmarco-v2-50m msmarco-v2-100m msmarco-v2"
BINS_DATASETS="${BINS_DATASETS:-msmarco scifact}"
# Layouts: true = 1 DB of embeddings (vec1), false = ID DB + embedding DB (vec0), text = 1 DB of compressed doc
# text with a client BM25 re-rank (-text, bins/TEXT.md; needs no embeddings). E.g. BINS_VECS=text for text only.
BINS_VECS="${BINS_VECS:-true false text}"
# -text coder settings (bins/TEXT.md); non-default ones get a _<format>o<order> suffix on the results folder
TEXT_FORMAT="${TEXT_FORMAT:-bow}"
TEXT_ORDER="${TEXT_ORDER:-3}"
for d in $BINS_DATASETS; do
    add_bins_ablations "$d"
done

# Two-DB (vec0) batch / rounds sweep (2026-11-11), IN ADDITION to the grid above: that grid's vec0 runs use the
# automatic batching (ID batch 24; embedding batch 3 x the largest bin, 1 round, or 3000 x 2 rounds past it; SciFact:
# the largest bin x 6 rounds, or 500 x 15) and stay the baseline. The client hint of a batched Piano DB is
# ~sqrt(batch x rows/2) entries (batch/2 partitions, each with its own hints), so the batch, not the DB size, is
# what makes vec0's hints big (PACMANN's batch is its neighbour count, 8-48).
# Each config is one vec value vb<B>r<R>i<I>: -vecBatch B lookups per embedding-DB round, -vecRounds R rounds,
# -idBatch I (candidates are sent round robin, rank 1 of every query term's bin first, the default now). Results:
# bins_vec0_<dataset>_k<K>_bs<BS>_dpb<DPB>_vb<B>_vr<R>_ib<I>.
# Pruned per dpb, with cap = VEC0_SWEEP_MAX_TERMS x dpb (about the most candidates a query has): a batch over the cap
# (almost all padding) or rounds past what the cap needs (B x (R-1) >= cap: the extra rounds fetch only padding);
# and a batch too big for the corpus (Piano partitions need >= 16 rows; pir_app also cuts it at run time when the
# pruned embedding DB is smaller).
#   every dataset   batch 10 / 50 / 100 / 2500 x rounds 1 / 3 / 10 / 30, ID batch 24 (as before), k 10, at three
#                   bin sizes of the grid (the batch's effect barely depends on bs): VEC0_SWEEP_<VB|VR|IB|BS|DPB|K>
#   scifact         its own grid, the same size as the others (236 configs, 4 bin sizes x every dpb): the small
#                   batches and many rounds where a 1,800-config local sweep found its smallest hints (batch 4 x 30
#                   rounds with ID batch 2 led its frontier); bs 1M is left out (its ID hint alone is 13-233 MB,
#                   PACMANN's whole hint 12 MB): VEC0_SCIFACT_<VB|VR|IB|BS|DPB|K>
# VEC0_SWEEP_DATASETS="" turns the sweep off; e.g. VEC0_SWEEP_DATASETS="msmarco scifact" limits it.
VEC0_SWEEP_DATASETS="${VEC0_SWEEP_DATASETS-all}"
VEC0_SWEEP_MAX_TERMS="${VEC0_SWEEP_MAX_TERMS:-8}"
vec0_sweep_param() {   # dataset key -> its values (VEC0_SCIFACT_<key> for scifact, VEC0_SWEEP_<key> otherwise)
    local d=$1 key=$2 v
    if [[ "$d" == scifact ]]; then
        v="VEC0_SCIFACT_$key"
        case $key in
            K)   echo "${!v:-10}" ;;
            BS)  echo "${!v:-100 1000 10000 100000}" ;;
            DPB) echo "${!v:-${DPB_LIST[*]}}" ;;
            VB)  echo "${!v:-4 16 50}" ;;
            VR)  echo "${!v:-3 10 30}" ;;
            IB)  echo "${!v:-2}" ;;
        esac
    else
        v="VEC0_SWEEP_$key"
        case $key in
            K)   echo "${!v:-10}" ;;
            BS)  echo "${!v:-100000 1000000 10000000}" ;;
            DPB) echo "${!v:-${DPB_LIST[*]}}" ;;
            VB)  echo "${!v:-10 50 100 2500}" ;;
            VR)  echo "${!v:-1 3 10 30}" ;;
            IB)  echo "${!v:-24}" ;;
        esac
    fi
}
add_vec0_sweep() {
    local d=$1 n k bs dpb ib vb vr cap
    n=$(db_size_of "$d" 2>/dev/null) || n=""
    for k in $(vec0_sweep_param "$d" K); do for bs in $(vec0_sweep_param "$d" BS); do
        [[ ( "$d" == msmarco-v2* || "$d" == miracl-en ) && "$bs" -lt 100000 ]] && continue   # as in the grid
        for dpb in $(vec0_sweep_param "$d" DPB); do
            cap=$(( VEC0_SWEEP_MAX_TERMS * dpb ))
            for ib in $(vec0_sweep_param "$d" IB); do for vb in $(vec0_sweep_param "$d" VB); do for vr in $(vec0_sweep_param "$d" VR); do
                (( vb <= cap && vb * (vr - 1) < cap )) || continue
                [[ -z "$n" ]] || (( vb <= 2 * n / 16 )) || continue
                [[ "$(tier_of "$d" "$k" "$bs" "$dpb" false)" == "$TIER" ]] || continue
                DATA_ARR+=("$d"); K_ARR+=("$k"); BS_ARR+=("$bs"); DPB_ARR+=("$dpb"); VEC_ARR+=("vb${vb}r${vr}i${ib}")
            done; done; done
        done
    done; done
}
if [[ " $BINS_VECS " == *" false "* ]]; then
    for d in $BINS_DATASETS; do
        [[ "$VEC0_SWEEP_DATASETS" == all || " $VEC0_SWEEP_DATASETS " == *" $d "* ]] && add_vec0_sweep "$d"
    done
fi

# Query-term sweep (2026-11-13), IN ADDITION to the above: PILLAR-Bin's answer to the tree's q* ablation. -maxTerms N
# keeps the first N analysed query terms (one ID-DB lookup each), as the tree's -q keeps the first q*. Three vec0
# configs per dataset, embedding batch 100 x 1 / 10 / 30 rounds (2 / 11 / 31 PIR rounds) on the dataset's best
# (bs, dpb) family of the sweep above, each at N = MAXTERMS_LIST. m0 (every term, the same as no -maxTerms) is run
# again next to them so the baseline comes from the same code. Vec value vb<B>r<R>i<I>m<N>, results folder
# bins_vec0_<dataset>_k10_bs<BS>_dpb<DPB>_vb<B>_vr<R>_ib<I>_mt<N> (plotting skips _mt runs unless INCLUDE_MAXTERMS=1;
# plotting/query_terms_table.py reads them). MAXTERMS_DATASETS="" turns it off; MAXTERMS_BASE_<dataset>="bs dpb"
# (dataset with - as _), MAXTERMS_LIST, MAXTERMS_VB, MAXTERMS_VR, MAXTERMS_IB override.
MAXTERMS_DATASETS="${MAXTERMS_DATASETS-all}"
MAXTERMS_LIST="${MAXTERMS_LIST:-0 1 2 3 4 6 8}"
maxterms_base() {   # dataset -> "bs dpb" (the family with the best vec0-sweep MRR in results_simulated, 2026-11-13)
    local v="MAXTERMS_BASE_${1//-/_}"
    [[ -n "${!v:-}" ]] && { echo "${!v}"; return; }
    case $1 in
        scifact)        echo "1000 1000" ;;
        msmarco)        echo "1000000 2500" ;;
        nq)             echo "1000000 500" ;;
        fever)          echo "10000000 500" ;;
        dbpedia-entity) echo "1000000 1000" ;;
        msmarco-v2-25m) echo "1000000 2500" ;;
        msmarco-v2-50m|msmarco-v2) echo "10000000 2500" ;;
    esac
}
add_maxterms_sweep() {
    local d=$1 bs dpb vb vr mt base
    base=$(maxterms_base "$d"); [[ -n "$base" ]] || return 0
    read -r bs dpb <<< "$base"
    [[ "$(tier_of "$d" 10 "$bs" "$dpb" false)" == "$TIER" ]] || return 0
    for vb in ${MAXTERMS_VB:-100}; do for vr in ${MAXTERMS_VR:-1 10 30}; do for mt in $MAXTERMS_LIST; do
        DATA_ARR+=("$d"); K_ARR+=(10); BS_ARR+=("$bs"); DPB_ARR+=("$dpb"); VEC_ARR+=("vb${vb}r${vr}i${MAXTERMS_IB:-24}m${mt}")
    done; done; done
}
if [[ " $BINS_VECS " == *" false "* ]]; then
    for d in $BINS_DATASETS; do
        [[ "$MAXTERMS_DATASETS" == all || " $MAXTERMS_DATASETS " == *" $d "* ]] && add_maxterms_sweep "$d"
    done
fi

# One "dataset k bs dpb vec" line per config of this tier (datasets/run_all.sh reads it to build CONFIG_LISTs)
if [[ "${1:-}" == "--list-raw" ]]; then
    for i in "${!DATA_ARR[@]}"; do echo "${DATA_ARR[$i]} ${K_ARR[$i]} ${BS_ARR[$i]} ${DPB_ARR[$i]} ${VEC_ARR[$i]}"; done
    exit 0
fi
if [[ -n "${CONFIG_LIST:-}" && "${1:-}" == "--count" ]]; then
    echo "$(grep -c . "$CONFIG_LIST") configs in $CONFIG_LIST (use --array=0-$(( $(grep -c . "$CONFIG_LIST") - 1 )))"; exit 0
fi
if [[ "${1:-}" == "--count" ]]; then
    echo "${#DATA_ARR[@]} configs in tier ${TIER} (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"
    exit 0
fi

FAIL_FILE="${SLURM_SUBMIT_DIR:-$PWD}/bins_low_fails"
finished=0

on_exit() {
    local rc=$?
    if [[ "$finished" -ne 1 ]]; then
        echo "${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} rc=${rc} $(date '+%F %T')" >> "$FAIL_FILE"
    fi
    noram_cleanup_job   # -noram folders a killed run left in temp_storage (noram_cleanup.sh)
}
trap on_exit EXIT
# time limit (SIGTERM 120 s before, --signal): log it so the next estimate for this config gets more time
trap 'kill "${srun_pid:-}" 2>/dev/null; [[ -n "${CFG_KEY:-}" ]] && record_failure bins "$dataset" "$CFG_KEY" "${OUT_DIR}/time_v.txt" 143 "$SECONDS" timeout; exit 143' TERM INT

# CONFIG_LIST (from datasets/run_all.sh): line i+1 = "dataset k bs dpb vec GB minutes ..." for array task i.
# Without it, task i is config i of this tier's grid (as before).
task_id=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${ARRAY_OFFSET:-0} ))
est_gb="?"; est_min="?"
if [[ -n "${CONFIG_LIST:-}" ]]; then
    line=$(sed -n "$(( task_id + 1 ))p" "$CONFIG_LIST")
    [[ -n "$line" ]] || { echo "task ${task_id} is past the end of $CONFIG_LIST"; finished=1; exit 0; }
    read -r dataset k_val bin_size docs_val vec_val est_gb est_min _ <<< "$line"
else
    (( task_id < ${#DATA_ARR[@]} )) || { echo "task ${task_id} is past the ${#DATA_ARR[@]} configs"; finished=1; exit 0; }
    dataset=${DATA_ARR[$task_id]}
    k_val=${K_ARR[$task_id]}
    bin_size=${BS_ARR[$task_id]}
    docs_val=${DPB_ARR[$task_id]}
    vec_val=${VEC_ARR[$task_id]}
fi
# low-RAM mode (NORAM=1 [NORAM_ABOVE_GB=n], slurm_common.sh bins_noram): PIR DB, hints and embeddings on disk
NORAM_RUN=0; bins_noram "$dataset" "$vec_val" "$docs_val" "$bin_size" && NORAM_RUN=1
CFG_KEY=$(bins_config_key "$vec_val" "$k_val" "$bin_size" "$docs_val" "$NORAM_RUN")

missing=$(bins_missing "$dataset" "$vec_val")
if [[ -n "$missing" ]]; then
    echo "SKIP bins ${dataset} ${CFG_KEY}: missing ${missing} (data/embeddings/Bluge index: datasets/run_all.sh)"
    finished=1; exit 0
fi
method="bins"

DB_SIZE=$(db_size_of "$dataset")

LAYOUT_ARGS=(-vectors="$vec_val")
TAG_SUFFIX=""
if [[ "$vec_val" == "text" ]]; then
    VEC_TAG="_text"
    LAYOUT_ARGS=(-text -text-format "$TEXT_FORMAT" -text-order "$TEXT_ORDER")
    [[ "$TEXT_FORMAT" != bow || "$TEXT_ORDER" != 3 ]] && TAG_SUFFIX="_${TEXT_FORMAT}o${TEXT_ORDER}"
elif [[ "$vec_val" == "true" ]]; then
    VEC_TAG="_vec1"
elif [[ "$vec_val" =~ ^vb([0-9]+)r([0-9]+)i([0-9]+)(m([0-9]+))?$ ]]; then   # vec0 sweep (add_vec0_sweep / add_maxterms_sweep)
    VEC_TAG="_vec0"
    LAYOUT_ARGS=(-vectors=false -vecBatch "${BASH_REMATCH[1]}" -vecRounds "${BASH_REMATCH[2]}"
                 -idBatch "${BASH_REMATCH[3]}" -vecOrder rr)
    [[ -n "${BASH_REMATCH[4]}" ]] && LAYOUT_ARGS+=(-maxTerms "${BASH_REMATCH[5]}")   # m0 = every term, as without it
    TAG_SUFFIX="$(vec0_sweep_suffix "$vec_val")"
else
    VEC_TAG="_vec0"
fi
# same results folder as an in-RAM run; metadata.json gets "noram": "true" (filter those out of compute-time plots)
(( NORAM_RUN )) && LAYOUT_ARGS+=(-noram -noram-dir "${NORAM_DIR:-temp_storage}")
LAYOUT_ARGS+=($(sim_args))   # SIMULATE=1 (slurm_common.sh): simulated PIR, results in $RESULTS_DIR (results_simulated)
OUT_DIR="${RESULTS_DIR:-/scratch/$USER/results}/${method}${VEC_TAG}_${dataset}_k${k_val}_bs${bin_size}_dpb${docs_val}${TAG_SUFFIX}"
if [[ -s "${OUT_DIR}/metadata.json" && -z "${FORCE:-}" ]]; then
    echo "Already done: ${OUT_DIR} (set FORCE=1 to rerun)"; finished=1; exit 0
fi
# OUT_DIR="../results/${method}${VEC_TAG}_${dataset}_k${k_val}_bs${bin_size}_dpb${docs_val}"
mkdir -p "$OUT_DIR"

cat <<EOF > "${OUT_DIR}/slurm_name.txt"
========================================
Array Task ID: $SLURM_ARRAY_TASK_ID
Method:        $method
Dataset:       $dataset
K Value:       $k_val
Bin Size:      $bin_size
Docs/Bin:      $docs_val
Output Dir:    $OUT_DIR
========================================
EOF

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Method:        $method"
echo "Dataset:       $dataset"
echo "K Value:       $k_val"
echo "Bin Size:      $bin_size"
echo "Docs/Bin:      $docs_val"
echo "Low-RAM:       $( (( NORAM_RUN )) && echo "yes (-noram, files in ${NORAM_DIR:-temp_storage}/)" || echo no)"
echo "Estimate:      ${est_gb} GB, ${est_min} min (requested: ${SLURM_MEM_PER_NODE:-?} MB)"
echo "Output Dir:    $OUT_DIR"
echo "========================================"

SECONDS=0
# peak RSS + wall time -> time_v.txt -> usage log (slurm_common.sh), which later estimates use
rm -f "${OUT_DIR}/peak_anon.txt"   # pir_app keeps its heap peak there while it runs (record_failure reads it)
PIR_PEAK_FILE="${OUT_DIR}/peak_anon.txt" srun bash ./measure_run.sh "${OUT_DIR}/time_v.txt" ./pir_app \
    -n "$DB_SIZE" \
    -t "$method" \
    -name "$dataset" \
    -k "$k_val" \
    -binSize "$bin_size" \
    -docsPerBin "$docs_val" \
    -outDir "$OUT_DIR" \
    -outFile "results" \
    -dataset "$DATASET_DIR" \
    -debug 1 \
    "${LAYOUT_ARGS[@]}" \
    -dim 192 &

srun_pid=$!
rc=0
wait "$srun_pid" || rc=$?
if (( rc != 0 )); then
    record_failure bins "$dataset" "$CFG_KEY" "${OUT_DIR}/time_v.txt" "$rc" "$SECONDS" || true
    exit "$rc"
fi
record_usage bins "$dataset" "$CFG_KEY" "${OUT_DIR}/time_v.txt" "$SECONDS" || true

mv "logs/Bins_${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}.out" "${OUT_DIR}/execution.log" || true
mv "logs/Bins_${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}.err" "${OUT_DIR}/error.log" || true

finished=1
echo "All done for this node."