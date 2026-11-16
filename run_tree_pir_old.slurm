#!/bin/bash -l
#SBATCH --job-name=Tree_Ablation_PIR_old
# Adapted from bm25-tree main-updated run_tree.slurm: same grid/paths, but runs bins-go `pir_app -t tree`
# (tree code on pir-old) so the output is results.json + metadata.json like bins/pacmann.
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=0-06:00:00
#SBATCH --constraint=amd
#SBATCH --array=0-960
#SBATCH --output=logs/Tree_%A_%a.out
#SBATCH --error=logs/Tree_%A_%a.err
#SBATCH --signal=B:TERM@120

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

#module --quiet purge
#module load hosts/hopper
#module load gnu9/9.3.0
# pir_app finds libngt/libhnsw through its rpath; this also covers a pir_app built without go_env.sh
export LD_LIBRARY_PATH="${NGT_PREFIX:-$HOME/opt/ngt-amd}/lib:${HNSW_PREFIX:-$HOME/opt/hnsw}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"


# The grid lives in tree_sweep_grid.sh (shared with datasets/prep_tree.sh, which builds the artifacts).
#   DATASET=msmarco|scifact|nq|dbpedia-entity|fever|hotpotqa|miracl-en|msmarco-v2[-25m|-50m|-100m]
#   SWEEP=legacy|oat|grid|all (default all)
#
# Normally datasets/run_all.sh submits this: it skips configs whose inputs are missing, groups the rest by
# estimated RAM/time and passes each group as CONFIG_LIST (one "d B r s L q c max_q ..." line per array task),
# with a matching --mem/--time/--partition. By hand, without CONFIG_LIST, task i runs config i of the grid:
#   bash run_tree_pir_old.slurm --count; DATASET=nq sbatch --array=0-960 run_tree_pir_old.slurm
#   (ARRAY_OFFSET=1000 shifts task ids for grids above MaxArraySize)
# Configs whose metadata.json exists are skipped (FORCE=1 reruns), and so are configs with a missing input.
# Every finished run appends its peak RSS and wall time to $USAGE_FILE (slurm_common.sh), which later estimates use.
source "${ROOT_DIR}/tree_sweep_grid.sh"

task_id=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${ARRAY_OFFSET:-0} ))
if [[ -n "${CONFIG_LIST:-}" ]]; then
    if [[ "${1:-}" == "--count" ]]; then
        echo "$(grep -c . "$CONFIG_LIST") configs in $CONFIG_LIST (use --array=0-$(( $(grep -c . "$CONFIG_LIST") - 1 )))"; exit 0
    fi
    line=$(sed -n "$(( task_id + 1 ))p" "$CONFIG_LIST")
    [[ -n "$line" ]] || { echo "task ${task_id} is past the end of $CONFIG_LIST, nothing to do"; exit 0; }
    read -r dataset B_val r_val s_val L_val q_words_val k_cand_val max_q_val est_gb est_min _ <<< "$line"
    SWEEP="list"
else
    DATASET="${DATASET:?set DATASET (see tree_sweep_grid.sh)}"
    SWEEP="${SWEEP:-all}"
    build_tree_grid "$DATASET" "$SWEEP" || exit 1
    if [[ "${1:-}" == "--count" ]]; then
        echo "${#DATA_ARR[@]} configs for ${DATASET} SWEEP=${SWEEP} (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"
        exit 0
    fi
    if (( task_id >= ${#DATA_ARR[@]} )); then
        echo "task ${task_id} is past the ${#DATA_ARR[@]} configs of ${DATASET} SWEEP=${SWEEP}, nothing to do"; exit 0
    fi
    dataset=${DATA_ARR[$task_id]}; L_val=${L_ARR[$task_id]}; k_cand_val=${KCAND_ARR[$task_id]}
    q_words_val=${QWORDS_ARR[$task_id]}; B_val=${B_ARR[$task_id]}; r_val=${R_ARR[$task_id]}
    s_val=${S_ARR[$task_id]}; max_q_val=${MAX_Q_ARR[$task_id]}; est_gb="?"; est_min="?"
fi
embed_val=192

BINARY="${ROOT_DIR}/pir_app"
# Non-default Cuckoo settings (TREE_CUCKOO_HASHES=3 TREE_CUCKOO_LF=0.88 roughly halves the Stage 1/2 tables, worth it
# for msmarco-v2) go in their own sub-folder so they never mix with the default runs.
read -r CUCKOO_HASHES CUCKOO_LF <<< "$(tree_cuckoo "$dataset")"   # msmarco-v2 defaults to 3 / 0.88 (slurm_common.sh)
CUCKOO_KICKS=$([[ "$CUCKOO_HASHES" == 3 ]] && echo 5000 || echo 1000)
CUCKOO_TAG=""
[[ "$CUCKOO_HASHES" != 2 || "$CUCKOO_LF" != 0.4 ]] && CUCKOO_TAG="h${CUCKOO_HASHES}_lf${CUCKOO_LF}/"
# _c<k_candidates> is part of the name now that k_candidates is swept (load_results.py reads it)
OUT_DIR="${RESULTS_DIR:-/scratch/$USER/results}/${CUCKOO_TAG}tree_grid_${dataset}_b${B_val}_r${r_val}_s${s_val}_L${L_val}_q${q_words_val}_c${k_cand_val}_k10"
if [[ -s "${OUT_DIR}/metadata.json" && -z "${FORCE:-}" ]]; then
    echo "Already done: ${OUT_DIR} (set FORCE=1 to rerun)"; exit 0
fi

missing=$(tree_missing "$dataset" "$B_val" "$r_val" "$s_val")
if [[ -n "$missing" ]]; then
    echo "SKIP ${dataset} B=${B_val} r=${r_val} s=${s_val} L=${L_val} q=${q_words_val} c=${k_cand_val}: missing ${missing}"
    echo "     (tree artifacts: DATASET=${dataset} sbatch datasets/tree_artifacts.slurm; data/embeddings: datasets/run_all.sh)"
    exit 0
fi
tree_paths "$dataset" "$B_val" "$r_val" "$s_val"
# One Cuckoo cache per dataset + shape (B r s), shared by every config of that shape (L, q, c don't change the
# tables): the first run builds and saves them, the others mmap them instead of rebuilding them on the heap, so they
# need far less RAM (tree_cache_ready / tree_plan in tree_sweep_grid.sh). Safe to share: a cache file is written
# to a temp file and renamed, the build is deterministic, and a stale cache (artifacts rebuilt) is detected and
# rebuilt. It used to be one folder per task, never reused (3.8 TB by 2026-11-10).
CACHE_DIR=$(tree_cache_dir "$dataset" "$B_val" "$r_val" "$s_val")
mkdir -p "$CACHE_DIR"
CACHE_ARGS="--cuckoo-cache-dir ${CACHE_DIR}"
CACHE_HIT=0; tree_cache_ready "$dataset" "$B_val" "$r_val" "$s_val" && CACHE_HIT=1
tree_paths "$dataset" "$B_val" "$r_val" "$s_val"   # tree_cache_* ran it in subshells; the run below needs its paths
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-} (config ${task_id}, ${SWEEP})"
echo "Dataset:       $dataset"
echo "B Value:       $B_val"
echo "r Value:       $r_val"
echo "s Value:       $s_val"
echo "L Value:       $L_val"
echo "k_candidates:  $k_cand_val"
echo "Query Words:   $q_words_val"
echo "Estimate:      ${est_gb} GB, ${est_min} min (requested: ${SLURM_MEM_PER_NODE:-?} MB)"
echo "Output Dir:    $OUT_DIR"
echo "Cuckoo cache:  $CACHE_DIR ($( (( CACHE_HIT )) && echo "ready: tables are mmap'd" || echo "not ready: this run builds the tables"))"
echo "========================================"

LOG_FILE="${OUT_DIR}/run_go_${dataset}_b${B_val}_r${r_val}_s${s_val}_L${L_val}_q${q_words_val}_c${k_cand_val}_k10.log"

SECONDS=0
echo "Running tree search..."
# peak RSS + wall time go to time_v.txt and then the usage log (slurm_common.sh record_usage)
rm -f "${OUT_DIR}/peak_anon.txt"
PIR_PEAK_FILE="${OUT_DIR}/peak_anon.txt" bash "${ROOT_DIR}/measure_run.sh" "${OUT_DIR}/time_v.txt" "$BINARY" \
    -t tree \
    -name "$dataset" \
    -n "$DB_SIZE" \
    -dim "$embed_val" \
    -dataset "${DATASET_DIR:-../../datasets}" \
    -outDir "$OUT_DIR" \
    -outFile results \
    --L "$L_val" \
    --k_candidates "$k_cand_val" \
    --query-words "$q_words_val" \
    --B "$B_val" \
    --r "$r_val" \
    --s "$s_val" \
    --k 10 \
    --max-queries "$max_q_val" \
    --k1 0.9 \
    --b 0.4 \
    --scale 10 \
    --precompute \
    --pir-batch-size 128 \
    --cuckoo-load-factor "$CUCKOO_LF" \
    --cuckoo-hashes "$CUCKOO_HASHES" \
    --cuckoo-max-kicks "$CUCKOO_KICKS" \
    --cuckoo-build-attempts 8 \
    --layout-path "$LAYOUT_PATH" \
    --index "$INDEX" \
    --queries "$QUERIES" \
    --qrels "$QRELS" \
    --vocab "$VOCAB" \
    --stage1-data-bin "$STAGE1_DATA" \
    --stage1-idmap-bin "$STAGE1_IDMAP" \
    --stage2-data-bin "$STAGE2_DATA" \
    --stage2-idmap-bin "$STAGE2_IDMAP" \
    --doc-embed "$DOC_EMBED" \
    --doc-id-map "$DOC_ID_MAP" \
    --query-embed "$QUERY_EMBED" \
    --embed-dim "$embed_val" \
    --block-permutation "$BLOCK_PERMS" \
    --pir-threads 16 \
    $(sim_args) \
    $CACHE_ARGS > "$LOG_FILE" 2>&1 &
run_pid=$!
CFG_KEY=$(tree_config_key "$B_val" "$r_val" "$s_val" "$L_val" "$q_words_val" "$k_cand_val" "$dataset")$( (( CACHE_HIT )) && echo _cached || echo _build)
# time limit (SIGTERM 120 s before it, see --signal) or OOM: log it so the next estimate for this config is bigger
trap 'kill -TERM "$run_pid" 2>/dev/null; record_failure tree "$dataset" "$CFG_KEY" "${OUT_DIR}/time_v.txt" 143 "$SECONDS" timeout; exit 143' TERM
rc=0
wait "$run_pid" || rc=$?
if (( rc != 0 )); then
    echo "pir_app failed with exit code $rc, see $LOG_FILE"
    record_failure tree "$dataset" "$CFG_KEY" "${OUT_DIR}/time_v.txt" "$rc" "$SECONDS" || true
    exit "$rc"
fi

elapsed_time=$SECONDS
echo "Time elapsed: $elapsed_time seconds"
# every table of this shape is now in the cache: later configs of the shape are sized as cache hits
tree_mark_cache_ready "$dataset" "$B_val" "$r_val" "$s_val" || true
record_usage tree "$dataset" "$CFG_KEY" "${OUT_DIR}/time_v.txt" "$elapsed_time" || true
echo "Execution finished."

# Copy SLURM logs into the output directory for reference
cp "${ROOT_DIR}/logs/Tree_${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}.out" "${OUT_DIR}/execution.log" || true
cp "${ROOT_DIR}/logs/Tree_${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}.err" "${OUT_DIR}/error.log" || true