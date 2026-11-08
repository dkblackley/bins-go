#!/bin/bash -l
#SBATCH --job-name=Tree_Grid
# Grid search over the PILLAR-Tree-only client-storage knobs. Every knob only changes
# how the tree lays out its own PIR tables; the documents, embeddings and the BM25
# bounds themselves are untouched, and none of it applies to PACMANN.
#
#   pir_batch     --pir-batch-size        Stage1/2 partitions = batch/2 (128 = today)
#   s3_batch      --stage3-batch-size     Stage3 partitions = batch/2   (32 = today)
#   cuckoo        --cuckoo-hashes / --cuckoo-load-factor
#                 2 hashes must stay below load 0.5; 3 hashes fill up to ~0.9 (fewer rows,
#                 3 probes per key instead of 2)                          (2 @ 0.4 = today)
#   key_bytes     --cuckoo-key-bytes      key fingerprint kept per row     (8 = today)
#                 a false match only loosens that node's bound (never lowers it)
#   ub_bits       --ub-bits               bits per upper bound, rounded up (8 = today)
#
# 4 values each, every combination: 4^5 = 1024 runs per dataset. The all-8/128/32/0.4
# combination is today's run exactly.
#
#   DATASET=scifact bash run_tree_grid_sweep.slurm --count
#   DATASET=scifact sbatch --array=0-1023%64 run_tree_grid_sweep.slurm
#   DATASET=msmarco sbatch --array=0-1023%32 run_tree_grid_sweep.slurm
#
# Results: ${RESULTS_ROOT}/<tag>/tree_grid_..., tag = pb<batch>_s3b<batch>_h<hashes>lf<load>_kb<bytes>_ub<bits>.
# metadata.json records cuckoo_hash_functions, cuckoo_key_bytes, ub_bits,
# stage1_entry_bytes and Stage3BatchSize, and hintbench.md the predicted sizes.
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=0-13:30:00
#SBATCH --constraint=amd
#SBATCH --output=logs/TreeGrid_%A_%a.out
#SBATCH --error=logs/TreeGrid_%A_%a.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"
export PATH=$PATH:/usr/local/go/bin

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/dblackle/results_tree_grid}"
DATASET="${DATASET:?set DATASET to msmarco or scifact}"

PIR_BATCHES=(128 64 32 16)
S3_BATCHES=(32 16 8 4)
CUCKOOS=("2|0.4" "2|0.48" "3|0.8" "3|0.88")   # hashes|load factor
KEY_BYTES=(8 4 2 1)
UB_BITS=(8 6 4 2)

COMBOS=()
for pb in "${PIR_BATCHES[@]}"; do
  for s3 in "${S3_BATCHES[@]}"; do
    for ck in "${CUCKOOS[@]}"; do
      for kb in "${KEY_BYTES[@]}"; do
        for ub in "${UB_BITS[@]}"; do
          COMBOS+=("${pb}|${s3}|${ck}|${kb}|${ub}")
        done
      done
    done
  done
done

if [[ "${1:-}" == "--count" ]]; then
    echo "${#COMBOS[@]} configs (use --array=0-$(( ${#COMBOS[@]} - 1 )))"
    exit 0
fi

IFS='|' read -r PB S3B HASHES LOAD KB UB <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
TAG="pb${PB}_s3b${S3B}_h${HASHES}lf${LOAD}_kb${KB}_ub${UB}"
# 3 hashes near 0.9 needs longer random walks to place the last keys
MAX_KICKS=1000
if [[ "$HASHES" == "3" ]]; then MAX_KICKS=5000; fi

# The tree config the grid is run on (edit to match a frontier config).
if [[ "$DATASET" == "msmarco" ]]; then
    B_val=32; r_val=32; s_val=8; L_val=100; q_words_val=8; k_cand_val=128; max_q_val=6980
else
    B_val=16; r_val=8;  s_val=4; L_val=20;  q_words_val=8; k_cand_val=20;  max_q_val=300
fi

embed_val=192
BINARY="${ROOT_DIR}/pir_app"
OUT_DIR="${RESULTS_ROOT}/${TAG}/tree_grid_${DATASET}_b${B_val}_r${r_val}_s${s_val}_L${L_val}_q${q_words_val}_k10"
mkdir -p "$OUT_DIR"

if [ "$DATASET" == "msmarco" ]; then
    DB_SIZE=8841823
    DATA_ROOT="${ROOT_DIR}/data/msmarco"
    ARTIFACT_DIR="${DATA_ROOT}/pir/B${B_val}_r${r_val}_s${s_val}"
    BLOCK_PERMS="data/block_permutations.json"
    INDEX="${ROOT_DIR}/data/msmarco_reordered"
    QUERIES="${ROOT_DIR}/data/queries.dev.small.tsv"
    QRELS="${ROOT_DIR}/data/qrels.dev.small.tsv"
    DOC_EMBED="${ROOT_DIR}/data/stage3_doc_aligned_shuffle.npy"
    DOC_ID_MAP="${ROOT_DIR}/data/stage3_doc_aligned.npy.ids"
    QUERY_EMBED="${ROOT_DIR}/data/stage3_query.npy"
    CACHE_ARGS=""   # default cache dir; layouts get their own cache file names
else
    DB_SIZE=5183
    DATA_ROOT="${ROOT_DIR}/data/scifact"
    ARTIFACT_DIR="${DATA_ROOT}/pir/B${B_val}_r${r_val}_s${s_val}"
    BLOCK_PERMS="${DATA_ROOT}/block_permutations.json"
    CACHE_DIR="${ROOT_DIR}/cuckoo_cache/scifact/B${B_val}_r${r_val}_s${s_val}/grid"
    mkdir -p "$CACHE_DIR"
    INDEX="${ROOT_DIR}/indexes/scifact"
    QUERIES="${DATA_ROOT}/queries.qrels.tsv"
    QRELS="${DATA_ROOT}/qrels.trec"
    DOC_EMBED="${DATA_ROOT}/clustered/stage3_e5_pca192_doc_embeddings.npy"
    DOC_ID_MAP="${DATA_ROOT}/clustered/stage3_e5_pca192_doc_embeddings.npy.ids"
    QUERY_EMBED="${DATA_ROOT}/stage3_e5_pca192_query_embeddings.npy"
    CACHE_ARGS="--cuckoo-cache-dir ${CACHE_DIR}"
fi
LAYOUT_PATH="${ARTIFACT_DIR}/layout.json"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${TAG} -> ${OUT_DIR}" | tee "${OUT_DIR}/slurm_name.txt"

# hintbench's prediction for these exact artifacts, next to the measured run
go run ./pirhint/cmd/hintbench -method tree -dataset "$DATASET" -docs "$DB_SIZE" \
    -B "$B_val" -r "$r_val" -s "$s_val" -batch "$PB" -stage3-batch "$S3B" \
    -load "$LOAD" -hashes "$HASHES" -key-bytes "$KB" -ub-bits "$UB" \
    -stage1-idmap "${ARTIFACT_DIR}/stage1_idmap.bin" -stage2-idmap "${ARTIFACT_DIR}/stage2/idmap.bin" \
    -json "${OUT_DIR}/hintbench.json" > "${OUT_DIR}/hintbench.md" 2>&1 || true

SECONDS=0
# shellcheck disable=SC2086  # CACHE_ARGS is a word list
"$BINARY" \
    -t tree -name "$DATASET" -n "$DB_SIZE" -dim "$embed_val" \
    -dataset "${DATASET_DIR:-../../datasets}" -outDir "$OUT_DIR" -outFile results \
    --L "$L_val" --k_candidates "$k_cand_val" --query-words "$q_words_val" \
    --B "$B_val" --r "$r_val" --s "$s_val" --k 10 --max-queries "$max_q_val" \
    --k1 0.9 --b 0.4 --scale 10 --precompute \
    --pir-batch-size "$PB" --stage3-batch-size "$S3B" \
    --cuckoo-hashes "$HASHES" --cuckoo-load-factor "$LOAD" \
    --cuckoo-key-bytes "$KB" --ub-bits "$UB" \
    --cuckoo-max-kicks "$MAX_KICKS" --cuckoo-build-attempts 8 \
    --layout-path "$LAYOUT_PATH" --index "$INDEX" --queries "$QUERIES" --qrels "$QRELS" \
    --vocab "${ARTIFACT_DIR}/stage1_vocab.json" \
    --stage1-data-bin "${ARTIFACT_DIR}/stage1_data.bin" --stage1-idmap-bin "${ARTIFACT_DIR}/stage1_idmap.bin" \
    --stage2-data-bin "${ARTIFACT_DIR}/stage2/data.bin" --stage2-idmap-bin "${ARTIFACT_DIR}/stage2/idmap.bin" \
    --doc-embed "$DOC_EMBED" --doc-id-map "$DOC_ID_MAP" --query-embed "$QUERY_EMBED" \
    --embed-dim "$embed_val" --block-permutation "$BLOCK_PERMS" --pir-threads 16 \
    $CACHE_ARGS > "${OUT_DIR}/run.log" 2>&1

echo "Time elapsed: $SECONDS seconds"
cp "${ROOT_DIR}/logs/TreeGrid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
