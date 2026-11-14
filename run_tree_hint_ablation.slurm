#!/bin/bash -l
#SBATCH --job-name=Tree_Hint_Ablation
# Client-storage ablation for PILLAR-Tree: one fixed tree config per dataset, re-run
# under each client-storage knob (see pirhint/README.md for what each one does and the
# predicted sizes). Same paths and arguments as run_tree_pir_old.slurm otherwise.
#
#   DATASET=msmarco sbatch --array=0-13 run_tree_hint_ablation.slurm
#   DATASET=scifact sbatch --array=0-13 run_tree_hint_ablation.slurm
#   DATASET=msmarco bash run_tree_hint_ablation.slurm --count
#
# Each knob writes to ${RESULTS_ROOT}/<knob>/tree_grid_..., so a knob's runs load on
# their own with RESULTS_DIR=${RESULTS_ROOT}/<knob> python plotting/main.py.
#
# The int8/fp16 knobs need the quantized Stage3 embeddings first:
#   python tree/tools/quantize_embeddings.py "$DOC_EMBED" --dtype int8
#   python tree/tools/quantize_embeddings.py "$DOC_EMBED" --dtype float16
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=0-13:30:00
#SBATCH --constraint=amd
#SBATCH --array=0-13
#SBATCH --output=logs/TreeHint_%A_%a.out
#SBATCH --error=logs/TreeHint_%A_%a.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"
export PATH=$PATH:/usr/local/go/bin

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/$USER/results_hint}"
DATASET="${DATASET:?set DATASET to msmarco or scifact}"

# The tree config every knob is applied to (edit to match a frontier config).
if [[ "$DATASET" == "msmarco" ]]; then
    B_val=32; r_val=32; s_val=8; L_val=100; q_words_val=8; k_cand_val=128; max_q_val=6980
else
    B_val=16; r_val=8;  s_val=4; L_val=20;  q_words_val=8; k_cand_val=20;  max_q_val=300
fi

# knob name | extra pir_app flags | Stage3 embedding precision (fp32, fp16, int8)
KNOBS=(
    "base||fp32"
    "download|--pir-download-small|fp32"
    "s3_fp16||fp16"
    "s3_int8||int8"
    "s3_batch16|--stage3-batch-size 16|fp32"
    "s3_batch8|--stage3-batch-size 8|fp32"
    "batch64|--pir-batch-size 64|fp32"
    "batch32|--pir-batch-size 32|fp32"
    "chunk1|--pir-chunk-factor 1|fp32"
    "chunk05|--pir-chunk-factor 0.5|fp32"
    "budget05|--pir-query-budget 0.5|fp32"
    "budget025|--pir-query-budget 0.25|fp32"
    "combined|--pir-download-small --pir-chunk-factor 1 --pir-query-budget 0.5|int8"
    "combined_small|--pir-download-small --pir-chunk-factor 1 --pir-query-budget 0.5 --pir-batch-size 32 --stage3-batch-size 16|int8"
)

if [[ "${1:-}" == "--count" ]]; then
    echo "${#KNOBS[@]} knobs (use --array=0-$(( ${#KNOBS[@]} - 1 )))"
    exit 0
fi

IFS='|' read -r KNOB KNOB_ARGS PRECISION <<< "${KNOBS[$SLURM_ARRAY_TASK_ID]}"
embed_val=192
BINARY="${ROOT_DIR}/pir_app"
OUT_DIR="${RESULTS_ROOT}/${KNOB}/tree_grid_${DATASET}_b${B_val}_r${r_val}_s${s_val}_L${L_val}_q${q_words_val}_k10"
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
    CACHE_ARGS=""
else
    DB_SIZE=5183
    DATA_ROOT="${ROOT_DIR}/data/scifact"
    ARTIFACT_DIR="${DATA_ROOT}/pir/B${B_val}_r${r_val}_s${s_val}"
    BLOCK_PERMS="${DATA_ROOT}/block_permutations.json"
    CACHE_DIR="${ROOT_DIR}/cuckoo_cache/scifact/B${B_val}_r${r_val}_s${s_val}/hint_${SLURM_ARRAY_TASK_ID}"
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

if [[ "$PRECISION" == "int8" ]]; then
    DOC_EMBED="${DOC_EMBED%.npy}_int8.npy"
elif [[ "$PRECISION" == "fp16" ]]; then
    DOC_EMBED="${DOC_EMBED%.npy}_float16.npy"
fi
if [[ ! -f "$DOC_EMBED" ]]; then
    echo "Missing $DOC_EMBED: run tree/tools/quantize_embeddings.py first (see the top of this file)"
    exit 1
fi

echo "Knob ${KNOB}: ${KNOB_ARGS:-<none>}, Stage3 ${PRECISION} -> ${OUT_DIR}"

# hintbench's prediction for these exact artifacts, next to the measured run
go run ./pirhint/cmd/hintbench -method tree -dataset "$DATASET" -docs "$DB_SIZE" \
    -B "$B_val" -r "$r_val" -s "$s_val" \
    -stage1-idmap "${ARTIFACT_DIR}/stage1_idmap.bin" -stage2-idmap "${ARTIFACT_DIR}/stage2/idmap.bin" \
    -json "${OUT_DIR}/hintbench.json" > "${OUT_DIR}/hintbench.md" 2>&1 || true

SECONDS=0
# shellcheck disable=SC2086  # KNOB_ARGS and CACHE_ARGS are word lists
"$BINARY" \
    -t tree -name "$DATASET" -n "$DB_SIZE" -dim "$embed_val" \
    -dataset "${DATASET_DIR:-../../datasets}" -outDir "$OUT_DIR" -outFile results \
    --L "$L_val" --k_candidates "$k_cand_val" --query-words "$q_words_val" \
    --B "$B_val" --r "$r_val" --s "$s_val" --k 10 --max-queries "$max_q_val" \
    --k1 0.9 --b 0.4 --scale 10 --precompute \
    --pir-batch-size 128 \
    --cuckoo-load-factor 0.4 --cuckoo-max-kicks 1000 --cuckoo-build-attempts 8 \
    --layout-path "$LAYOUT_PATH" --index "$INDEX" --queries "$QUERIES" --qrels "$QRELS" \
    --vocab "${ARTIFACT_DIR}/stage1_vocab.json" \
    --stage1-data-bin "${ARTIFACT_DIR}/stage1_data.bin" --stage1-idmap-bin "${ARTIFACT_DIR}/stage1_idmap.bin" \
    --stage2-data-bin "${ARTIFACT_DIR}/stage2/data.bin" --stage2-idmap-bin "${ARTIFACT_DIR}/stage2/idmap.bin" \
    --doc-embed "$DOC_EMBED" --doc-id-map "$DOC_ID_MAP" --query-embed "$QUERY_EMBED" \
    --embed-dim "$embed_val" --block-permutation "$BLOCK_PERMS" --pir-threads 16 \
    $KNOB_ARGS $CACHE_ARGS > "${OUT_DIR}/run.log" 2>&1

echo "Time elapsed: $SECONDS seconds"
cp "${ROOT_DIR}/logs/TreeHint_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
