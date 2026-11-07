#!/bin/bash -l
#SBATCH --job-name=Tree_Ablation_PIR_old
# Adapted from bm25-tree main-updated run_tree.slurm: same grid/paths, but runs bins-go `pir_app -t tree`
# (tree code on pir-old) so the output is results.json + metadata.json like bins/pacmann.
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=0-13:30:00
#SBATCH --constraint=amd
#SBATCH --array=0-959
#SBATCH --output=logs/Tree_%A_%a.out
#SBATCH --error=logs/Tree_%A_%a.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

#module --quiet purge
#module load hosts/hopper
#module load gnu9/9.3.0
export PATH=$PATH:/usr/local/go/bin


DATA_ARR=(); L_ARR=(); KCAND_ARR=(); QWORDS_ARR=(); B_ARR=(); R_ARR=(); S_ARR=(); MAX_Q_ARR=()

# Helper function to generate the grid for one dataset
add_tree_ablations() {
    local d=$1 k_cand=$2 max_q=$3 B r s q W
    for B in $4; do
        for r in $5; do
            for s in $6; do
                for q in $7; do
                    for W in $8; do
                        DATA_ARR+=("$d"); B_ARR+=("$B"); R_ARR+=("$r"); S_ARR+=("$s")
                        QWORDS_ARR+=("$q"); L_ARR+=("$W"); KCAND_ARR+=("$k_cand"); MAX_Q_ARR+=("$max_q")
                    done
                done
            done
        done
    done
}

DATASET="${DATASET:?set DATASET to msmarco, scifact, nq, hotpotqa, fever or msmarco-v2[-25m|-50m|-100m]}"
#                                    k_cand max_q  B              r               s        q*          W
if [[ "$DATASET" == "msmarco" ]]; then
    add_tree_ablations "msmarco" 128 6980 "16 32 64"   "8 32 128"   "4 8 16" "2 4 6 8"   "50 100 200 400"
elif [[ "$DATASET" == "scifact" ]]; then
    add_tree_ablations "scifact" 20  300  "8 16 32 64" "4 8 32 128" "2 4 8"  "4 8 12 16" "5 10 20 40 80"
elif [[ "$DATASET" =~ ^(nq|hotpotqa|fever|msmarco-v2.*)$ ]]; then
    # 2.7M-138M docs, same grid as msmarco. Build every (B, r, s) first in one go:
    #   DATASET=$DATASET sbatch datasets/tree_artifacts.slurm     (or let datasets/run_all.sh chain it)
    add_tree_ablations "$DATASET" 128 10000 "16 32 64"   "8 32 128"   "4 8 16" "2 4 6 8"   "50 100 200 400"
else
    echo "Unknown DATASET: $DATASET"; exit 1
fi

if [[ "${1:-}" == "--count" ]]; then
    echo "${#DATA_ARR[@]} configs for ${DATASET} (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"
    exit 0
fi

task_id=$SLURM_ARRAY_TASK_ID
dataset=${DATA_ARR[$task_id]}
L_val=${L_ARR[$task_id]}
k_cand_val=${KCAND_ARR[$task_id]}
q_words_val=${QWORDS_ARR[$task_id]}
B_val=${B_ARR[$task_id]}
r_val=${R_ARR[$task_id]}
s_val=${S_ARR[$task_id]}
max_q_val=${MAX_Q_ARR[$task_id]}
embed_val=192

BINARY="${ROOT_DIR}/pir_app"
OUT_DIR="/scratch/dblackle/results/tree_grid_${dataset}_b${B_val}_r${r_val}_s${s_val}_L${L_val}_q${q_words_val}_k10"
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset:       $dataset"
echo "B Value:       $B_val"
echo "r Value:       $r_val"
echo "s Value:       $s_val"
echo "L Value:       $L_val"
echo "k_candidates:  $k_cand_val"
echo "Query Words:   $q_words_val"
echo "Output Dir:    $OUT_DIR"
echo "========================================"

# Pre-define dataset-specific variables using absolute paths
if [ "$dataset" == "msmarco" ]; then
    DB_SIZE=8841823
    DATA_ROOT="${ROOT_DIR}/data/msmarco"
    ARTIFACT_DIR="${DATA_ROOT}/pir/B${B_val}_r${r_val}_s${s_val}"
    LAYOUT_PATH="${ARTIFACT_DIR}/layout.json"
    BLOCK_PERMS="data/block_permutations.json"
    INDEX="${ROOT_DIR}/data/msmarco_reordered"
    QUERIES="${ROOT_DIR}/data/queries.dev.small.tsv"
    # QUERIES="/scratch/dblackle/tree_stuff/new_tests/msmarco/queries.tsv"
    QRELS="${ROOT_DIR}/data/qrels.dev.small.tsv"
    # QRELS="/scratch/dblackle/tree_stuff/new_tests/msmarco/qrels.tsv"
    VOCAB="${ARTIFACT_DIR}/stage1_vocab.json"
    STAGE1_DATA="${ARTIFACT_DIR}/stage1_data.bin"
    STAGE1_IDMAP="${ARTIFACT_DIR}/stage1_idmap.bin"
    STAGE2_DATA="${ARTIFACT_DIR}/stage2/data.bin"
    STAGE2_IDMAP="${ARTIFACT_DIR}/stage2/idmap.bin"
    DOC_EMBED="${ROOT_DIR}/data/stage3_doc_aligned_shuffle.npy"
    DOC_ID_MAP="${ROOT_DIR}/data/stage3_doc_aligned.npy.ids"
    QUERY_EMBED="${ROOT_DIR}/data/stage3_query.npy"
    # QUERY_EMBED="/scratch/dblackle/tree_stuff/new_tests/msmarco/data/query_embeddings.npy"
    CACHE_ARGS=""

elif [ "$dataset" == "scifact" ]; then
    DB_SIZE=5183
    DATA_ROOT="${ROOT_DIR}/data/scifact"
    ARTIFACT_DIR="${DATA_ROOT}/pir/B${B_val}_r${r_val}_s${s_val}"
    BLOCK_PERMS="${DATA_ROOT}/block_permutations.json"

    CACHE_DIR="${ROOT_DIR}/cuckoo_cache/scifact/B${B_val}_r${r_val}_s${s_val}/task_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "$CACHE_DIR"

    LAYOUT_PATH="${ARTIFACT_DIR}/layout.json"
    INDEX="${ROOT_DIR}/indexes/scifact"
    QUERIES="${DATA_ROOT}/queries.qrels.tsv"
    # QUERIES="/scratch/dblackle/tree_stuff/new_tests/scifact/queries.qrels.tsv"
    QRELS="${DATA_ROOT}/qrels.trec"
    # QRELS="/scratch/dblackle/tree_stuff/new_tests/scifact/qrels.tsv"
    VOCAB="${ARTIFACT_DIR}/stage1_vocab.json"
    STAGE1_DATA="${ARTIFACT_DIR}/stage1_data.bin"
    STAGE1_IDMAP="${ARTIFACT_DIR}/stage1_idmap.bin"
    STAGE2_DATA="${ARTIFACT_DIR}/stage2/data.bin"
    STAGE2_IDMAP="${ARTIFACT_DIR}/stage2/idmap.bin"
    DOC_EMBED="${DATA_ROOT}/clustered/stage3_e5_pca192_doc_embeddings.npy"
    DOC_ID_MAP="${DATA_ROOT}/clustered/stage3_e5_pca192_doc_embeddings.npy.ids"
    QUERY_EMBED="${DATA_ROOT}/stage3_e5_pca192_query_embeddings.npy"
    # QUERY_EMBED="/scratch/dblackle/tree_stuff/new_tests/scifact/query_embeddings.npy"


    CACHE_ARGS="--cuckoo-cache-dir ${CACHE_DIR}"

elif [[ "$dataset" =~ ^(nq|hotpotqa|fever|msmarco-v2.*)$ ]]; then
    # Layout written by datasets/prep_tree.sh
    DATASET_ROOT="${DATASET_DIR:-../../datasets}"
    DB_SIZE=$(cat "${DATASET_ROOT}/${dataset}/num_docs.txt")
    DATA_ROOT="${ROOT_DIR}/data/${dataset}_tree"
    ARTIFACT_DIR="${DATA_ROOT}/B${B_val}_r${r_val}_s${s_val}"
    BLOCK_PERMS=""

    CACHE_DIR="${ROOT_DIR}/cuckoo_cache/${dataset}/B${B_val}_r${r_val}_s${s_val}/task_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "$CACHE_DIR"

    LAYOUT_PATH="${ARTIFACT_DIR}/layout.json"
    INDEX="${DATA_ROOT}/index_${dataset}"
    QUERIES="${DATA_ROOT}/queries.qrels.tsv"
    QRELS="${DATA_ROOT}/qrels.trec"
    VOCAB="${ARTIFACT_DIR}/stage1_vocab.json"
    STAGE1_DATA="${ARTIFACT_DIR}/stage1_data.bin"
    STAGE1_IDMAP="${ARTIFACT_DIR}/stage1_idmap.bin"
    STAGE2_DATA="${ARTIFACT_DIR}/stage2/data.bin"
    STAGE2_IDMAP="${ARTIFACT_DIR}/stage2/idmap.bin"
    DOC_EMBED="${DATA_ROOT}/stage3_doc.npy"
    DOC_ID_MAP="${DATA_ROOT}/stage3_doc.npy.ids"
    # rows follow queries.jsonl order; tree_main matches them to the analyzed queries by ID
    QUERY_EMBED="${DATASET_ROOT}/${dataset}/${dataset}_query_embed_f32.npy"

    CACHE_ARGS="--cuckoo-cache-dir ${CACHE_DIR}"

else
    echo "Unknown dataset: $dataset"
    exit 1
fi

LOG_FILE="${OUT_DIR}/run_go_${dataset}_b${B_val}_r${r_val}_s${s_val}_L${L_val}_q${q_words_val}_k10.log"

SECONDS=0
echo "Running tree search..."
"$BINARY" \
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
    --cuckoo-load-factor 0.4 \
    --cuckoo-max-kicks 1000 \
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
    $CACHE_ARGS > "$LOG_FILE" 2>&1

elapsed_time=$SECONDS
echo "Time elapsed: $elapsed_time seconds"
echo "Execution finished."

# Copy SLURM logs into the output directory for reference
cp "${ROOT_DIR}/logs/Tree_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/execution.log" || true
cp "${ROOT_DIR}/logs/Tree_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/error.log" || true