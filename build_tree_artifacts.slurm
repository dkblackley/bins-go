#!/bin/bash -l
#SBATCH --job-name=Tree_Artifacts
# Builds the B*_r*_s* tree artifacts that run_tree_pir_old.slurm needs for a sweep.
# One array task per (B, r) pair; each task writes stage1 once and stage2 for every s
# of that pair, and skips any artifact dir that is already complete (FORCE=1 rebuilds).
#
#   DATASET=msmarco SWEEP=all bash build_tree_artifacts.slurm --count   # how many tasks
#   DATASET=msmarco SWEEP=all sbatch --array=0-<N-1> build_tree_artifacts.slurm
#
# Needs the pyserini env (same as the old prepare_data*.py scripts).
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/TreeArt_%A_%a.out
#SBATCH --error=logs/TreeArt_%A_%a.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

source "${ROOT_DIR}/tree_sweep_grid.sh"
DATASET="${DATASET:?set DATASET to msmarco or scifact}"
SWEEP="${SWEEP:-all}"
build_tree_grid "$DATASET" "$SWEEP" || exit 1

# (B, r) pairs and the s values each one needs, e.g. "32 32 2,4,8"
mapfile -t PAIRS < <(tree_grid_shapes | awk '{k=$1" "$2; if (k in s) s[k]=s[k]","$3; else s[k]=$3} END {for (k in s) print k, s[k]}' | sort -n -k1,1 -k2,2)

if [[ "${1:-}" == "--count" ]]; then
    printf '%s\n' "${PAIRS[@]}"
    echo "${#PAIRS[@]} (B, r) tasks for ${DATASET} SWEEP=${SWEEP} (use --array=0-$(( ${#PAIRS[@]} - 1 )))"
    exit 0
fi

if [[ "$DATASET" == "msmarco" ]]; then
    INDEX="${ROOT_DIR}/data/msmarco_reordered"
    OUT_ROOT="${ROOT_DIR}/data/msmarco/pir"
else
    INDEX="${ROOT_DIR}/indexes/scifact"
    OUT_ROOT="${ROOT_DIR}/data/scifact/pir"
fi

# Reuse the term IDs of an artifact that already exists so every config shares one vocab.
VOCAB="${VOCAB:-$(ls "${OUT_ROOT}"/B*_r*_s*/stage1_vocab.json 2>/dev/null | head -n1 || true)}"
VOCAB_ARGS=()
[[ -n "$VOCAB" ]] && VOCAB_ARGS=(--vocab "$VOCAB")

read -r B r S_LIST <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"
echo "Dataset=${DATASET} B=${B} r=${r} s=${S_LIST} vocab=${VOCAB:-<from index>}"

python3 tree/build_tree_artifacts.py --index "$INDEX" --out-root "$OUT_ROOT" \
    --B "$B" --r "$r" --s "$S_LIST" ${VOCAB_ARGS[@]+"${VOCAB_ARGS[@]}"} ${FORCE:+--force}
