#!/bin/bash
#SBATCH --job-name=import_emb
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/import_%j.out
#SBATCH --error=logs/import_%j.err
#
# Server side of "embed locally, use on the server" (datasets/README.md "Embedding locally"): for every dataset whose
# float32 embeddings don't exist yet, widen the float16 files you copied to ../datasets/<name>/local_import/ into
# <name>_corpus_embed_f32.npy (+ queries + PCA), after checking the ids match the server's corpus.jsonl.
#   bash datasets/import_local_embeds.sh nq hotpotqa fever msmarco-v2      # or: sbatch datasets/import_local_embeds.sh ...
# Then queue what comes after the embeddings (embed jobs for datasets that are now done are skipped):
#   STAGES="embed tree-db" bash datasets/build_all.sh nq hotpotqa fever msmarco-v2-all
# msmarco-v2 (106 GB of float32) takes ~10-30 min depending on the disk. Datasets already done are left alone.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ROOT="${DATASETS_ROOT:-../datasets}"
[[ $# -gt 0 ]] || { sed -n 10,17p "$0"; exit 1; }
python=python3; command -v python3 >/dev/null || python=python
for d in "$@"; do
    echo "== $d"
    $python datasets/import_local_embeds.py --dataset-dir "$ROOT/$d" --name "$d"
    case $? in 0) ;; 3) echo "   (server copy kept)";; *) echo "   FAILED for $d" >&2 ;; esac
done
