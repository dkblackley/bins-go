#!/bin/bash
# Queue everything the grid search needs, but not the grid search itself: download + convert, e5/PCA embeddings on
# GPU, and the Lucene index + tree artifacts. Same chaining as run_all.sh (sbatch --dependency=afterok), so submit and
# walk away; launch the grids later with STAGES="bins pacmann tree" bash datasets/run_all.sh <datasets>.
#
#   bash datasets/build_all.sh nq hotpotqa fever
#   bash datasets/build_all.sh msmarco-v2-all          # msmarco-v2, then the 25m/50m/100m subsamples once it's embedded
#   STAGES="embed tree-db" bash datasets/build_all.sh nq   # skip stages that are already done
#   DRY_RUN=1 bash datasets/build_all.sh nq            # print the sbatch commands, submit nothing
#
# Stages, in order (default: all three):
#   prep     download + convert                 datasets/prep_datasets.slurm
#   embed    e5 + PCA embeddings (GPU)           datasets/embed.slurm (msmarco-v2/miracl-en: PCA -> shards -> merge)
#   tree-db  Lucene index + tree artifacts       datasets/tree_artifacts.slurm          (after embed)
# Doesn't need pir_app. DATASETS_ROOT (default ../datasets), NUM_SHARDS and WORKERS work as in run_all.sh.
set -euo pipefail

[[ $# -gt 0 ]] || { sed -n 2,15p "$0"; exit 1; }

export STAGES="${STAGES:-prep embed tree-db}"
for s in $STAGES; do
    case "$s" in
        prep|embed|tree-db) ;;
        *) echo "build_all.sh only does prep/embed/tree-db (got '$s'); use run_all.sh for the grids" >&2; exit 1 ;;
    esac
done

exec bash "$(dirname "$0")/run_all.sh" "$@"
