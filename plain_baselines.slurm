#!/bin/bash -l
#SBATCH --job-name=PlainBaselines
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --output=logs/PlainBaselines_%x_%j.out
#SBATCH --error=logs/PlainBaselines_%x_%j.err
#
# Non-private BM25 / dense / hybrid baselines for ONE dataset (cmd/plain_baselines), added to baselines.json.
# Submit through run_plain_baselines.sh, which builds the binary and sizes memory/time per dataset:
#   bash run_plain_baselines.sh                      # every dataset
#   bash run_plain_baselines.sh nq msmarco-v2-25m    # some
# By hand: DATASET=nq BIN=bin/plain_baselines sbatch plain_baselines.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

DATASET="${DATASET:?set DATASET}"
BIN="${BIN:-bin/plain_baselines.latest}"
ROOT="${DATASETS_ROOT:-../datasets}"
OUT="${BASELINES_OUT:-baselines.json}"
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

echo "$(date +%FT%T) $DATASET on $(hostname), $GOMAXPROCS threads, $BIN, root $ROOT -> $OUT"
TIME=(); [[ -x /usr/bin/time ]] && TIME=(/usr/bin/time -v)   # peak RSS at the end of the log
"${TIME[@]}" "$BIN" -name "$DATASET" -dataset "$ROOT" -out "$OUT" -threads "$GOMAXPROCS" ${EXTRA_ARGS:-}
