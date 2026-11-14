#!/bin/bash -l
#SBATCH --job-name=Coeus_Baseline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00              # a config only builds its postings from the cached tokens
#SBATCH --account=evgenios
#SBATCH --qos=normal
#SBATCH --array=0-23               # for the default scifact + msmarco; other datasets: see --count below
#SBATCH --output=logs/Coeus_%A_%a.out
#SBATCH --error=logs/Coeus_%A_%a.err

# Coeus baseline (see baselines/README.md): exact TF-IDF quality on the real corpus + server time from
# Coeus' own matmult/opt2 benchmark (calibrated once, cached next to the binary) + communication from
# the artifact's encryption parameters.
# Needs `baselines/setup_baselines.sh coeus` first.
#
# Sweep: keyword dictionary size F (multiples of 8192; the paper used 65,536) x tf weighting x k
#   2 datasets x 3 F x 2 tf x 2 k = 24 tasks. Configs with F past the vocabulary exit without writing anything
#   (they'd duplicate a smaller F with extra upload), so scifact produces fewer results.
#
# Quality + communication (bytes -> LAN/WAN time) only: the opt2 server-time benchmark is never run. Fits in 32G / 6 h
# up to msmarco (postings ~12 B per (doc, term): ~5 GB on msmarco); msmarco-v2 (~50 GB) needs --mem=120G
# --time=2-00:00:00.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Datasets, e.g.
#   COEUS_DATASETS="nq dbpedia-entity fever" bash run_coeus.slurm --count     # prints the --array range to use
#   COEUS_DATASETS="nq dbpedia-entity fever" sbatch --array=0-<N-1> run_coeus.slurm
# Any dataset in datasets/prepare_dataset.py's layout works. The corpus is streamed, so memory is the postings only
# (~12 B per (doc, term) pair: about 50 GB for msmarco-v2). Tokenising 138M docs takes hours, hence the 2-day limit.
#
# The tokenised corpus depends only on the dataset, so it is cached in $COEUS_TOKEN_CACHE (default
# <datasets>/coeus_cache; ~4 B per token, ~40 GB for msmarco-v2) and shared by every F/tf/k config.
# COEUS_STAGE=prep builds it, one task per dataset; datasets/run_all.sh submits that first and the configs with
# --dependency=afterok on it. A config task fails straight away if the cache isn't there (--require-token-cache)
# rather than tokenising itself, so submit Coeus through run_all.sh (STAGES=coeus).
#   COEUS_DATASETS=msmarco COEUS_STAGE=prep bash run_coeus.slurm --count      # 1 per dataset
DATASET_DIR="${DATASET_DIR:-../../datasets}"
COEUS_DATASETS="${COEUS_DATASETS:-scifact msmarco}"
COEUS_STAGE="${COEUS_STAGE:-run}"
COEUS_TOKEN_CACHE="${COEUS_TOKEN_CACHE:-${DATASET_DIR}/coeus_cache}"
PREP_ARR=($COEUS_DATASETS)
DATA_ARR=(); F_ARR=(); TF_ARR=(); K_ARR=()
for d in $COEUS_DATASETS; do
    for f in 8192 32768 262144; do   # 2026-11-09, was 8192 16384 32768 65536 131072 262144
        for tf in raw log; do
            for k in 10 100; do
                DATA_ARR+=("$d"); F_ARR+=("$f"); TF_ARR+=("$tf"); K_ARR+=("$k")
            done
        done
    done
done

if [[ "${1:-}" == "--count" ]]; then
    if [[ "$COEUS_STAGE" == prep ]]; then echo "${#PREP_ARR[@]} prep tasks (use --array=0-$(( ${#PREP_ARR[@]} - 1 )))"
    else echo "${#DATA_ARR[@]} configs (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))"; fi
    exit 0
fi

source datasets/env.sh     # conda env main (finds conda itself; CONDA_BASE/CONDA_ENV to override)

task_id=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$COEUS_STAGE" == prep ]]; then   # tokenise this dataset's corpus into the shared cache, then done
    dataset=${PREP_ARR[$task_id]}
    echo "Coeus prep: tokenising $dataset into $COEUS_TOKEN_CACHE"
    python baselines/coeus/run_coeus.py --dataset "$DATASET_DIR" --name "$dataset" \
        --token-cache "$COEUS_TOKEN_CACHE" --tokenize-only
    exit 0
fi
dataset=${DATA_ARR[$task_id]}
features=${F_ARR[$task_id]}
tf=${TF_ARR[$task_id]}
k_val=${K_ARR[$task_id]}

OUT="${RESULTS_ROOT:-/scratch/$USER/results}/coeus_${dataset}_k${k_val}_f${features}_tf${tf}"
if [[ -s "$OUT/metadata.json" && -z "${FORCE:-}" ]]; then echo "Already done: $OUT (FORCE=1 reruns)"; exit 0; fi
echo "Coeus: dataset=$dataset F=$features tf=$tf k=$k_val"
python baselines/coeus/run_coeus.py \
    --dataset "$DATASET_DIR" \
    --name "$dataset" \
    --k "$k_val" \
    --features "$features" \
    --tf "$tf" \
    --results-root "${RESULTS_ROOT:-/scratch/$USER/results}" \
    --token-cache "$COEUS_TOKEN_CACHE" --require-token-cache \
    --parallelism "${SLURM_CPUS_PER_TASK:-1}"
