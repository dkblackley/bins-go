#!/bin/bash
# Shared environment for the dataset slurm jobs: `source datasets/env.sh`. Everything can be overridden from the
# submitting shell (sbatch passes your environment through), e.g. CONDA_BASE=... CONDA_ENV=... HF_HOME=...

# conda: $CONDA_BASE, else whatever `conda` is on PATH, else the usual install locations
if [[ -z "${CONDA_BASE:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
    for c in "${CONDA_BASE:-}" /projects/evgenios/dblackle/miniforge "$HOME/miniforge3" "$HOME/miniconda3"; do
        [[ -n "$c" && -f "$c/etc/profile.d/conda.sh" ]] && { CONDA_BASE="$c"; break; }
    done
fi
if [[ ! -f "${CONDA_BASE:-}/etc/profile.d/conda.sh" ]]; then
    echo "env.sh: can't find conda (tried '${CONDA_BASE:-}'); set CONDA_BASE=<dir containing etc/profile.d/conda.sh>" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-main}"; set -u

# Hugging Face model cache (e5-base-v2): big and re-downloaded if lost, so keep it off $HOME
if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -d "/scratch/$USER" ]]; then HF_HOME="/scratch/$USER/hf"; else HF_HOME="/projects/evgenios/$USER/hf"; fi
fi
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "env.sh: conda=$CONDA_BASE env=${CONDA_DEFAULT_ENV:-?} python=$(command -v python) HF_HOME=$HF_HOME" >&2
