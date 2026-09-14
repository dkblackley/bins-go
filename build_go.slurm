#!/bin/bash -l
#SBATCH --job-name=Bins_Build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=build.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

bash build.sh