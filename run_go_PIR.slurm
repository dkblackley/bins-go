#!/bin/bash -l
#SBATCH --job-name=go_job
#SBATCH --partition=bigmem
#SBATCH --output=go.out
#SBATCH --error=go.err
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000G
#SBATCH --time=2-00:30:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Clean module state and load Hopper defaults + Go
module --quiet purge
module load hosts/hopper
module load go/1.23.1    # or: module load go/1.22.2  (use the exact version you saw)

# Sanity checks
echo "Loaded modules:"; module list
echo "Go toolchain: $(go version)"

# Use all CPUs Slurm gave you; keep caches off $HOME
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"
export GOPATH="${SLURM_TMPDIR:-/tmp}/${USER}/gopath"
export GOMODCACHE="${GOPATH}/pkg/mod"
mkdir -p "$GOMODCACHE"

# If compute nodes can't reach the internet, pre-vendor deps and uncomment:
# export GOFLAGS="-mod=vendor"

# Build & run your main.go
go build -v -o app ./main.go
srun ./app
