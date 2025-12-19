#!/bin/bash -l
#SBATCH --job-name=go_job
#SBATCH --partition=bigmem
#SBATCH --output=go.out
#SBATCH --error=go.err
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=240G
#SBATCH --time=2-00:30:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Clean module state and load Hopper defaults + Go
module --quiet purge
module load hosts/hopper
# module load go/1.23.1    # or: module load go/1.22.2  (use the exact version you saw)


export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

# Use cached toolchain via symlink created in $HOME/opt/go1.24
export GOROOT="/home/dblackle/go/pkg/mod/golang.org/toolchain@v0.0.1-go1.24.5.linux-amd64"
export PATH="$GOROOT/bin:$PATH"

# Fail fast with a clear error if it's missing
ls -l "$GOROOT/bin/go"
go version

export GOTOOLCHAIN=local



# If compute nodes can't reach the internet, pre-vendor deps and uncomment:
# export GOFLAGS="-mod=vendor"

# Build & run your main.go
go build -v -o app ./main.go
# default flags are MS marco expected to be found in the ../datasets directory
srun ./app

# srun ./app -n 8841823 -d 192 -m 32 -k 100 -q 1000 \
#   -input ../datasets/Son/my_vectors_192_f64.npy \
#   -query ../datasets/Son/query_192_f64.npy \
#   -step  10 -parallel 3 -rtt 50 \
#   -output  msmarco_embeddings_8841823_192_32_output.txt \
#   -report  msmarco_embeddings_8841823_192_32_report.txt
