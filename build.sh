#!/bin/bash
# setup.sh
# Run this directly on the command line before submitting your SLURM job.

set -euo pipefail


rm -rf ../datasets/trec-covid/*graph_aux.txt
rm -rf ../datasets/scifact/*graph_aux.txt
rm -rf ../datasets/msmarco/*graph_aux.txt

rm -rf ../datasets/trec-covid/*.ngt
rm -rf ../datasets/msmarco/*.ngt
rm -rf ../datasets/scifact/*.ngt

rm -rf ../results/bins_*
rm -rf ../results/pacmann_*


rm -rf logs/Pacmann_*
rm -rf logs/Bins_*

# Clean module state and load Hopper defaults + compilers
module --quiet purge
module load hosts/hopper
module load gnu9/9.3.0

# Native libs
export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"

# cgo compile & link flags
export CGO_CXXFLAGS="-std=c++11"
export CGO_CFLAGS="-I${NGT_PREFIX}/include"
export CGO_LDFLAGS="-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt -L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"

# Go toolchain
export GOROOT="/home/dblackle/go/pkg/mod/golang.org/toolchain@v0.0.1-go1.24.5.linux-amd64"
export PATH="$GOROOT/bin:$PATH"
export GOTOOLCHAIN=local

echo "Tidying modules..."
go mod tidy

echo "Compiling the binary..."
go build -v -o pir_app ./main.go

echo "Verifying linked libraries..."
ldd ./pir_app | egrep 'ngt|hnsw' || true

# Setup directory structure for logs
mkdir -p logs results

echo "Build complete. Ready to submit SLURM array."