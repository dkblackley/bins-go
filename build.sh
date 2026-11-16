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

# Modules, NGT/HNSW cgo flags and a Go >= 1.24 toolchain (see go_env.sh)
source go_env.sh

echo "Tidying modules..."
go mod tidy

echo "Compiling the binary..."
go build -v -o pir_app ./main.go

echo "Verifying linked libraries..."
ldd ./pir_app | egrep 'ngt|hnsw' || true

# Setup directory structure for logs
mkdir -p logs results

echo "Build complete. Ready to submit SLURM array."