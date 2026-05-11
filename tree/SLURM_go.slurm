#!/bin/bash -l
#SBATCH --job-name=bm25_tree
#SBATCH --partition=bigmem
#SBATCH --output=go.out
#SBATCH --error=go.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=240G
#SBATCH --time=2-00:30:00

# set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Clean module state and load Hopper defaults + Go
module --quiet purge
module load hosts/hopper
# module load go/1.23.1    # or: module load go/1.22.2  (use the exact version you saw)


module load gnu9/9.3.0

export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"

# Use cached toolchain via symlink created in $HOME/opt/go1.24
export GOROOT="/home/dblackle/go/pkg/mod/golang.org/toolchain@v0.0.1-go1.24.5.linux-amd64"
export PATH="$GOROOT/bin:$PATH"

# Fail fast with a clear error if it's missing
ls -l "$GOROOT/bin/go"
go version

export GOTOOLCHAIN=local

go mod tidy



# set -euo pipefail
# cd "$SLURM_SUBMIT_DIR"

# # Clean module state and load Hopper defaults + Go
# module --quiet purge
# module load hosts/hopper
# module load go/1.23.1    

# # Sanity checks
# echo "Loaded modules:"; module list
# echo "Go toolchain: $(go version)"

# # Use all CPUs Slurm gave you; keep caches off $HOME
# export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-1}"
# export GOPATH="${SLURM_TMPDIR:-/tmp}/${USER}/gopath"
# export GOMODCACHE="${GOPATH}/pkg/mod"
# mkdir -p "$GOMODCACHE"

# export GOTOOLCHAIN=local

# If compute nodes can't reach the internet, pre-vendor deps and uncomment:
# export GOFLAGS="-mod=vendor"

# go build -v -o app ./cmd/main.go
# srun ./app --qrels data/qrels.dev.small.tsv --index data/msmarco_reordered --queries data/queries.dev.small.tsv --max-queries 6980 --precompute --doc-embed data/stage3_doc_aligned_shuffle.npy --doc-id-map data/stage3_doc_aligned.npy.ids --query-embed data/stage3_query.npy --embed-dim 192 --stage1-data-bin data/stage1_data.bin --stage1-idmap-bin data/stage1/stage1_idmap.bin --vocab data/stage1_vocab.json --query-words 8 --pir-batch-size 128 2>&1 | grep "ms\|MRR"


go build -o bm25-search ./cmd



RESULTS_BASE="../results"
K_VALUES=(10 50 100 500 1000)
#K_VALUES=(50)

for k in "${K_VALUES[@]}"; do
  outdir="${RESULTS_BASE}/tree_${k}"
  mkdir -p "${outdir}"

  tsv_out="${outdir}/tree_out_k${k}.tsv"

  echo "=== k=${k} ==="

  # 1) Run Go app (write JSON directly into the results directory)
  ./bm25-search --k "${k}" --qrels data/qrels.dev.small.tsv --index data/msmarco_reordered --queries data/queries.dev.small.tsv --max-queries 6980 --precompute --doc-embed data/stage3_doc_aligned_shuffle.npy --doc-id-map data/stage3_doc_aligned.npy.ids --query-embed data/stage3_query.npy --embed-dim 192 --stage1-data-bin data/stage1_data.bin --stage1-idmap-bin data/stage1/stage1_idmap.bin --vocab data/stage1_vocab.json --query-words 8 --pir-batch-size 128 2>&1 | grep "ms\|MRR"

  mv "approx_precompute_reranked_${k}.tsv" "${tsv_out}"

  [[ -f go.err ]] || { echo "ERROR: go.err not found in $(pwd)";  }
  [[ -f go.out ]] || { echo "ERROR: go.out not found in $(pwd)"; }

  cp -f go.err "${outdir}/go_${k}.err"
  cp -f go.out "${outdir}/go_${k}.out"

  echo "Saved: ${outdir}"
done


echo "\nDONE\n"
