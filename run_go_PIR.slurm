#!/bin/bash -l
#SBATCH --job-name=go_job
#SBATCH --partition=bigmem
#SBATCH --output=go.out
#SBATCH --error=go.err
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500G
#SBATCH --time=2-00:30:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Clean module state and load Hopper defaults + Go
module --quiet purge
module load hosts/hopper
# module load go/1.23.1    # or: module load go/1.22.2  (use the exact version you saw)


module load gnu9/9.3.0

# --- native libs installed under $HOME ---
export NGT_PREFIX="$HOME/opt/ngt-amd"
export HNSW_PREFIX="$HOME/opt/hnsw"

# cgo compile & link flags
export CGO_CXXFLAGS="-std=c++11"
export CGO_CFLAGS="-I${NGT_PREFIX}/include"   # NGT headers used by gongt
export CGO_LDFLAGS="\
-L${NGT_PREFIX}/lib -Wl,-rpath,${NGT_PREFIX}/lib -lngt \
-L${HNSW_PREFIX}/lib -Wl,-rpath,${HNSW_PREFIX}/lib -lhnsw"


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

test -f "${NGT_PREFIX}/lib/libngt.so"  || { echo "Missing libngt.so";  exit 1; }
test -f "${HNSW_PREFIX}/lib/libhnsw.so" || { echo "Missing libhnsw.so"; exit 1; }


go mod tidy

# Build & run your main.go
go build -v -o app ./main.go

# sanity: confirm both libs resolve
echo "Linked libs:"
ldd ./app | egrep 'ngt|hnsw' || true



srun ./app -n 8841823 -t Pacmann -filename msmarco -k 100
python3 json_to_tsv.py out.json pacmann_out.tsv

#srun ./app -n 8841823 -t bins -filename msmarco -k 100 -save -thresh 5
#python3 json_to_tsv.py out.json bins_out.tsv
# python3 re_rank --qrels ../datasets/msmarco/qrels/dev.tsv --step3-output bins_out.tsv --queries ../datasets/msmarco/queries.dev.small.tsv --documents documents.txt --k 10


# srun ./app -n 8841823 -d 192 -m 32 -k 100 -q 1000 \
#   -input ../datasets/Son/my_vectors_192_f64.npy \
#   -query ../datasets/Son/query_192_f64.npy \
#   -step  10 -parallel 3 -rtt 50 \
#   -output  msmarco_embeddings_8841823_192_32_output.txt \
#   -report  msmarco_embeddings_8841823_192_32_report.txt


