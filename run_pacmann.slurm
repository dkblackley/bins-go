#!/bin/bash -l
#SBATCH --job-name=pacmann_pir
#SBATCH --partition=bigmem
#SBATCH --output=pac.out
#SBATCH --error=pac.err
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=750G
#SBATCH --time=2-06:30:00

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

#srun ./app -n 199 -t Pacmann -name debug -outFile pacmann_debug.json
#python3 json_to_tsv.py pacmann_debug.json pacmann_debug_out.tsv

RESULTS_BASE="../results"
K_VALUES=(10 50 100 500 1000)
#K_VALUES=(50)

for k in "${K_VALUES[@]}"; do
  outdir="${RESULTS_BASE}/pacmann_${k}"
  mkdir -p "${outdir}"

  json_out="${outdir}/pacmann_out_k${k}.json"
  tsv_out="${outdir}/pacmann_out_k${k}.tsv"

  echo "=== k=${k} ==="

  # 1) Run Go app (write JSON directly into the results directory)
  srun ./app -n 8841823 -t Pacmann -name msmarco -k "${k}" -outFile "${json_out}"

  # 2) Convert JSON -> TSV (write TSV directly into the results directory)
  python3 json_to_tsv.py "${json_out}" "${tsv_out}"


  # 4) Copy pacmann.err / pacmann.out into the results directory with k-specific names
  [[ -f pac.err ]] || { echo "ERROR: pac.err not found in $(pwd)";  }
  [[ -f pac.out ]] || { echo "ERROR: pac.out not found in $(pwd)"; }

  cp -f pac.err "${outdir}/pac_${k}.err"
  cp -f pac.out "${outdir}/pac_${k}.out"

  echo "Saved: ${outdir}"
done

echo "All done."
