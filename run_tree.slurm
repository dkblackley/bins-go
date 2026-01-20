#!/bin/bash -l
#SBATCH --job-name=tree_pir
#SBATCH --partition=bigmem
#SBATCH --output=tree.out
#SBATCH --error=tree.err
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

# Just do this outside instead of in the slurm script to avoid headaches
go mod tidy
#
## Build & run your main.go
go build -v -o tree_bin ./main.go

# sanity: confirm both libs resolve
echo "Linked libs:"
ldd ./tree_bin | egrep 'ngt|hnsw' || true


RESULTS_BASE="../results"
K_VALUES=(10 50 100 500 1000)
# K_VALUES=(1000)

for k in "${K_VALUES[@]}"; do
  outdir="${RESULTS_BASE}/tree_${k}"
  mkdir -p "${outdir}"

  json_out="${outdir}/tree_out_k${k}.json"
  tsv_out="${outdir}/tree_out_k${k}.tsv"

  echo "=== k=${k} ==="

  # 1) Run Go app (write JSON directly into the results directory)
  # srun ./tree --n 8841823 --t tree --name msmarco --k "${k}" --outFile "${json_out}" --dataset ../../../datasets --qrels ../../../datasets/tree/qrels.dev.small.tsv --index ../../../datasets/tree/msmarco_reordered --queries ../../../datasets/tree/queries.dev.small.tsv --max-queries 6980 --precompute --doc-embed ../../../datasets/tree/stage3_doc_aligned_shuffle.npy --doc-id-map ../../../datasets/tree/stage3_doc_aligned.npy.ids --query-embed ../../../datasets/tree/stage3_query.npy --embed-dim 192 --stage1-data-bin ../../../datasets/tree/stage1_data.bin --stage1-idmap-bin ../../../datasets/tree/stage1/stage1_idmap.bin --vocab ../../../datasets/tree/stage1_vocab.json --query-words 8 --pir-batch-size 128 2>&1 | grep "ms\|MRR"
  srun ./tree_bin --n 8841823 --t tree --name msmarco --k "${k}" --outFile "${json_out}" --dataset ../../../datasets --qrels ../../../datasets/tree/qrels.dev.small.tsv --index ../../../datasets/tree/msmarco_reordered --queries ../../../datasets/tree/queries.dev.small.tsv --max-queries 6980 --precompute --doc-embed ../../../datasets/tree/stage3_doc_aligned_shuffle.npy --doc-id-map ../../../datasets/tree/stage3_doc_aligned.npy.ids --query-embed ../../../datasets/tree/stage3_query.npy --embed-dim 192 --stage1-data-bin ../../../datasets/tree/stage1_data.bin --stage1-idmap-bin ../../../datasets/tree/stage1/stage1_idmap.bin --vocab ../../../datasets/tree/stage1_vocab.json --query-words 8 --pir-batch-size 128

  # 2) Convert JSON -> TSV (write TSV directly into the results directory)
  python3 json_to_tsv.py "${json_out}" "${tsv_out}"

  # 4) Copy tree.err / tree.out into the results directory with k-specific names
  [[ -f tree.err ]] || { echo "ERROR: tree.err not found in $(pwd)";  }
  [[ -f tree.out ]] || { echo "ERROR: tree.out not found in $(pwd)"; }

  # mv step4_reranked_output.tsv "${outdir}/reranked_tree_k${k}.tsv"
  mv "tree${k}_metadata.json" "${outdir}/tree_${k}_metadata.json"

  cp -f tree.err "${outdir}/tree_${k}.err"
  cp -f tree.out "${outdir}/tree_${k}.out"

  echo "Saved: ${outdir}"
done

echo "All done."


#srun ./app -n 8841823 -t bins -name msmarco -k 1000 -save -thresh 5 -outFile bins_out.json
#python3 json_to_tsv.py bins_out.json bins_out.tsv
#
#
#set -e
#
#source /home/dblackle/miniconda3/etc/profile.d/conda.sh
#
#export CONDA_ENVS_PATH=/scratch/dblackle/conda/envs
#export CONDA_PKGS_DIRS=/scratch/dblackle/conda/pkgs
#
#conda activate /home/dblackle/.conda/envs/main
#
#
#
## conda install -c conda-forge sentencepiece -y
#
#echo "CONDA DONE"
#
#
#python3 re_rank.py --qrels ../datasets/msmarco/qrels/dev.tsv --step3-output bins_out.tsv --queries ../datasets/msmarco/queries.dev.small.tsv --documents ../datasets/msmarco/collection.tsv --k 1000
