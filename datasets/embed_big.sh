#!/bin/bash
# Submit the sharded embedding chain for a big corpus (default MS MARCO v2, 138M passages).
#   bash datasets/embed_big.sh                 # msmarco-v2, 24 shards (~6M passages / ~1-2h each on an A100)
#   DATASET=foo NUM_SHARDS=8 bash datasets/embed_big.sh
# Output is ~106 GB of float32 (138M x 192); the shards need the same again until "finish" deletes them.
set -euo pipefail
export DATASET="${DATASET:-msmarco-v2}"
export NUM_SHARDS="${NUM_SHARDS:-24}"

pca=$(MODE=pca sbatch --parsable --time=0-02:00:00 datasets/embed.slurm)
shards=$(MODE=shard sbatch --parsable --dependency=afterok:${pca} --array=0-$((NUM_SHARDS - 1)) datasets/embed.slurm)
finish=$(MODE=finish sbatch --parsable --dependency=afterok:${shards} --time=0-06:00:00 datasets/embed.slurm)
echo "Submitted ${DATASET}: pca=${pca} shards=${shards} (x${NUM_SHARDS}) finish=${finish}"
