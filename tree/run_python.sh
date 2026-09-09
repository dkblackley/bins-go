#!/bin/bash
# python prepare_data.py --index msmarco_reordered --terms-from-index --layout-stats data/layout_with_stats.json

# python prepare_data_stage2.py --index msmarco_reordered --vocab stage1_vocab.json --out-dir stage2 --r 128

python opt3.py --index msmarco_reordered \
  --queries queries.dev.small.tsv \
  --k 200 --k_candidates 200 --B 32 --r 128 --L 200 --s 8 --max-queries 6980 \
  --qrels qrels.dev.small.tsv \
--precompute --stage1-data-bin stage1_data.bin --stage1-idmap-bin stage1/stage1_idmap.bin --vocab stage1_vocab.json --max-terms 16

python step4.py --qrels qrels.dev.small.tsv --step3-output approx_200_200_200.tsv --documents /media/tson1997/indexes/collection.tsv --queries queries.dev.small.tsv