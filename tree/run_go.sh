#!/bin/bash
export PATH=$PATH:/usr/local/go/bin
go build -o bm25-search ./cmd

./bm25-search --qrels data/qrels.dev.small.tsv --index data/msmarco_reordered --queries data/queries.dev.small.tsv --max-queries 6980 --precompute --doc-embed data/stage3_doc_aligned_shuffle.npy --doc-id-map data/stage3_doc_aligned.npy.ids --query-embed data/stage3_query.npy --embed-dim 192 --stage1-data-bin data/stage1_data.bin --stage1-idmap-bin data/stage1/stage1_idmap.bin --vocab data/stage1_vocab.json --query-words 8 --pir-batch-size 128 2>&1 | grep "ms\|MRR" 
