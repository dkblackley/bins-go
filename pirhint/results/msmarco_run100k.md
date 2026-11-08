## Tree, msmarco (synthetic sizes): 8841823 docs, B=32 r=32 s=8, batch 128, Stage3 batch 32, dim 192

Stage1 items per level: [2599883 19483532 80430370 149982912]; Stage2 items: 219720999; leaves: 276307

### Tree client storage

| layout | client | vs first | s1 | s2 | s3 | server DB | online/search | refresh/search | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 5.9 GB | 1.00x | 2.2 GB | 658 MB | 3.1 GB | 38.0 GB | 18.6 MB | 57.9 MB | one Cuckoo DB per Stage1 level, one Stage2 DB, Stage3 fp32 |
| s1 one DB for all levels | 5.0 GB | 0.85x | 1.3 GB | 658 MB | 3.1 GB | 38.0 GB | 28.4 MB | 71.1 MB | every level in one Cuckoo table |
| s1 per level per hash | 6.7 GB | 1.14x | 3.0 GB | 658 MB | 3.1 GB | 38.0 GB | 17.1 MB | 55.9 MB | 2 half-size tables per level, same batch each |
| s1 per level per hash, batch/2 | 5.9 GB | 1.00x | 2.2 GB | 658 MB | 3.1 GB | 38.0 GB | 18.1 MB | 57.0 MB | 2 half-size tables per level, half the batch each |
| s2 one DB per sub-block | 7.8 GB | 1.33x | 2.2 GB | 2.6 GB | 3.1 GB | 62.6 GB | 36.3 MB | 70.2 MB | 4 DBs with 1-byte values |
| s2 one DB per leaf block | 33.7 GB | 5.72x | 2.2 GB | 28.5 GB | 3.1 GB | 38.0 GB | 455 MB | 578 MB | 276307 small DBs, 1 partition each |
| download if cheaper | 5.9 GB | 1.00x | 2.2 GB | 658 MB | 3.1 GB | 38.0 GB | 18.6 MB | 57.9 MB | client takes the plain DB when it is smaller than its hints |
| s3 fp16 | 4.3 GB | 0.74x | 2.2 GB | 658 MB | 1.6 GB | 34.9 GB | 17.9 MB | 41.3 MB | Stage3 embeddings stored as fp16 |
| s3 int8 | 3.6 GB | 0.61x | 2.2 GB | 658 MB | 800 MB | 33.3 GB | 17.5 MB | 33.0 MB | Stage3 embeddings stored as int8 |
| s3 batch 16 | 4.7 GB | 0.80x | 2.2 GB | 658 MB | 1.9 GB | 38.0 GB | 18.4 MB | 61.5 MB | --stage3-batch-size 16 (8 partitions) |
| s3 batch 8 | 4.4 GB | 0.75x | 2.2 GB | 658 MB | 1.6 GB | 38.0 GB | 18.2 MB | 67.0 MB | --stage3-batch-size 8 (4 partitions) |
| s1/s2 batch 64 | 5.2 GB | 0.89x | 1.6 GB | 547 MB | 3.1 GB | 38.0 GB | 19.9 MB | 61.5 MB | --pir-batch-size 64 |
| s1/s2 batch 32 | 4.6 GB | 0.78x | 1.1 GB | 341 MB | 3.1 GB | 38.0 GB | 26.2 MB | 66.1 MB | --pir-batch-size 32 |
| chunk factor 1 | 4.7 GB | 0.79x | 1.8 GB | 562 MB | 2.3 GB | 38.0 GB | 35.5 MB | 57.9 MB | --pir-chunk-factor 1 |
| chunk factor 0.5 | 4.0 GB | 0.68x | 1.5 GB | 551 MB | 1.9 GB | 38.0 GB | 69.4 MB | 57.9 MB | --pir-chunk-factor 0.5 |
| query budget 0.5 | 4.1 GB | 0.70x | 1.5 GB | 461 MB | 2.2 GB | 38.0 GB | 18.6 MB | 116 MB | --pir-query-budget 0.5 |
| query budget 0.25 | 3.3 GB | 0.55x | 1.1 GB | 363 MB | 1.8 GB | 38.0 GB | 18.6 MB | 232 MB | --pir-query-budget 0.25 |
| combined | 1.8 GB | 0.30x | 1.1 GB | 366 MB | 387 MB | 33.3 GB | 34.5 MB | 66.0 MB | download if cheaper + s3 int8 + chunk factor 1 + query budget 0.5 |
| combined + smaller batches | 1.0 GB | 0.17x | 613 MB | 183 MB | 245 MB | 33.3 GB | 49.8 MB | 84.1 MB | combined, --pir-batch-size 32, --stage3-batch-size 16 |

## Measured with pir-old on random data, 100000 docs (Stage1 [661257 1191561 1702180], Stage2 2485018 items)

| layout | client (measured) | client (pirhint) | online, one search | preprocess | query |
|---|---|---|---|---|---|
| current | 500 MB | 500 MB | 3.2 MB | 5.59s | 0.043s |
| s1 one DB for all levels | 401 MB | 401 MB | 3.5 MB | 5.55s | 0.093s |
| s1 per level per hash | 566 MB | 566 MB | 3.2 MB | 5.29s | 0.037s |
| s1 per level per hash, batch/2 | 500 MB | 500 MB | 3.0 MB | 5.10s | 0.038s |
| s2 one DB per sub-block | 685 MB | 685 MB | 4.4 MB | 11.63s | 0.087s |
| s2 one DB per leaf block | 768 MB | 768 MB | 6.2 MB | 13.59s | 0.828s |
| download if cheaper | 363 MB | 363 MB | 1.8 MB | 5.13s | 0.036s |
| s3 fp16 | 395 MB | 395 MB | 2.6 MB | 5.08s | 0.040s |
| s3 int8 | 343 MB | 343 MB | 2.1 MB | 4.85s | 0.040s |
| s3 batch 16 | 464 MB | 464 MB | 2.7 MB | 5.19s | 0.039s |
| s3 batch 8 | 398 MB | 398 MB | 2.6 MB | 5.14s | 0.037s |
| s1/s2 batch 64 | 405 MB | 405 MB | 3.6 MB | 5.65s | 0.043s |
| s1/s2 batch 32 | 365 MB | 365 MB | 3.6 MB | 5.09s | 0.067s |
| chunk factor 1 | 392 MB | 392 MB | 4.8 MB | 6.77s | 0.034s |
| chunk factor 0.5 | 370 MB | 370 MB | 9.2 MB | 13.59s | 0.053s |
| query budget 0.5 | 351 MB | 351 MB | 3.2 MB | 4.04s | 0.035s |
| query budget 0.25 | 291 MB | 291 MB | 3.1 MB | 3.48s | 0.037s |
| combined | 169 MB | 169 MB | 3.3 MB | 4.95s | 0.032s |
| combined + smaller batches | 98.6 MB | 98.6 MB | 4.7 MB | 5.31s | 0.050s |
