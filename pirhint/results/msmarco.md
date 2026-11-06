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

#### DBs of "current"

| DB | rows | entry | partitions x rows | chunk / set | hints | DB | hints/DB | downloaded |
|---|---|---|---|---|---|---|---|---|
| s1/level0 | 6499708 | 40 B | 64 x 101558 | 1024 / 100 | 118 MB | 248 MB | 0.48 | false |
| s1/level1 | 48708830 | 40 B | 64 x 761076 | 2048 / 372 | 314 MB | 1.8 GB | 0.17 | false |
| s1/level2 | 201075925 | 40 B | 64 x 3141812 | 4096 / 768 | 678 MB | 7.5 GB | 0.09 | false |
| s1/level3 | 374957280 | 40 B | 64 x 5858708 | 8192 / 716 | 1.1 GB | 14.0 GB | 0.08 | false |
| s2/cuckoo | 549302498 | 16 B | 64 x 8582852 | 8192 / 1048 | 658 MB | 8.2 GB | 0.08 | false |
| s3/embeddings | 1105228 | 6144 B | 16 x 69077 | 1024 / 68 | 3.1 GB | 6.3 GB | 0.49 | false |

#### DBs of "combined + smaller batches"

| DB | rows | entry | partitions x rows | chunk / set | hints | DB | hints/DB | downloaded |
|---|---|---|---|---|---|---|---|---|
| s1/level0 | 6499708 | 40 B | 16 x 406232 | 1024 / 400 | 31.9 MB | 248 MB | 0.13 | false |
| s1/level1 | 48708830 | 40 B | 16 x 3044302 | 2048 / 1488 | 78.6 MB | 1.8 GB | 0.04 | false |
| s1/level2 | 201075925 | 40 B | 16 x 12567246 | 4096 / 3072 | 196 MB | 7.5 GB | 0.03 | false |
| s1/level3 | 374957280 | 40 B | 16 x 23434830 | 8192 / 2864 | 306 MB | 14.0 GB | 0.02 | false |
| s2/cuckoo | 549302498 | 16 B | 16 x 34331407 | 8192 / 4192 | 183 MB | 8.2 GB | 0.02 | false |
| s3/embeddings | 1105228 | 1536 B | 8 x 138154 | 512 / 272 | 245 MB | 1.6 GB | 0.15 | false |

## Bins, msmarco: 8841823 docs, bs=1000000, dpb=100, dim 192

### Bins client storage

| layout | client | vs first | id | vec | all | server DB | online/search | refresh/search | note |
|---|---|---|---|---|---|---|---|---|---|
| vec1 (one DB) | 24.5 GB | 1.00x | 0 B | 0 B | 24.5 GB | 71.5 GB | 2.1 MB | 65.3 MB | ids and embeddings in each bin |
| vec0 (ids, then embeddings) | 3.3 GB | 0.13x | 167 MB | 3.1 GB | 0 B | 6.7 GB | 2.3 MB | 32.0 MB | Vec DB batch 300 |
| vec0 + download if cheaper | 3.3 GB | 0.13x | 167 MB | 3.1 GB | 0 B | 6.7 GB | 2.3 MB | 32.0 MB |  |
| vec0 + fp16 | 1.7 GB | 0.07x | 167 MB | 1.6 GB | 0 B | 3.5 GB | 1.6 MB | 16.2 MB | Vec DB embeddings as fp16 |
| vec0 + int8 | 993 MB | 0.04x | 167 MB | 826 MB | 0 B | 2.0 GB | 1.2 MB | 8.3 MB | Vec DB embeddings as int8 |
| vec0 + Vec batch 75 | 1.7 GB | 0.07x | 167 MB | 1.6 GB | 0 B | 6.7 GB | 2.1 MB | 37.8 MB | fewer partitions; more Piano queries land in each, more sequential work |
| vec0 + Vec batch 18 | 949 MB | 0.04x | 167 MB | 782 MB | 0 B | 6.7 GB | 2.6 MB | 52.4 MB | fewer partitions; more Piano queries land in each, more sequential work |

#### DBs of "vec0 (ids, then embeddings)"

| DB | rows | entry | partitions x rows | chunk / set | hints | DB | hints/DB | downloaded |
|---|---|---|---|---|---|---|---|---|
| id/bins | 1000000 | 400 B | 12 x 83334 | 1024 / 84 | 167 MB | 381 MB | 0.44 | false |
| vec/embeddings | 8841823 | 768 B | 150 x 58946 | 512 / 116 | 3.1 GB | 6.3 GB | 0.49 | false |
