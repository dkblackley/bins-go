## Tree, scifact (synthetic sizes): 5183 docs, B=16 r=8 s=4, batch 128, Stage3 batch 32, dim 192

Stage1 items per level: [29309 118112 237575]; Stage2 items: 381267; leaves: 324

### Tree client storage

| layout | client | vs first | s1 | s2 | s3 | server DB | online/search | refresh/search | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 87.9 MB | 1.00x | 34.6 MB | 18.8 MB | 34.6 MB | 33.0 MB | 2.2 MB | 3.5 MB | one Cuckoo DB per Stage1 level, one Stage2 DB, Stage3 fp32 |
| s1 one DB for all levels | 72.1 MB | 0.82x | 18.8 MB | 18.8 MB | 34.6 MB | 33.0 MB | 2.6 MB | 3.7 MB | every level in one Cuckoo table |
| s1 per level per hash | 99.3 MB | 1.13x | 45.9 MB | 18.8 MB | 34.6 MB | 33.0 MB | 2.2 MB | 3.5 MB | 2 half-size tables per level, same batch each |
| s1 per level per hash, batch/2 | 87.9 MB | 1.00x | 34.6 MB | 18.8 MB | 34.6 MB | 33.0 MB | 2.2 MB | 3.5 MB | 2 half-size tables per level, half the batch each |
| s2 one DB per sub-block | 144 MB | 1.64x | 34.6 MB | 75.0 MB | 34.6 MB | 76.7 MB | 3.3 MB | 4.4 MB | 4 DBs with 1-byte values |
| s2 one DB per leaf block | 108 MB | 1.23x | 34.6 MB | 38.9 MB | 34.6 MB | 33.0 MB | 2.6 MB | 3.9 MB | 324 small DBs, 1 partition each |
| download if cheaper | 33.0 MB | 0.38x | 14.7 MB | 14.5 MB | 3.8 MB | 33.0 MB | 0 B | 0 B | client takes the plain DB when it is smaller than its hints |
| s3 fp16 | 70.7 MB | 0.80x | 34.6 MB | 18.8 MB | 17.3 MB | 31.1 MB | 1.5 MB | 2.2 MB | Stage3 embeddings stored as fp16 |
| s3 int8 | 62.0 MB | 0.71x | 34.6 MB | 18.8 MB | 8.7 MB | 30.2 MB | 1.2 MB | 1.5 MB | Stage3 embeddings stored as int8 |
| s3 batch 16 | 87.9 MB | 1.00x | 34.6 MB | 18.8 MB | 34.6 MB | 33.0 MB | 2.0 MB | 3.4 MB | --stage3-batch-size 16 (8 partitions) |
| s3 batch 8 | 73.6 MB | 0.84x | 34.6 MB | 18.8 MB | 20.3 MB | 33.0 MB | 1.8 MB | 3.4 MB | --stage3-batch-size 8 (4 partitions) |
| s1/s2 batch 64 | 74.4 MB | 0.85x | 24.6 MB | 15.2 MB | 34.6 MB | 33.0 MB | 2.2 MB | 3.6 MB | --pir-batch-size 64 |
| s1/s2 batch 32 | 62.8 MB | 0.71x | 18.2 MB | 10.1 MB | 34.6 MB | 33.0 MB | 2.5 MB | 3.7 MB | --pir-batch-size 32 |
| chunk factor 1 | 64.5 MB | 0.73x | 26.2 MB | 15.0 MB | 23.3 MB | 33.0 MB | 2.9 MB | 3.5 MB | --pir-chunk-factor 1 |
| chunk factor 0.5 | 56.9 MB | 0.65x | 19.9 MB | 12.9 MB | 24.0 MB | 33.0 MB | 4.5 MB | 3.5 MB | --pir-chunk-factor 0.5 |
| query budget 0.5 | 69.1 MB | 0.79x | 27.4 MB | 13.1 MB | 28.6 MB | 33.0 MB | 2.2 MB | 5.4 MB | --pir-query-budget 0.5 |
| query budget 0.25 | 61.5 MB | 0.70x | 22.6 MB | 10.3 MB | 28.6 MB | 33.0 MB | 2.2 MB | 7.1 MB | --pir-query-budget 0.25 |
| combined | 23.3 MB | 0.26x | 12.9 MB | 9.4 MB | 972 KB | 30.2 MB | 1.1 MB | 1.0 MB | download if cheaper + s3 int8 + chunk factor 1 + query budget 0.5 |
| combined + smaller batches | 14.4 MB | 0.16x | 8.8 MB | 4.6 MB | 972 KB | 30.2 MB | 1.9 MB | 1.8 MB | combined, --pir-batch-size 32, --stage3-batch-size 16 |

#### DBs of "current"

| DB | rows | entry | partitions x rows | chunk / set | hints | DB | hints/DB | downloaded |
|---|---|---|---|---|---|---|---|---|
| s1/level0 | 73273 | 16 B | 64 x 1145 | 128 / 12 | 6.0 MB | 1.1 MB | 5.37 | false |
| s1/level1 | 295280 | 16 B | 64 x 4614 | 256 / 20 | 12.7 MB | 4.5 MB | 2.81 | false |
| s1/level2 | 593938 | 16 B | 64 x 9281 | 256 / 40 | 15.9 MB | 9.1 MB | 1.76 | false |
| s2/cuckoo | 953168 | 16 B | 64 x 14894 | 256 / 60 | 18.8 MB | 14.5 MB | 1.29 | false |
| s3/embeddings | 648 | 6144 B | 16 x 41 | 16 / 4 | 34.6 MB | 3.8 MB | 9.11 | false |

#### DBs of "combined + smaller batches"

| DB | rows | entry | partitions x rows | chunk / set | hints | DB | hints/DB | downloaded |
|---|---|---|---|---|---|---|---|---|
| s1/level0 | 73273 | 16 B | 16 x 4580 | 128 / 36 | 1.6 MB | 1.1 MB | 1.40 | true |
| s1/level1 | 295280 | 16 B | 16 x 18455 | 256 / 76 | 3.2 MB | 4.5 MB | 0.71 | false |
| s1/level2 | 593938 | 16 B | 16 x 37122 | 256 / 148 | 4.5 MB | 9.1 MB | 0.49 | false |
| s2/cuckoo | 953168 | 16 B | 16 x 59573 | 256 / 236 | 4.6 MB | 14.5 MB | 0.32 | false |
| s3/embeddings | 648 | 1536 B | 8 x 81 | 16 / 8 | 4.3 MB | 972 KB | 4.58 | true |

## Bins, scifact: 5183 docs, bs=5000, dpb=100, dim 192

### Bins client storage

| layout | client | vs first | id | vec | all | server DB | online/search | refresh/search | note |
|---|---|---|---|---|---|---|---|---|---|
| vec1 (one DB) | 1.2 GB | 1.00x | 0 B | 0 B | 1.2 GB | 366 MB | 2.1 MB | 8.5 MB | ids and embeddings in each bin |
| vec0 (ids, then embeddings) | 44.8 MB | 0.04x | 9.1 MB | 35.7 MB | 0 B | 5.7 MB | 1.1 MB | 2.4 MB | Vec DB batch 100 |
| vec0 + download if cheaper | 5.7 MB | 0.00x | 1.9 MB | 3.8 MB | 0 B | 5.7 MB | 0 B | 0 B |  |
| vec0 + fp16 | 27.2 MB | 0.02x | 9.1 MB | 18.1 MB | 0 B | 3.8 MB | 562 KB | 1.2 MB | Vec DB embeddings as fp16 |
| vec0 + int8 | 18.4 MB | 0.01x | 9.1 MB | 9.3 MB | 0 B | 2.9 MB | 300 KB | 653 KB | Vec DB embeddings as int8 |
| vec0 + Vec batch 25 | 26.2 MB | 0.02x | 9.1 MB | 17.1 MB | 0 B | 5.7 MB | 821 KB | 2.7 MB | fewer partitions; more Piano queries land in each, more sequential work |
| vec0 + Vec batch 6 | 17.6 MB | 0.01x | 9.1 MB | 8.6 MB | 0 B | 5.7 MB | 732 KB | 3.7 MB | fewer partitions; more Piano queries land in each, more sequential work |

#### DBs of "vec0 (ids, then embeddings)"

| DB | rows | entry | partitions x rows | chunk / set | hints | DB | hints/DB | downloaded |
|---|---|---|---|---|---|---|---|---|
| id/bins | 5000 | 400 B | 12 x 417 | 64 / 8 | 9.1 MB | 1.9 MB | 4.75 | false |
| vec/embeddings | 5183 | 768 B | 50 x 104 | 32 / 4 | 35.7 MB | 3.8 MB | 9.41 | false |
