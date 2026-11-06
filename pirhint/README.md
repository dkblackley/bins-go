# Client storage (the Piano hint): where it goes and how to cut it

`pirhint` computes the client storage of any PIR DB layout with the exact
formulas `pir-old` (tree) and `pianopir` (bins) allocate with
(`pirhint_test.go` checks both, byte for byte). `cmd/hintbench` uses it to
compare layouts at full MS MARCO scale in about a second, and `-run` builds
every layout for real on random data to measure it.

```bash
go run ./pirhint/cmd/hintbench -dataset msmarco                       # synthetic sizes
go run ./pirhint/cmd/hintbench -dataset scifact -B 16 -r 8 -s 4
go run ./pirhint/cmd/hintbench -dataset msmarco -B 32 -r 32 -s 8 \
    -stage1-idmap data/msmarco/pir/B32_r32_s8/stage1_idmap.bin \
    -stage2-idmap data/msmarco/pir/B32_r32_s8/stage2/idmap.bin       # exact sizes from your artifacts
go run ./pirhint/cmd/hintbench -dataset msmarco -run -run-docs 100000 # real pir-old, measured
```

Synthetic sizes assume random (unclustered) blocks over a Zipf vocabulary, so
they overestimate Stage1/Stage2 (clustering shares terms within a block). The
current MS MARCO estimate is 5.9 GB against the paper's "over 4.2 GB", so the
relative numbers are the point. Run it with `-stage1-idmap/-stage2-idmap`
on the server for the exact ones. `run_tree_hint_ablation.slurm` does that for
every knob.

Saved outputs: [`results/msmarco.md`](results/msmarco.md),
[`results/scifact.md`](results/scifact.md),
[`results/msmarco_run100k.md`](results/msmarco_run100k.md) (measured).

## Why the hint is big

Per partition of `P` rows with `E`-byte entries, Piano keeps about
`15*C + 6*sqrt(P)*ln(P)` entries, where `C` is about `2.8*sqrt(P)`. Two things
follow from that:

1. **A small DB costs more as hints than as data.** The hint is larger than the
   partition once `P` drops below about 10,000 rows. SciFact's tree Stage3 has
   41 rows per partition, so its hints are **9.1x** the DB. Every SciFact DB is
   over 1x, so the whole SciFact tree DB (33 MB) is smaller than its hints (88 MB).
2. **Splitting a DB makes it worse, not better.** `k` partitions of an `N`-row
   DB hold about `sqrt(k*N)*E`. Splitting one DB into two doubles `k`, so it
   costs `sqrt(2)` more. Merging does the opposite. The `SimpleBatchPianoPIR`
   batch already splits every DB into `batch/2` partitions, which is where most
   of the hint comes from.

## Your two tree ideas, measured (MS MARCO, synthetic, B=32 r=32 s=8)

| layout | client | vs now |
|---|---|---|
| now: one Cuckoo DB per Stage1 level, one Stage2 DB | 5.9 GB | 1.00x |
| Round 1: one table per Cuckoo hash function per level, same batch | 6.7 GB | 1.14x |
| Round 1: the same with half the batch each | 5.9 GB | 1.00x |
| Round 1: **all levels in one DB** (the opposite) | 5.0 GB | 0.85x |
| Stage 2: one DB per sub-block | 7.8 GB | 1.33x |
| Stage 2: one DB per leaf block | 33.7 GB | 5.72x |

Round 1 already has one DB per level (`Stage1LevelDB`), each a two-hash Cuckoo
table. Splitting the two hash positions into two tables helps nothing: each
table must still be probed for every key, so the batch is the same and the
number of partitions only goes up. Per-sub-block and per-leaf DBs have to be
probed in full on every query (otherwise the server learns which sub-block or
leaf), and every extra DB brings its own fixed-size hint. Merging all Stage1
levels into one DB does save 15%, but each probe then pays for the bigger
partition (online 18.6 to 28.4 MB a search), so I wouldn't lead with it.

## Fixes that do work (all added as flags, defaults unchanged)

MS MARCO tree, synthetic sizes, each fix applied on its own:

| fix | flag | client | trade-off |
|---|---|---|---|
| Stage3 embeddings int8 | `--doc-embed *_int8.npy` | 3.6 GB (0.61x) | cosine changes by about 5e-4 |
| Stage3 embeddings fp16 | `--doc-embed *_float16.npy` | 4.3 GB (0.74x) | none measurable |
| Stage1/2 batch 32 | `--pir-batch-size 32` | 4.6 GB (0.78x) | more queries per partition, online 18.6 to 26.2 MB |
| Stage3 batch 8 | `--stage3-batch-size 8` | 4.4 GB (0.75x) | same |
| smaller Piano chunks | `--pir-chunk-factor 1` | 4.7 GB (0.79x) | uploads twice as large, online 35.5 MB |
| refresh hints twice as often | `--pir-query-budget 0.5` | 4.1 GB (0.70x) | offline re-download per search doubles |
| download a DB smaller than its hints | `--pir-download-small` | no change on MS MARCO | **SciFact: 88 MB to 33 MB** |
| all of the above | | **1.0 GB (0.17x)** | online 18.6 to 49.8 MB a search |

SciFact tree: 88 MB now, **14 MB** with everything. SciFact bins (vec0, dpb
100): 45 MB now, **5.7 MB** with `--pir-download-small` alone.

What each one does:

* **`tree/tools/quantize_embeddings.py`** writes an int8 (per-vector absmax)
  or fp16 copy of the Stage3 embeddings. The reranker now reads 1- and 2-byte
  `.npy` dtypes. Cosine ignores a per-vector scale, so int8 needs no scale
  stored. Stage3 (8 docs x 192 dims x 4 B = 6 KB rows) is 3.1 of the 5.9 GB, and
  it is the same for every B/r/s, which is part of why the tree configs line
  up on the client-storage axis. The same applies to the bins Vec DB (vec0
  dpb 100: 3.3 GB to 1.0 GB at int8), but that one is not wired in yet.
* **`--pir-batch-size` / `--stage3-batch-size`** set the partition count
  (`batch/2`). pir-old answers any number of lookups per call, so a smaller
  batch puts more sequential Piano queries in each partition instead of failing.
* **`--pir-chunk-factor f`** sets `ChunkSize = next pow2 >= f*sqrt(P)` (2 now).
  The primary hints are `15*ChunkSize`, so they shrink with `f`. The upload,
  `SetSize = P/ChunkSize`, grows by the same factor. The power-of-two rounding
  alone costs up to 2x today (Stage3: `2*sqrt(69077) = 526` rounds up to 1024).
* **`--pir-query-budget s`** scales the queries a hint table serves before
  it is rebuilt (`sqrt(P)*ln(P)` now). The backup hints are proportional to it.
  The cost is a full re-stream of the DB `1/s` times as often (the
  "refresh/search" column in hintbench).
* **`--pir-download-small`** (pir-old and pianopir): when a DB's hints would be
  at least its size, the client downloads the DB and looks rows up locally.
  The choice uses only public sizes and nothing is sent online, so it is
  trivially private. This also answers the reviewer's "the client could just
  download the index" literally, for the DBs where that is true.

## Further ideas (not implemented)

* **A stateless PIR for Stage3/the Vec DB.** It fetches a few hundred rows a
  query from a big-entry DB, which is the worst case for Piano's hint. A
  hint-free single-server scheme such as YPIR or Respire, or a SimplePIR-style
  scheme, moves that cost to server compute. PILLAR does not depend on Piano,
  which is worth saying in the rebuttal.
* **Shorter Cuckoo keys.** Every Stage1/2 row carries an 8-byte key for 4 to 32
  bytes of scores (Stage2 at s=8: 8 key + 4 value, padded to 16). A 2 to 4 byte
  fingerprint plus byte-granular rows would cut Stage1/2 by up to 2x. That
  needs non-word-aligned PIR rows.
* **Fewer, fatter Stage3 rows** (16 docs a row) halves the rows but doubles the
  entry, so the hint only grows by `sqrt(2)`. It is only worth it together with
  int8.
* **PCA below 192 dims** for Stage3 is linear in the hint, like int8.

## Bins: the new plot

`plotting/plot_bins_client_storage_split.py` draws client storage against hash
table size and against docs per bin, a dataset per column in the style of
`plot_bins_ablation_split.py`. It has a line for the single DB (vec1), for the
split layout's total (vec0), and for each of vec0's two hints: the ID DB alone
(the non-embedding DB) and the embedding DB alone. It reads
`client_storage_mb_bm25` and `client_storage_mb_vec`, which `load_results.py`
already produces from existing runs. Enable it in `plotting/main.py`.
