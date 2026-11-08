# Extra datasets (ICLR rebuttal)

Reviewers asked for more, and larger, datasets. These are the ones wired in, all standard RAG / open-domain QA
benchmarks with qrels, so MRR and Recall work exactly like MS MARCO and SciFact:

Average words per query / per doc are from the BEIR paper (Thakur et al. 2021, Table 1). For reference, MS MARCO is
5.96 / 55.98 and SciFact is 12.37 / 213.63. Run `python datasets/query_stats.py <dataset dir>` for the full
distribution (% of queries with at most 10/15/20 words); `prepare_dataset.py` also writes it to `stats.json`.

| `-name`            | Docs        | Queries      | Words/query | Words/doc | Use?                                         |
|--------------------|-------------|--------------|-------------|-----------|----------------------------------------------|
| `dbpedia-entity`   | 4,635,922   | 400          | 5.39        | 49.68     | Yes: shortest queries and docs of the lot    |
| `nq`               | 2,681,468   | 3,452        | 9.16        | 78.88     | Yes: *the* open-domain QA / RAG benchmark    |
| `fever`            | 5,416,568   | 6,666        | 8.13        | 84.76     | Optional                                     |
| `hotpotqa`         | 5,233,329   | 7,405        | 17.61       | 46.30     | No: multi-hop queries are too long           |
| `miracl-en`        | 32,893,221  | 799 (dev)    | short questions (not measured) | Wikipedia passages | Optional: the only judged set between 8.8M and 138M |
| `msmarco-v2`       | 138,364,198 | 3,903 (dev1) | same style as MS MARCO v1 | ~same | Yes: bigger than PACMANN's SIFT100M |
| `msmarco-v2-100m`  | 100,000,000 | 3,903        |             |           | Subsample, matches SIFT100M exactly          |
| `msmarco-v2-25m/50m` | 25M / 50M | 3,903        |             |           | Subsamples for a scaling plot                |

With SciFact (5K) and MS MARCO v1 (8.8M) that gives sizes from 5K to 138M. Subsamples keep every doc that is
relevant to some query and fill the rest at random (seed 42), so the scores stay comparable across sizes.

## Where the datasets live

| Where | Folder | Set by |
|---|---|---|
| Server (slurm scripts, run from the scratch copy of the repo) | `../datasets` | `DATASETS_ROOT` |
| Your machine (`run_local.sh`, `embed_local.sh`) | `../../../datasets` | `DATASETS_ROOT` |

Each dataset is a subfolder `<name>/` (corpus.jsonl, queries.jsonl, qrels/, num_docs.txt, embeddings); raw
downloads go to `raw/` under the same folder (`RAW_ROOT`).

## One command

```bash
bash build.sh                                   # pir_app
bash datasets/run_all.sh nq hotpotqa fever      # prep -> embed -> tree artifacts -> bins/pacmann/tree grids
bash datasets/run_all.sh msmarco-v2-all         # 138M, then the 25m/50m/100m subsamples after it's embedded
bash datasets/build_all.sh nq fever             # only prep -> embed -> tree artifacts, no grids (no pir_app needed)
DRY_RUN=1 bash datasets/run_all.sh nq           # show the sbatch commands without submitting
STAGES="bins pacmann" bash datasets/run_all.sh nq   # only some stages
```

Every job is queued straight away with `--dependency=afterok` on the jobs it needs. `DATASETS_ROOT` (default
`../datasets`, i.e. running from the scratch copy of the repo) is used for every stage.

## Locally (no SLURM)

```bash
bash datasets/run_local.sh nq dbpedia-entity     # prep -> embed on your GPU -> tree artifacts -> one config per method
```
Uses `../../../datasets` by default (`DATASETS_ROOT=...` to change). See the header of the script for options.

## Pipeline step by step (run from the repo root)

```bash
# 1. download + convert -> ../datasets/<name>/{corpus.jsonl, queries.jsonl, qrels/test.tsv, num_docs.txt}
DATASET=nq sbatch datasets/prep_datasets.slurm          # also hotpotqa, fever, msmarco-v2
#    (no internet on compute nodes? run `bash datasets/download.sh <name> ../datasets/raw` on the login node first)

# 2. e5-base-v2 + PCA192 embeddings (same model as msmarco/scifact, but a PCA per dataset)
DATASET=nq sbatch datasets/embed.slurm                   # one GPU job for the 2-5M sets
bash datasets/embed_big.sh                               # msmarco-v2: PCA -> 24 sharded GPU jobs -> merge + queries

# 3. subsamples of MS MARCO v2 (copies the embedding rows, no re-encoding)
DATASET=msmarco-v2-100m sbatch datasets/prep_datasets.slurm

# 4. run: the bins / pacmann slurms take a dataset list and print the array size to use
DATASET=nq sbatch datasets/bins_index.slurm             # bins only: Bluge index (= ./pir_app -t index -name nq)
BINS_DATASETS="nq hotpotqa fever" bash run_bins_high.slurm --count
BINS_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_bins_high.slurm      # and run_bins_low.slurm
PACMANN_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_pacmann.slurm

# 5. tree: Lucene index + every (B, r, s) of the grid in one pass (tree/build_tree_artifacts.py), then run the grid
DATASET=nq sbatch datasets/tree_artifacts.slurm
DATASET=nq bash run_tree_pir_old.slurm --count
DATASET=nq sbatch --array=0-<N-1> run_tree_pir_old.slurm
```

`pir_app -name <name>` picks up any dataset laid out this way (see `GetDatasets` in `main.go`), and `-n` comes from
`num_docs.txt`.

## Racing two GPU queues (embed jobs stuck in Priority)

`embed.slurm` asks for a DGX `A100.40gb`: only 16 GPUs on 2 nodes, often held by multi-day jobs. `GPU=<profile>`
queues the same embed (+ tree-db) chain on another pool with fp16, 32 GB RAM and 2-10 h limits so it backfills
sooner. Run it next to the series that's already queued:

```bash
git pull                                    # in the dir you submitted from: the queued jobs pick up the new python too
sgpu                                        # which A100.80gb / MIG slices are free right now
GPU=a100 STAGES="embed tree-db" bash datasets/build_all.sh nq hotpotqa fever msmarco-v2-all
squeue --start -u $USER                     # slurm's estimated start times
```

| `GPU=` | pool | preemptible | ~speed vs RTX 5060 Ti (fp32) |
|---|---|---|---|
| `any` | any GPU on gpuq / contrib-gpuq / contrib-h100, MIG included, 4 h | on contrib | 0.7-10x |
| `a100` (pick this), 2 h | A100.80gb on gpuq **or** contrib-gpuq, 96 GPUs | on contrib (jobs requeue and resume) | ~5x |
| `a100-gpuq` | A100.80gb on gpuq, 40 | no | ~5x |
| `dgx` | A100.40gb on gpuq, 16 (same pool as embed.slurm) | no | ~5x |
| `mig40` | MIG 3g.40gb on gpuq, 24 | no | ~2x |
| `mig20` | MIG 2g.20gb on gpuq, 8 | no | ~1.3x |
| `h100` | H100.80gb on contrib-h100, 4 | yes | ~8x |
| `b200` | B200.180gb on contrib-b200, 8 (reserved) | yes | ~10x |

Override any piece with `GPU_PARTITION`, `GPU_GRES`, `GPU_QOS`, `GPU_HOURS`. Speeds are estimates from peak specs
(fp16 tensor cores; H100/B200 are capped by CPU tokenisation), check the tqdm rate in the first log.

**Any GPU.** `GPU=any` asks for `--gres=gpu:1` with no type on gpuq, contrib-gpuq and contrib-h100, so slurm takes
the first GPU of any kind, MIG slices included, with a 4 h limit. Every embed job now also requeues itself 5 min
before its limit and resumes from its checkpoint (`REQUEUE_ON_TIMEOUT=0` turns that off), so landing on a slow
1g.10gb slice costs a requeue, not the work. `a100` is still the better bet: 2 h is about 2x what any of
nq/hotpotqa/fever or one msmarco-v2 shard needs on an A100 in fp16.

**Who wins.** The first job to finish an output creates `<output>.done` and moves its copy into place; any other job
for the same output throws its copy away and exits 3, so its tree-db / prep jobs never start (they sit in
`DependencyNeverSatisfied`). Shards and the PCA are shared between series (one PCA per dataset, reused if it
exists), so a shard finished by either series counts. When one series has won, clean up the other:

```bash
scancel -n emb_nq            # or emb_nq_a100 etc.: cancel the loser's embed jobs
squeue -u $USER -h -t PD -o "%i %r" | awk '$2=="DependencyNeverSatisfied"{print $1}' | xargs -r scancel
```

**Resume.** Embeddings go to `<output>.partial-<tag>.npy` with a checkpoint every ~100k docs. A preempted (requeued)
job, or one resubmitted with the same `GPU=`, carries on from the checkpoint. `OVERWRITE=1` redoes a finished output.
All embed jobs now also use TF32 for fp32 matmuls (`EMBED_TF32=0` turns it off): ~3x faster on A100 than plain fp32.

## Embedding locally (float16), importing on the server

Embed on your own GPU, copy the float16 files up, widen them to float32 on the server. That's a third racer
next to the slurm series: whichever finishes first wins.

```bash
# your machine, from the repo root (uses ../../../datasets)
bash datasets/embed_local.sh nq hotpotqa fever dbpedia-entity    # download + prepare + embed, prints an rsync line
bash datasets/embed_local.sh msmarco-v2                          # the long one, see the times below
# copy each dataset's files to <server>/datasets/<name>/local_import/ with the printed rsync line, then on the server
bash datasets/import_local_embeds.sh nq hotpotqa fever msmarco-v2   # or sbatch it; skips anything already done
STAGES="embed tree-db" bash datasets/build_all.sh nq hotpotqa fever msmarco-v2-all   # done embeds aren't requeued
```

* **Streaming.** Neither the corpus nor the embeddings are ever held in RAM. A CPU thread reads, parses and
  tokenizes the next 8 batches of 2048 docs (`EMBED_PREFETCH`) while the GPU embeds the current one, and rows go
  straight into a memory-mapped `.npy`. The machine needs ~3 GB of RAM and ~2 GB of VRAM at `ENCODE_BATCH=128`.
* **Resume.** Re-run the same command after a crash, reboot or Ctrl-C. It checkpoints about every 100k docs, and
  on restart it skips the finished lines without parsing them.
* **float16 to float32** is exact (every fp16 value is a fp32 value). fp16 rounding moves a vector by ~1e-4.
  The import checks every id against the server's corpus.jsonl before replacing anything. prepare_dataset.py is
  deterministic, so the same download gives the same corpus on both machines.
* **Only part of msmarco-v2.** Copy the server's `msmarco-v2/pca_e5_768_to_192.faiss` to your dataset folder,
  then `NUM_SHARDS=24 SHARDS="18 19 20 21 22 23" bash datasets/embed_local.sh msmarco-v2`. The import turns
  those into finished server shards, so the server's shard jobs skip them.

Disk on your machine (fp16, raw downloads kept, the .tar/.zip deleted after extraction):

| dataset | raw | corpus.jsonl | embeddings fp16 (fp32) | total |
|---|---|---|---|---|
| nq | ~1.5 GB | ~1.5 GB | 1.0 GB (2.1) | ~4 GB |
| hotpotqa | ~2 GB | ~2 GB | 2.0 GB (4.0) | ~6 GB |
| fever | ~3 GB | ~3.2 GB | 2.1 GB (4.2) | ~8.5 GB |
| dbpedia-entity | ~2 GB | ~1.9 GB | 1.8 GB (3.6) | ~6 GB |
| msmarco-v2 | 20 GB | ~55-60 GB | 53 GB (106) | ~130 GB |
| **all** | | | **60 GB (120)** | **~155 GB** |

To get back ~30 GB, delete `raw/` once a dataset is prepared. In fp32, msmarco-v2 alone would push the total
past 200 GB.

Time on an RTX 5060 Ti 16 GB. These are estimates scaled from NQ taking 2 h in fp32, and fp16 is about 2x on that card:

| | nq | hotpotqa | fever | dbpedia | msmarco-v2 |
|---|---|---|---|---|---|
| fp16 embed | ~1 h | ~1.2 h | ~2.2 h | ~1.1 h | ~35-40 h (~1.5 h per 1/24 shard) |
| download + prepare | minutes | minutes | minutes | minutes | 20 GB download + ~2-3 h |

So all four small sets take ~6 h. msmarco-v2 doesn't fit in 12 h locally; it's quicker as shards on the server
A100s, or split between both.

## Notes

- Doc IDs are renumbered `0..N-1` (`docid_map.tsv` keeps the originals). This keeps the PIR ID entries short and
  fixed width, and the tree's `docmap.bin` needs integer IDs.
- Each dataset gets its own `pca_e5_768_to_192.faiss`. The old scripts wrote one PCA to the working directory, so
  embedding a second dataset silently changed the PCA the first one's queries used.
- MS MARCO v2 at 192 float32 dims is ~106 GB of vectors. `LoadFloat32MatrixFromNpy` no longer copies every row, and
  the ID maps no longer load the whole corpus text, so peak memory per copy roughly halved.
- `fever` and `trec-covid` are wired in too, but are optional.
- Tree artifacts come from `tree/build_tree_artifacts.py`. It gives byte-identical output to
  `build_small_test_artifacts.py`, but reads each term's postings once for every (B, r, s), splits terms over
  processes, and never allocates a per-term array of size N/B. The collaborator's `prepare_data.py` /
  `prepare_data_stage2.py` can't be used as they are: the `--vocab` path crashes (`getTerm` on a str), the vocab
  JSON write is commented out, and stage 2 reads a hard-coded `data/layout_with_stats.json` from a missing
  `gen_layout.py`.
