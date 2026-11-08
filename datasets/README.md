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
#    the bins grid includes the -text layout (bins/TEXT.md); BINS_VECS="true false" leaves it out, BINS_VECS=text runs only it
PACMANN_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_pacmann.slurm

# 5. tree: Lucene index + every (B, r, s) of the grid in one pass (tree/build_tree_artifacts.py), then run the grid
DATASET=nq sbatch datasets/tree_artifacts.slurm
DATASET=nq bash run_tree_pir_old.slurm --count
DATASET=nq sbatch --array=0-<N-1> run_tree_pir_old.slurm
```

`pir_app -name <name>` picks up any dataset laid out this way (see `GetDatasets` in `main.go`), and `-n` comes from
`num_docs.txt`.

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
