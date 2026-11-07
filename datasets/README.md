# Extra datasets (ICLR rebuttal)

Reviewers asked for more, and larger, datasets. These are the ones wired in, all standard RAG / open-domain QA
benchmarks with qrels, so MRR and Recall work exactly like MS MARCO and SciFact:

| `-name`            | Docs        | Test queries | Why                                                           |
|--------------------|-------------|--------------|---------------------------------------------------------------|
| `nq`               | 2,681,468   | 3,452        | Natural Questions (BEIR): *the* open-domain QA / RAG benchmark |
| `hotpotqa`         | 5,233,329   | 7,405        | Multi-hop QA over Wikipedia (BEIR), common RAG eval            |
| `fever`            | 5,416,568   | 6,666        | Fact verification over Wikipedia (BEIR), optional 4th          |
| `msmarco-v2`       | 138,364,198 | 3,903 (dev1) | MS MARCO v2 passages: bigger than PACMANN's SIFT100M           |
| `msmarco-v2-100m`  | 100,000,000 | 3,903        | Subsample of the above, matches SIFT100M exactly               |
| `msmarco-v2-25m/50m` | 25M / 50M | 3,903        | Subsamples for a scaling plot                                  |

With SciFact (5K) and MS MARCO v1 (8.8M) that gives sizes from 5K to 138M. Subsamples keep every doc that is
relevant to some query and fill the rest at random (seed 42), so the scores stay comparable across sizes.

## Pipeline (run from the repo root)

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
BINS_DATASETS="nq hotpotqa fever" bash run_bins_high.slurm --count
BINS_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_bins_high.slurm      # and run_bins_low.slurm
PACMANN_DATASETS="nq hotpotqa fever" sbatch --array=0-<N-1> run_pacmann.slurm

# 5. tree: build artifacts per (B, r, s), then run the grid
bash datasets/prep_tree.sh nq 32 8 4
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
- The tree's `build_small_test_artifacts.py` is plain Python. Expect hours for the 5M sets. For msmarco-v2 use the
  collaborator's optimised `tree/prepare_data.py` / `tree/prepare_data_stage2.py`.
