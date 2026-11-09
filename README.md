# bins-go

PIR retrieval methods for PILLAR, plus the baselines they are compared against. One Go binary, `pir_app`, runs
every method:

| `-t` | Method | What the server's PIR DB holds |
|---|---|---|
| `bins` | PILLAR-Bin | per-keyword bins of doc embeddings (`-vectors=true`, "vec1"), or an ID DB + an embedding DB (`-vectors=false`, "vec0"), or compressed doc text (`-text`, **unmerged PR #8**) |
| `tree` | PILLAR-Tree | BM25 upper-bound tree (Stage 1/2 Cuckoo tables) + Stage 3 embeddings, on the vendored old PIR (`pir-old/`) |
| `pacmann` | PACMANN baseline | NGT graph + embeddings |
| `tiptoe` | Tiptoe baseline | real upstream protocol, only in `pir_app_tiptoe` (built with `-tags tiptoe`) |
| `index` | (not a method) | builds the Bluge keyword index that `bins` needs, then exits |

Coeus and the non-private baselines are Python scripts (sections 4.5 and 6).

Everything is run from the repo root. The quickest path to a full run of a new dataset on the cluster is:

```bash
bash setup_pacmann.sh                         # once: NGT + hnsw native libs
bash build.sh                                 # pir_app   (careful, see 2.2)
bash datasets/run_all.sh nq dbpedia-entity    # download -> embed -> index/artifacts -> bins/pacmann/tree grids
```

and locally, on scifact:

```bash
bash run_scifact_local.sh
```

> **What is merged and what isn't.** Everything below is on `main` except the bins text mode (`-text`,
> `BINS_VECS`, `bins_text_*` result folders, `bins/cmd/textbench`) and the unpadded PIR byte/storage accounting,
> which are in **PR #8** (`claude/project-thread-ur6tb7`, still open). Sections that depend on it are marked
> **[PR #8]**.

> **Baselines: always use the real (non-simulated) versions.** For Tiptoe that's `MODE=real` (`./pir_app_tiptoe`,
> built by `setup_baselines.sh tiptoe`); for Coeus it's `SERVER_TIME=1` (the artifact's `opt2` kernel, built by
> `setup_baselines.sh coeus`). The slurm scripts still **default to the simulated/quality-only paths**
> (`run_tiptoe.slurm` defaults to `MODE=sim`, `run_tiptoe_msmarco_v2.slurm` is simulation-only, `run_coeus.slurm`
> defaults to `SERVER_TIME=0`), so every command you write or are given must set `MODE=real` / `SERVER_TIME=1`
> explicitly. Don't use the simulator, `simplepir_bench` or `run_tiptoe_msmarco_v2.slurm` for results unless you
> have been explicitly asked to. Agents: never hand out or change a Tiptoe/Coeus command that would use them.

---

## Contents

1. [Requirements](#1-requirements)
2. [Building](#2-building)
3. [Datasets: download, convert, embed](#3-datasets-download-convert-embed)
4. [Per-method setup](#4-per-method-setup)
5. [Running on the cluster (slurm)](#5-running-on-the-cluster-slurm)
6. [Running locally](#6-running-locally)
7. [Where results go, and plotting](#7-where-results-go-and-plotting)
8. [Sizes and memory](#8-sizes-and-memory)
9. [Gotchas](#9-gotchas)
10. [Map of the repo](#10-map-of-the-repo)

---

## 1. Requirements

| What | Needed for | Notes |
|---|---|---|
| Go >= 1.24 | everything (`go.mod`) | `go_env.sh` finds one: `$GOROOT`, a cached toolchain, `/usr/local/go`, or `go` on PATH |
| gcc + cmake + make | NGT/hnsw libs (`setup_pacmann.sh`) | Hopper: `module load gnu9/9.3.0` (done by `go_env.sh`) |
| NGT + libhnsw | **every** `pir_app` build (PACMANN is cgo-linked in) | `bash setup_pacmann.sh` installs them to `~/opt/ngt-amd`, `~/opt/hnsw` |
| Python 3.10 + `requirements.txt` | dataset prep, embeddings, tree artifacts, baselines, plots | plus `faiss` (`faiss-gpu` or `faiss-cpu`) for the PCA. On the cluster the jobs activate conda env `main` via `datasets/env.sh` (`CONDA_BASE`, `CONDA_ENV` to change) |
| Java 21 | tree artifacts (pyserini/Lucene), `baselines.py` BM25 | `conda install -c conda-forge openjdk=21`; `prep_tree.sh` refuses older versions |
| A GPU | embeddings (e5-base-v2) | cluster: `embed.slurm` asks for one A100 40GB. Locally a 16 GB card is plenty (`FP16=1` halves the time) |
| `huggingface-cli` | only `miracl-en` download | |
| SEAL 4.1.1 + GCC >= 10 | only **real** Tiptoe | `baselines/setup_baselines.sh tiptoe` (Hopper: `module load gnu12`) |
| SEAL 3.5.6 | only Coeus server time | `baselines/setup_baselines.sh coeus` |

Install the Python side once:

```bash
conda create -n main python=3.10 && conda activate main
pip install -r requirements.txt faiss-cpu      # or faiss-gpu on a GPU box
conda install -c conda-forge openjdk=21
```

Compute nodes without internet: run the downloads (`datasets/download.sh`) and `go mod download` on the login node
first.

---

## 2. Building

### 2.1 Native libs (once)

```bash
bash setup_pacmann.sh
```

Builds NGT (`yahoojapan/NGT`, QBG off) into `~/opt/ngt-amd` and `libhnsw.so` (from the `hnswgo` version pinned in
`go.mod`) into `~/opt/hnsw`, no root needed. It writes `~/opt/pacmann_env.sh` with the CGO flags. Override the
locations with `NGT_PREFIX` / `HNSW_PREFIX`.

### 2.2 `pir_app`

```bash
bash build.sh                 # or: sbatch build_go.slurm (16G, 30 min)
```

`build.sh` sources `go_env.sh` (modules, CGO flags, Go toolchain), runs `go mod tidy`, builds `./pir_app` and
checks it links `libngt`/`libhnsw`.

**Warning:** `build.sh` also deletes `../results/bins_*`, `../results/pacmann_*`, the cached PACMANN graphs
(`../datasets/{scifact,msmarco,trec-covid}/*.ngt`, `*graph_aux.txt`) and `logs/Bins_*`, `logs/Pacmann_*`. If you
only want to rebuild, run the two lines yourself:

```bash
source go_env.sh && go build -o pir_app ./main.go
```

### 2.3 Optional extra binaries

```bash
./baselines/setup_baselines.sh sim       # ./simplepir_bench, for simulated Tiptoe (no SEAL)
./baselines/setup_baselines.sh tiptoe    # SEAL 4.1.1 in ~/opt/seal-4.1 + ./pir_app_tiptoe (real Tiptoe)
./baselines/setup_baselines.sh coeus     # SEAL 3.5.6 + Coeus' matmult/bin/opt2 (Coeus server time)
./baselines/setup_baselines.sh all
```

A plain `pir_app` exits with a clear message if you pass `-t tiptoe`.

---

## 3. Datasets: download, convert, embed

### 3.1 Which datasets

| `-name` | Docs | Queries | Words/query | Status |
|---|---|---|---|---|
| `scifact` | 5,183 | 300 | 12.4 | original; small, use it locally |
| `msmarco` | 8,841,823 | 6,980 (dev small) | 6.0 | original |
| `nq` | 2,681,468 | 3,452 | 9.2 | new |
| `dbpedia-entity` | 4,635,922 | 400 | 5.4 | new, shortest queries |
| `fever` | 5,416,568 | 6,666 | 8.1 | new, optional |
| `miracl-en` | 32,893,221 | 799 | short | new, optional, not verified end to end |
| `msmarco-v2` | 138,364,198 | 3,903 (dev1) | ~6 | new |
| `msmarco-v2-25m/50m/100m` | 25M / 50M / 100M | 3,903 | ~6 | subsamples of v2 (all relevant docs kept + random fill, seed 42) |
| `hotpotqa`, `trec-covid` | 5.2M / 171K | | 17.6 / | wired in but not used (hotpotqa queries are too long) |

`python datasets/query_stats.py <dataset dir>` prints the query-length distribution. More detail is in
[`datasets/README.md`](datasets/README.md).

### 3.2 Layout on disk

Everything lives under one `DATASETS_ROOT` (the same folder you pass to `pir_app -dataset`):

```
<DATASETS_ROOT>/
  raw/<name>/                         downloads (datasets/download.sh)
  <name>/corpus.jsonl                 doc IDs renumbered 0..N-1 (docid_map.tsv keeps the originals)
  <name>/queries.jsonl, qrels/test.tsv, num_docs.txt, stats.json
  <name>/pca_e5_768_to_192.faiss      one PCA per dataset
  <name>/<name>_corpus_embed_f32.npy  N x 192 float32 (+ .ids)
  <name>/<name>_query_embed_f32.npy
  index_<name>/                       Bluge index for bins (pir_app -t index)
  tiptoe/<name>_dpc*_m*_b*/           Tiptoe preprocessed DBs
  results/                            local runs (datasets/run_local.sh)
```

`msmarco`, `scifact` and `trec-covid` keep their original file names (see `GetDatasets` in `main.go`:
`index_marco`, `msmarco/queries.dev.small.jsonl`, `trec_*_embed_f32.npy`, ...). Any other name laid out as above
works with no code change.

Default roots: `datasets/run_all.sh` and the dataset slurms use `../datasets` (i.e. you run from a copy of the repo
in scratch). The run slurms on their own default to `DATASET_DIR=../../datasets`, but `run_all.sh` pins both.
`run_local.sh` and `run_scifact_local.sh` use `../../../datasets`.

### 3.3 The pipeline (cluster)

| Step | Script | Resources | Output |
|---|---|---|---|
| download | `datasets/download.sh <name> <raw dir>` (login node if compute nodes have no internet) | network; msmarco-v2 is a ~20 GB tar | `raw/<name>/` |
| convert | `DATASET=nq sbatch datasets/prep_datasets.slurm` (calls download.sh if needed, then `prepare_dataset.py`) | 64G, 2 CPUs, up to 1 day (msmarco-v2 ~2-4 h) | `<name>/corpus.jsonl`, queries, qrels, `num_docs.txt` |
| embed (≤ 6M docs) | `DATASET=nq sbatch datasets/embed.slurm` | 1x A100 40GB, 100G RAM, 12 h; ~2-6 h for NQ/FEVER | PCA + corpus/query `.npy` |
| embed (msmarco-v2, miracl-en) | `bash datasets/embed_big.sh` (or `DATASET=miracl-en NUM_SHARDS=8 bash datasets/embed_big.sh`) | PCA job, then 24 GPU shard jobs (~1-2 h each, fp16), then a merge | ~106 GB of float32 for v2; shards need the same again until the merge deletes them |
| subsamples | `DATASET=msmarco-v2-100m sbatch datasets/prep_datasets.slurm` (also -25m, -50m), **after** v2 is embedded | | copies v2's embedding rows, no re-encoding |
| bins index | `DATASET=nq sbatch datasets/bins_index.slurm` (= `./pir_app -t index`) | 64G, 8 CPUs (run_all gives v2/miracl 500G on bigmem) | `index_<name>/` |
| tree artifacts | `DATASET=nq sbatch datasets/tree_artifacts.slurm` | 120G, 16 CPUs, up to 2 days; v2: `WORKERS=12 sbatch --partition=bigmem --mem=1000G ...` | `data/<name>_tree/` (see 4.2) |

`datasets/run_all.sh` chains all of these with `--dependency=afterok` (section 5). `datasets/build_all.sh nq fever`
does only prep -> embed -> tree artifacts, without any grid and without needing `pir_app`.

The embeddings are e5-base-v2 + a per-dataset PCA to 192 dims, the same model as msmarco/scifact. fp16 is only used
for the sharded (100M+) corpora.

### 3.4 Getting scifact and msmarco (the originals)

These predate the pipeline above and are not renumbered.

* **scifact**: `bash datasets/download.sh scifact ../../../datasets/raw` gives the BEIR files, which already match
  `GetDatasets` (`scifact/corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`): move that folder to
  `<DATASETS_ROOT>/scifact`. Embed with `collection_preproc.py` / `query_preproc.py`, writing
  `scifact_corpus_embed_f32.npy` / `scifact_query_embed_f32.npy` (the commands are in `datasets/run_local.sh`, embed
  stage), and build the bins index with `./pir_app -t index -name scifact -dataset <root>`. It is small enough for
  any laptop and CPU-only embedding.
* **msmarco**: the download commands (collection, `queries.dev.small.tsv`, `qrels.dev.tsv`) are kept, commented,
  in `pre_proc_vec.slurm`; `json_to_tsv.py` / `collection_preproc.py` convert and embed. The tree's original msmarco
  artifacts (`data/msmarco_reordered`, `data/msmarco/pir/`, `data/stage3_*`) come from the collaborator's data
  folder (link in `tree/README.md`); `datasets/prep_tree.sh msmarco` only adds missing `B*_r*_s*` dirs to it.

---

## 4. Per-method setup

What each method needs before it can run, and what builds it.

### 4.1 Bins (PILLAR-Bin)

Needs: corpus, queries, qrels, the corpus/query embeddings (not for `-text`) and the Bluge index.

```bash
./pir_app -t index -name nq -dataset ../datasets        # or DATASET=nq sbatch datasets/bins_index.slurm
```

The bins themselves are built inside `pir_app` on every run (they depend on `-binSize`/`-docsPerBin`), so there is
nothing else to prepare. The PIR hint is generated at start-up too.

Flags you'll use: `-binSize` (number of bins), `-docsPerBin`, `-vectors=true|false`, `-k`, `-dim 192`. Client-hint
knobs (vec0 only): `-idBatch`, `-vecBatch`, `-vecRounds`. `-pir-download-small` downloads a DB in the clear when its
hint would be at least as big.

**[PR #8] Text mode:** `-text` stores compressed doc text (Porter stems + tf, arithmetic coded) instead of
embeddings and the client re-ranks with BM25. It needs no embedding files. Extra flags: `-text-format bow|seq`,
`-text-order 0-3`, `-text-idf-bits`, `-text-train-docs`, `-k1`, `-b`. To measure bytes/doc on a real corpus:
`go run ./bins/cmd/textbench -corpus <corpus.jsonl> -train 200000 -test 200000`. Details in `bins/TEXT.md` on that
branch. The tree does not support text.

### 4.2 Tree (PILLAR-Tree)

Needs: a single-threaded Lucene index, the Stage 1/2 tables for each (B, r, s) shape, Stage 3 doc embeddings in
index order, and the analysed queries/qrels. All of it comes from one script:

```bash
bash datasets/prep_tree.sh nq                     # every (B, r, s) the sweep needs (SWEEP=all)
SWEEP=oat bash datasets/prep_tree.sh nq           # only what SWEEP=oat needs
TREE_B=32 TREE_R=8 TREE_S=4 bash datasets/prep_tree.sh nq   # one shape (what run_local.sh does)
DATASET=nq sbatch datasets/tree_artifacts.slurm   # the same on the cluster
```

It calls `tree/build_tree_artifacts.py` (reads each term's postings once for all shapes, splits terms over
`WORKERS` processes). Existing shapes are skipped; `FORCE=1` rebuilds. Output:

```
data/<name>_tree/index_<name>/            Lucene index (+ docmap.bin)
data/<name>_tree/B<B>_r<r>_s<s>/          layout.json, stage1_data.bin, stage1_idmap.bin, stage1_vocab.json, stage2/{data,idmap}.bin
data/<name>_tree/stage3_doc.npy(.ids)
data/<name>_tree/queries.qrels.tsv, queries.qrels.analyzed.json, qrels.trec
```

msmarco and scifact keep their original index/Stage 3 files; only missing shape dirs are added under
`data/<name>/pir/` (`--db-only`). Don't use the collaborator's `tree/prepare_data*.py`: they are broken (see
`datasets/README.md`, Notes).

The grid (B, r, s, L, q, k_candidates) is defined once in `tree_sweep_grid.sh` and shared by the artifact build and
the runs. `SWEEP=legacy|oat|grid|all` (default `all`: 961 configs / 51 shapes on msmarco-sized sets, 1,675 / 57 on
scifact). See [`tree/SWEEP.md`](tree/SWEEP.md).

```bash
DATASET=nq bash tree_sweep_grid.sh --count      # configs and shapes
DATASET=nq bash tree_sweep_grid.sh --mem        # RAM model per shape
DATASET=nq bash tree_sweep_grid.sh --summary    # ready / missing per memory bucket
DATASET=nq bash tree_sweep_grid.sh --plan | head
```

Client-hint knobs: `-pir-batch-size`, `-stage3-batch-size`, `-cuckoo-hashes 2|3`, `-cuckoo-load-factor`,
`-cuckoo-key-bytes`, `-ub-bits`. For msmarco-v2, `TREE_CUCKOO_HASHES=3 TREE_CUCKOO_LF=0.88` roughly halves the
Stage 1/2 tables (results go to an `h3_lf0.88/` sub-folder). `go run ./pirhint/cmd/hintbench -dataset msmarco`
predicts client storage in a second (see `pirhint/README.md`).

### 4.3 PACMANN

Needs: the NGT + hnsw libs (`setup_pacmann.sh`, section 2.1) and the corpus/query embeddings. That's all. The
graph is built by `pir_app` on the first run of each `(neighb, steps, k)` and cached next to the embeddings as
`<name>_neigh<m>_step<s>_k<k>_graph.npy` (+ `.ngt`). The first run of a config is therefore much slower;
`GraphGenerationTime` is recorded in `metadata.json`.

### 4.4 Tiptoe

Two modes, same output folder and keys. **Use real** (see the note at the top); sim is only a cross-check:

* **sim** (default in the slurms, don't use it for results): `baselines/tiptoe/simulate_tiptoe.py` + `./simplepir_bench`
  (`setup_baselines.sh sim`). Exact quality and DB shapes, payload-only bytes (a lower bound), measured server time.
* **real**: `./pir_app_tiptoe -t tiptoe` runs the upstream code (`setup_baselines.sh tiptoe`). Compile-checked only,
  so run one small config (scifact, task 0) first and compare `MRRPreReRank` with the sim run.

Both need a preprocessed DB from `baselines/tiptoe/preprocess_tiptoe.py` (k-means clusters + URL bundles). The
slurm scripts and `run_local.sh` run it for you; it goes to `<DATASETS_ROOT>/tiptoe/`. See
[`baselines/README.md`](baselines/README.md).

### 4.5 Coeus

`baselines/coeus/run_coeus.py`: exact TF-IDF quality, exact communication from the artifact's SEAL parameters, and
server time from the artifact's own `opt2` kernel (`setup_baselines.sh coeus`; without it `TotalAnswerTime` is
left out). **Always run it with `SERVER_TIME=1`** (the note at the top). The first run calibrates `opt2` and caches the fit next to the binary, so run one task alone first.

---

## 5. Running on the cluster (slurm)

### 5.1 One command: `datasets/run_all.sh`

```bash
bash datasets/run_all.sh nq dbpedia-entity fever
bash datasets/run_all.sh msmarco-v2-all                  # v2, then the 25m/50m/100m subsamples after it's embedded
STAGES="bins pacmann" bash datasets/run_all.sh nq        # data already prepared
STAGES="tree-db tree" bash datasets/run_all.sh msmarco scifact   # originals: their data already exists
DRY_RUN=1 bash datasets/run_all.sh nq                    # print the sbatch lines and the grouping, submit nothing
```

Stages, in order (default all): `prep embed tree-db bins-index bins pacmann tree`. A stage you leave out is assumed
done. Each job waits on the ones it needs (`afterok`); if one fails, its dependants sit in
`DependencyNeverSatisfied`: fix it, `scancel` them, and re-run with `STAGES` set to what's left.

Environment knobs: `DATASETS_ROOT` (default `../datasets`), `SWEEP` (tree grid), `WORKERS` (tree artifact
processes), `NUM_SHARDS` (big embeddings), `MAX_ARRAY` (default 1000, chunk size for arrays),
`RESULTS_DIR` (default `/scratch/$USER/results`). **[PR #8]** `BINS_VECS` (default `true false text`).

`run_all.sh` does **not** queue Tiptoe or Coeus; submit those by hand (5.4).

Re-running `run_all.sh` only submits what's left: configs with a `metadata.json` are skipped, and so are configs a
pending or running job already covers (`queued_dirs` in `slurm_common.sh` maps each queued array task to its config
through `logs/lists/submitted.tsv` or sacct's `SubmitLine`). Pacmann is submitted as an `--array` list of the
unfinished configs only.

### 5.1b Checking runs and re-running after a fix: `check_runs.sh`, `run_me.sh`

```bash
bash check_runs.sh            # queue, failed jobs, done / queued / crashed / not started per method + dataset, errors
bash run_me.sh                # after every fix: pull, rebuild pir_app if needed, resubmit crashed + not started
DRY_RUN=1 bash run_me.sh      # what it would submit
```

`run_me.sh` never deletes results (unlike `build.sh`). It pulls, restores the scripts' exec bits, rebuilds `pir_app`
only when it's missing, older than the Go code or can't find `libngt`/`libhnsw` (built to `pir_app.new` and renamed,
so running jobs are unaffected), calls `run_all.sh` with `STAGES="bins pacmann tree"` (fresh estimates; OOM /
time-limit runs come back bigger), and writes the current time to `logs/check/since`. `check_runs.sh` only lists
failures after that time, so ones from before the fix drop out (`SINCE=2026-10-01 bash check_runs.sh` to see older
ones, `bash check_runs.sh --mark` to move it by hand). Its log goes to `logs/run_me/<timestamp>.log`.

### 5.2 How bins and tree jobs are sized (memory/time buckets)

For `bins` and `tree`, `run_all.sh` doesn't submit one big array with a worst-case `--mem`. It:

1. **Lists every config** (`run_bins_low/high.slurm --list-raw`, `tree_sweep_grid.sh --plan`).
2. **Checks inputs.** Configs whose data, embeddings, index or tree artifacts are missing are skipped and counted
   (`SKIPPED 935 of 961 configs: inputs not there yet (first missing: ...)`). The check is off when the jobs that
   make those inputs are queued in the same call. Configs with a `metadata.json` are skipped too (`FORCE=1` reruns).
3. **Estimates** memory and time per config (`slurm_common.sh`):
   * measured before on this dataset: peak RSS x1.2 (+1 GB), wall time x1.5 (+10 min)
   * measured on another dataset: scaled by the docs ratio (memory^1.0 tree, ^0.8 bins; time^1.0), x1.5
   * never measured: a model (`tree_mem_gb` from the DB size; `bins_model` fitted to 427 nq/msmarco runs: vocab x
     docs-per-bin for the top-doc lists, plus the PIR DB, which grows with binSize x docsPerBin until it saturates)
4. **Groups** configs into memory buckets (16/32/64/100/250/500/1000/1500 GB) x time buckets (1/2/4/6/12/24/48 h),
   and submits one array per group with exactly that `--mem`/`--time`. ≤100 GB goes to `normal`, above to `bigmem`
   (bins: `run_bins_low.slurm` / `run_bins_high.slurm`; tree: `run_tree_pir_old.slurm` for both). Configs
   estimated above 1500 GB are skipped. The config list for each array is saved in `logs/lists/<timestamp>/`
   and passed as `CONFIG_LIST`.

**`resource_usage.tsv`.** Every bins/tree run goes through `measure_run.sh` (GNU time, or polling `VmHWM`) and
appends `method dataset config peak_GB seconds date [failed:oom|timeout]` to `$RESULTS_DIR/resource_usage.tsv`
(override with `USAGE_FILE`). A run killed for memory (exit 137 or peak ≥ 90% of the request) is logged at 1.5x the
memory, a timeout at 2x the time, so just re-running `run_all.sh` puts it in a bigger bucket. The more runs come
back, the tighter the requests get, which matters because bigmem queues for hours.

Inspect a plan without submitting:

```bash
DRY_RUN=1 STAGES="bins tree" bash datasets/run_all.sh nq msmarco
DATASET=nq bash tree_sweep_grid.sh --plan | head        # per config: ... GB minutes source missing
```

Tuning: `MEM_HEADROOM`, `TIME_HEADROOM`, `SCALED_HEADROOM`, `MEM_BUCKETS`, `TIME_BUCKETS`, `NORMAL_MAX_GB`.

PACMANN is not bucketed: one array at the script's 120G (v2/miracl-en: bigmem 1500G).

### 5.3 The run scripts on their own

Each has `--count`, which prints the `--array` range to use. Without `CONFIG_LIST`, task *i* runs config *i* of the
script's own grid at the script's `#SBATCH` resources.

| Script | Grid per dataset | Datasets variable (default) | Default resources |
|---|---|---|---|
| `run_bins_low.slurm` / `run_bins_high.slurm` | vec x k{10,100} x 11 binSize x 11 docsPerBin = 484 (v2/miracl: binSize ≥ 100k only, 220), split into low/high tiers. **[PR #8]** + text = 726 | `BINS_DATASETS` (`msmarco scifact`) | 120G / 1500G bigmem, 7.5 h |
| `run_pacmann.slurm` | k{10,100} x steps{5..30} x neighb{8..48} = 72 | `PACMANN_DATASETS` (`msmarco scifact`) | 120G normal, 5 h |
| `run_tree_pir_old.slurm` | `tree_sweep_grid.sh` (961 / 1,675 on scifact) | `DATASET` (one), `SWEEP` | 100G normal, 6 h |
| `run_tiptoe.slurm` | DPC(7) x MA(2) x bundle(4) x hint(2) x k(2) = 224 | `TIPTOE_DATASETS` (`scifact nq fever msmarco`), `MODE=sim|real` | 32G, 2.5 h (real on msmarco: `--mem=64G`) |
| `run_tiptoe_msmarco_v2.slurm` | **simulation only, don't use** (see the note at the top): one task per preprocessed DB, runs its 4 (hint, k) sims | `TIPTOE_V2_DATASETS` (`msmarco-v2-50m msmarco-v2`) | 32G, 12 h (64 tasks by default) |
| `run_coeus.slurm` | F(6) x tf{raw,log} x k(2) = 24 | `COEUS_DATASETS` (`scifact msmarco`) | 120G, 2 days |

```bash
BINS_DATASETS="nq fever" bash run_bins_high.slurm --count
BINS_DATASETS="nq fever" sbatch --array=0-<N-1> run_bins_high.slurm
DATASET=nq sbatch --array=0-960 run_tree_pir_old.slurm
DATASET=scifact ARRAY_OFFSET=1000 sbatch --array=0-674 run_tree_pir_old.slurm   # 2nd chunk past MaxArraySize
```

Tasks re-check their inputs at start (`SKIP ... missing <file>`, exit 0) and skip finished configs.

### 5.4 Baselines on the cluster

```bash
# always real Tiptoe (MODE=real) and Coeus with server time (SERVER_TIME=1), see the note at the top
TIPTOE_DATASETS="nq dbpedia-entity fever" bash run_tiptoe.slurm --count
MODE=real TIPTOE_DATASETS="nq dbpedia-entity fever" sbatch --array=0-<N-1>%64 --mem=64G run_tiptoe.slurm
SERVER_TIME=1 COEUS_DATASETS="scifact msmarco nq" sbatch --array=0 --constraint=amd run_coeus.slurm       # calibrates opt2 first
SERVER_TIME=1 COEUS_DATASETS="scifact msmarco nq" sbatch --array=1-<N-1> --constraint=amd run_coeus.slurm
```

### 5.5 Client-storage sweeps

| Script | What | Results root |
|---|---|---|
| `run_bins_batch_sweep.slurm` | vec0 `-vecBatch` x `-vecRounds` x `-idBatch` | `RESULTS_ROOT` (default `/scratch/dblackle/results_vecbatch`)`/<tag>/` |
| `run_tree_grid_sweep.slurm` | tree Piano/Cuckoo knobs, 4^5 = 1024 per dataset | `/scratch/dblackle/results_tree_grid/<tag>/` |
| `run_tree_hint_ablation.slurm` | one tree config under each knob (14 tasks) | `/scratch/dblackle/results_hint/<knob>/` |

Load one tag with `RESULTS_DIR=<root>/<tag> python plotting/main.py`.

### 5.6 Old scripts

`run_tree.slurm` (pre-pir-old tree), `big_bins.slurm`, `run_test.sh` (random-docs "test" method) and the
`tree/*.slurm|*.sh` files are kept for reproducing old runs; use the scripts above instead.

---

## 6. Running locally

### 6.1 scifact, every method: `run_scifact_local.sh`

```bash
bash run_scifact_local.sh
JOBS=1 FORCE=1 bash run_scifact_local.sh     # final timings: one config at a time, rerun finished ones
```

Expects `../../../datasets/scifact/` (3.4) and `~/opt/pacmann_env.sh`. It builds `pir_app` and runs the full
Tiptoe (224) and Coeus (24) scifact sweeps, `JOBS` at a time (default 8). The bins, pacmann and tree commands
(including the one-off tree artifact build with `build_small_test_artifacts.py`) are in the file but
**commented out**: uncomment the ones you want. Tiptoe is real if `./pir_app_tiptoe` exists, otherwise simulated.
Results: `../../../datasets/{bins,pacmann,tree}/` and `../../../datasets/{tiptoe,coeus}/<config>/`, logs in
`../../../datasets/baseline_logs/`.

Everything on scifact fits in a few GB of RAM.

### 6.2 Any BEIR set, one config per method: `datasets/run_local.sh`

```bash
bash datasets/run_local.sh nq dbpedia-entity
STAGES="embed tree-db run" bash datasets/run_local.sh nq     # skip stages that are done
```

Stages (default all): `prep` (download + convert), `embed` (local GPU), `tree-db` (one shape, `TREE_B/R/S`, default
32/8/4), `run` (bins index if missing, then one bins vec1 + **[PR #8]** one bins text + one pacmann + one tree
config), `baselines` (one Tiptoe + one Coeus config). Each stage is skipped if its output exists (`FORCE=1`).
Handles `nq`, `hotpotqa`, `fever`, `trec-covid`, `dbpedia-entity` only; the big sets need the cluster.

Knobs: `DATASETS_ROOT` (default `../../../datasets`), `RESULTS_DIR` (default `<root>/results`), `K`, `BINS_BS`,
`BINS_DPB`, `TREE_L`, `QUERY_WORDS`, `FP16=1`, `ENCODE_BATCH`, `WORKERS`, `TT_DPC`, `COEUS_F`.

Rough local needs: nq (2.7M docs) embeds in a few hours on a 16 GB GPU; the single bins vec1 config at dpb 100 is
~30 GB RAM by the model, tree ~20-60 GB depending on shape, so dbpedia/fever may be tight on a 32 GB machine.
scifact is the safe choice.

### 6.3 A single run by hand

```bash
./pir_app -n 5183 -t bins -name scifact -k 10 -binSize 1000 -docsPerBin 100 -vectors=true -dim 192 \
  -dataset ../../../datasets -outDir out/bins -outFile results -debug 1
./pir_app -n 5183 -t pacmann -name scifact -k 10 -steps 15 -neighb 32 -dim 192 \
  -dataset ../../../datasets -outDir out/pacmann -outFile results -debug 1
```

The tree command is long (index, layout, stage files, queries, embeddings); copy it from `datasets/run_local.sh`
or `run_scifact_local.sh`.

---

## 7. Where results go, and plotting

### 7.1 Folders

| Method | Folder name (under the results root) |
|---|---|
| bins | `bins_vec1_<ds>_k<K>_bs<binSize>_dpb<docsPerBin>`, `bins_vec0_...`, **[PR #8]** `bins_text_...` |
| pacmann | `pacmann_<ds>_k<K>_steps<S>_neighb<M>` |
| tree | `tree_grid_<ds>_b<B>_r<r>_s<s>_L<L>_q<q>_c<k_candidates>_k10` (non-default Cuckoo: under `h3_lf0.88/`) |
| tiptoe | `tiptoe_<ds>_c<clusters>_m<multi-assign>_b<bundle>_h<hint MB>_k<K>` |
| coeus | `coeus_<ds>_k<K>_f<features>_tf<raw\|log>` |

Results roots:

* slurm: `RESULTS_DIR`, default `/scratch/$USER/results` (bins, pacmann, tree); `RESULTS_ROOT`, default
  `/scratch/dblackle/results` (tiptoe, coeus)
* `datasets/run_local.sh`: `<DATASETS_ROOT>/results` (same folder names)
* `run_scifact_local.sh`: `../../../datasets/{bins,pacmann,tree,tiptoe,coeus}`

Each folder has `metadata.json` (every metric: `MRR`, `Recall`, `MRRPreReRank`, `ClientStorageMB`,
`TotalByteSent`, `TotalLANTime`, `TotalWANTime`, `TotalAnswerTime`, `DBSizeInBytesMB`, ...), `results*.json`
(per-query answers), and on slurm `slurm_name.txt`, `execution.log`, `error.log`, `time_v.txt`. Failed tasks are
listed in `bins_low_fails`, `pacman_fails`, ... in the repo root, and live logs are in `logs/`.

Also in the results root: `resource_usage.tsv` (5.2) and `tiptoe_v2_logs/`.

**[PR #8]** `TotalByteSent`, LAN/WAN times and `ClientStorageMB` stop counting zero padding for bins (the old
values stay as `*Padded` keys). PACMANN and the tree are unaffected.

### 7.2 Plots

```bash
cd plotting
RESULTS_DIR=/scratch/$USER/results python main.py               # every figure -> plotting/figures/
RESULTS_DIR=/data/results:/data/tiptoe:/data/coeus python main.py   # several roots
python fixed_recall.py --k 10 --methods bins tree pacmann --datasets msmarco nq
```

`plotting/globals.py` holds the dataset/method lists (and **[PR #8]** `TEXT_BINS = []` hides the text runs).
Each `plot_*.py` also runs on its own.

### 7.3 Non-private baselines

```bash
python baselines.py --dataset-root ../datasets --datasets msmarco scifact nq --k 10 100 --out baselines
```

BM25, exact dense, hybrid (fixed alpha, best alpha, RRF) and BM25-top-N + dense rerank, with no PIR. Writes
`baselines/<ds>_k<K>.json`. Needs pyserini + Java 21 (falls back to `rank_bm25`).

---

## 8. Sizes and memory

Model estimates before any run is measured (`tree_sweep_grid.sh --mem`, `bins_model` in `slurm_common.sh`). Real
peaks replace them once runs log to `resource_usage.tsv`.

| Dataset | Docs | Embeddings on disk (N x 192 x 4 B) | Tree RAM per config (min-max shape) | Bins RAM, vec0 dpb100 / vec1 dpb100 / vec1 dpb2500 |
|---|---|---|---|---|
| scifact | 5.2K | 4 MB | ~4 GB | 2 / 2 / 14 GB |
| nq | 2.7M | 2.1 GB | 22-63 GB | 12 / 27 / 517 GB |
| dbpedia-entity | 4.6M | 3.6 GB | 26-67 GB | 18 / 39 / 719 GB |
| fever | 5.4M | 4.2 GB | 45-125 GB | 20 / 43 / 790 GB |
| msmarco | 8.8M | 6.8 GB | 50-135 GB | 29 / 60 / 1062 GB |
| miracl-en | 32.9M | 25 GB | 288-835 GB | 86 / 154 / 2359 GB |
| msmarco-v2-25m | 25M | 19 GB | 161-453 GB | 68 / 126 / 1996 GB |
| msmarco-v2-100m | 100M | 77 GB | 634-1791 GB | 231 / 365 / 4661 GB |
| msmarco-v2 | 138M | 106 GB | 875-2474 GB | 311 / 475 / 5694 GB |

Anything estimated above 1500 GB is skipped by `run_all.sh`. The msmarco-v2 tree needs the Cuckoo option
(`TREE_CUCKOO_HASHES=3 TREE_CUCKOO_LF=0.88`), which roughly halves Stage 1/2.

Other sizes worth knowing:

* msmarco's server-side tree DB is ~38 GB on its own. Tree client hint on msmarco: ~5.9 GB synthetic estimate,
  ~4.3 GB with all Cuckoo options (`pirhint`).
* **[PR #8]** bins text: ~67 B/doc + ~2 B index on a proxy corpus, against 768 B for an embedding (so ~0.6 GB of
  text for msmarco instead of ~6.3 GiB). Not yet measured on the real corpora (run `textbench`).
* Tiptoe: preprocessing peaks at ~4 GB + <100 B/doc (~15 GB on v2); the v2 npy DB is ~34 GB per copy (deleted after
  its sims unless `KEEP_TT=1`). Real Tiptoe on msmarco: ~64 GB.
* Coeus: ~12 B per (doc, term) of postings, ~50 GB on v2, plus a ~25 GB temp token file in `--tmp-dir`. The real
  Coeus matrix for msmarco would be ~8.4 TiB, which is why it is modelled.
* Tree artifact build on v2: a common term's postings can be ~10 GB per worker, hence 1000G and 12 workers.

---

## 9. Gotchas

* `build.sh` deletes old bins/pacmann results and cached PACMANN graphs (2.2). `run_me.sh` rebuilds without that.
* Don't commit `pir_app`: a binary built on another machine has that machine's library paths baked in (its rpath),
  so on the cluster it fails with `libngt.so.2: cannot open shared object file` (exit 127). It's in `.gitignore`.
* If your checkout is on a filesystem without exec bits, run `git config core.fileMode false` there, or a commit
  from it drops the scripts' `+x` and every tree run fails with exit 126 (the run scripts now call
  `bash measure_run.sh`, and `run_me.sh` restores the bits, so this can't break runs again).
* Three different default dataset roots: `../datasets` (run_all + dataset slurms), `../../datasets` (run slurms on
  their own), `../../../datasets` (local scripts). Set `DATASETS_ROOT` / `DATASET_DIR` if in doubt.
* The sweep scripts default to `/scratch/dblackle/...`; tiptoe/coeus use `RESULTS_ROOT`, the rest `RESULTS_DIR`.
* A new dataset name must be added to `GetDatasets` (`main.go`, only if it doesn't follow the generic layout),
  `baselines/common.py` `DATASETS`, `DATASET_EMBEDS` in `baselines/tiptoe/preprocess_tiptoe.py`, and
  `DATASET_SIZES` in `slurm_common.sh` (for the resource model before `num_docs.txt` exists).
* Subsamples (`msmarco-v2-*m`) must be prepared after v2 is **embedded**; `run_all.sh msmarco-v2-all` orders that.
* Tiptoe caps clusters at 25,196, so on 100M+ docs use `--docs-per-cluster 4000` or more (the v2 script merges
  DPC values that collapse to the cap).
* `go.mod` replaces `underhood` (bad pseudo-version) and uses `third_party/simplepir` (upstream doesn't compile on
  Go 1.24); see `third_party/README.md`.
* `plotting/client_estimate.py` is imported by `load_results.py` but is not in the repo.

---

## 10. Map of the repo

```
main.go                     pir_app entry point, flags, GetDatasets
bins/                       PILLAR-Bin (bins/textcodec, bins/cmd/textbench: [PR #8])
tree/                       PILLAR-Tree (Go in tree/internal, artifact builder build_tree_artifacts.py, SWEEP.md)
pir-old/                    the old PIR the tree runs on
pianopir/                   Piano PIR used by bins
Pacmann/                    PACMANN (graphann = NGT graph build)
tiptoe/, baselines/         Tiptoe wrapper (-tags tiptoe), Tiptoe/Coeus scripts, setup_baselines.sh
pirhint/                    client-hint calculator + hintbench
datasets/                   download / convert / embed / index / artifacts, run_all.sh, run_local.sh
plotting/                   load_results.py, main.py, plot_*.py, fixed_recall.py
baselines.py                non-private BM25 / dense / hybrid baselines
slurm_common.sh             dataset sizes, usage log, memory/time estimates and buckets
measure_run.sh              peak RSS + wall time wrapper
tree_sweep_grid.sh          the tree grid (shared by artifacts and runs)
run_*.slurm                 per-method slurm arrays (section 5.3)
setup_pacmann.sh, go_env.sh, build.sh, build_go.slurm   build
collection_preproc.py, query_preproc.py                 e5 + PCA embedding
```
