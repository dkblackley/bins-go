# PILLAR: private retrieval-augmented generation with PIR

Code for the paper's experiments. One Go binary, `pir_app`, runs every retrieval method; Python scripts prepare the
datasets, run the Coeus / non-private baselines, and draw the figures and tables.

| `pir_app -t` | Method | What the server's PIR database holds |
|---|---|---|
| `bins` | PILLAR-Bin | per-keyword bins of document embeddings (`-vectors=true`), an ID DB + an embedding DB (`-vectors=false`), or compressed document text (`-text`) |
| `tree` | PILLAR-Tree | a BM25 upper-bound tree (Stage 1/2 Cuckoo tables) + Stage 3 embeddings |
| `pacmann` | PACMANN baseline | an NGT graph + embeddings |
| `tiptoe` | Tiptoe baseline | the upstream protocol (build with `-tags tiptoe`); the experiments use the simulator in `baselines/tiptoe/` |
| `index` | (setup) | builds the keyword index that `bins` needs, then exits |

`pir_app -simulate` runs the same protocols without the PIR hint/XOR work: answers, quality, bytes sent, LAN/WAN times
and client storage are exact, and server timings are reported as -1. `pir_app -noram` keeps the PIR databases,
client hints and embeddings on disk (mmap) for machines with little RAM; results are identical, only slower.

## Requirements

* Go >= 1.24, gcc, cmake
* NGT + hnsw native libraries (PACMANN is linked into every build): `bash setup_pacmann.sh` installs them to `~/opt`
* Python 3.10 with `requirements.txt` and `faiss-cpu` (or `faiss-gpu`)
* Java 21 (pyserini / Lucene, for the tree artifacts and the BM25 baseline)
* A GPU for the e5-base-v2 embeddings of the larger corpora (a 16 GB card is enough)
* Only for the real (non-simulated) Tiptoe: SEAL 4.1.1, `baselines/setup_baselines.sh tiptoe`

```bash
conda create -n main python=3.10 && conda activate main
pip install -r requirements.txt faiss-cpu
conda install -c conda-forge openjdk=21
bash setup_pacmann.sh
source go_env.sh && go build -o pir_app ./main.go
```

## Running the experiments: `local_run.sh`

One script runs the whole pipeline and every parameter sweep of the paper, one configuration after another:

```bash
DATASETS=scifact bash local_run.sh     # everything on SciFact: minutes, a few GB of RAM
bash local_run.sh --count              # how many configurations each sweep has, per dataset
DRY_RUN=1 bash local_run.sh tree       # print the commands of one sweep instead of running them
bash local_run.sh                      # every dataset and sweep (the largest configs need TBs of RAM)
```

Stages, in order: `prep` (download + convert), `embed` (e5-base-v2 + PCA to 192 dims), `index` (PILLAR-Bin's keyword
index), `treedb` (PILLAR-Tree's artifacts), then the sweeps `bins`, `pacmann`, `tree`, `tiptoe`, `coeus`, `docfetch`
(cost of privately downloading the returned documents) and `plain` (non-private BM25 / dense / hybrid). Name stages
to run only those, e.g. `bash local_run.sh bins tree`. Finished configurations are skipped, so the script can be
stopped and restarted. Options (`DATASETS_ROOT`, `RESULTS_DIR`, `SIMULATE=1`, `NORAM=1`, `ONLY=<regex>`, ...) are
listed at the top of the file; the grids are written out in it, except the tree's, which is in `tree_sweep_grid.sh`
(see [`tree/SWEEP.md`](tree/SWEEP.md)). Each configuration is an independent command, so the loops translate
directly into job arrays for a cluster scheduler.

`SIMULATE=1` runs the PIR protocols without the hint and XOR work: answers, quality, bytes sent, LAN/WAN time and
client storage are those of a real run, server timings are -1, and far less RAM is needed.

A single run by hand:

```bash
./pir_app -t bins -name scifact -n 5183 -k 10 -binSize 1000 -docsPerBin 100 -vectors=true -dim 192 \
  -dataset ../datasets -outDir out/bins -outFile results
./pir_app -t pacmann -name scifact -n 5183 -k 10 -steps 15 -neighb 32 -dim 192 \
  -dataset ../datasets -outDir out/pacmann -outFile results
```

## Datasets

SciFact, NQ, DBPedia-Entity, FEVER, MS MARCO v1 and v2 (with 25M/50M/100M subsamples of v2) and MIRACL-en, from 5K
to 138M documents. See [`datasets/README.md`](datasets/README.md).

## Baselines

* Tiptoe and Coeus: [`baselines/README.md`](baselines/README.md). Both report exact retrieval quality and exact
  communication (and the LAN/WAN times derived from it).
* Non-private BM25 / dense / hybrid: the `plain` stage (`cmd/plain_baselines`), or `python baselines.py --dataset-root ../datasets --datasets scifact nq --k 10 100`.

## Results and plots

Each run writes a folder such as `bins_vec1_<dataset>_k10_bs<binSize>_dpb<docsPerBin>/` with `metadata.json` (MRR,
Recall, client storage, bytes sent, LAN/WAN and server times, DB size, ...) and the per-query answers.

```bash
cd plotting
RESULTS_DIR=/path/to/results python main.py      # every figure -> plotting/figures/
python working_rebuttal.py                       # the tables
```

## Repository layout

```
main.go             pir_app entry point and flags
bins/               PILLAR-Bin (bins/textcodec: compressed document text)
tree/               PILLAR-Tree (Go in tree/internal, artifact builder build_tree_artifacts.py)
pianopir/, pir-old/ the Piano PIR implementations used by the bins and the tree
diskstore/          -noram: mmap'd storage for PIR databases, hints and embeddings
Pacmann/            PACMANN baseline
tiptoe/, baselines/ Tiptoe wrapper and the Tiptoe / Coeus scripts
pirhint/            client-hint size calculator
cmd/                docfetch (private document download cost), plain_baselines, simplepir_bench
datasets/           download / convert / embed / index / artifacts, run_all.sh, run_local.sh
plotting/           result loading, figures, tables
third_party/        vendored SimplePIR (patched for Go 1.24)
```

## License

MIT, see `LICENSE`.
