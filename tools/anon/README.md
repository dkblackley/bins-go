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

## Quick start

The local scripts expect the datasets in `../../../datasets` (set `DATASETS_ROOT` to change it).

```bash
bash datasets/run_local.sh nq        # download, convert, embed, build indexes, then one config per method
bash run_scifact_local.sh            # SciFact (fits on a laptop): the Tiptoe/Coeus sweeps; the bins, PACMANN and
                                     # tree commands are in the file, commented, to enable as needed
```

A single run by hand on SciFact (BEIR files from `bash datasets/download.sh scifact <raw dir>`, embedded with
`collection_preproc.py` / `query_preproc.py`):

```bash
./pir_app -t index -name scifact -dataset ../datasets
./pir_app -t bins -name scifact -n 5183 -k 10 -binSize 1000 -docsPerBin 100 -vectors=true -dim 192 \
  -dataset ../datasets -outDir out/bins -outFile results
./pir_app -t pacmann -name scifact -n 5183 -k 10 -steps 15 -neighb 32 -dim 192 \
  -dataset ../datasets -outDir out/pacmann -outFile results
```

The tree needs its artifacts first (`bash datasets/prep_tree.sh scifact`); the full command is in
`datasets/run_local.sh`.

## Datasets

`datasets/` downloads, converts and embeds SciFact, MS MARCO (v1, v2 and 25M/50M/100M subsamples of v2), NQ,
DBPedia-Entity, FEVER and MIRACL-en. Embeddings are e5-base-v2 reduced to 192 dimensions with a per-dataset PCA. See
[`datasets/README.md`](datasets/README.md).

## Full experiments (slurm)

```bash
bash datasets/run_all.sh nq dbpedia-entity fever     # download -> embed -> indexes/artifacts -> bins/pacmann/tree grids
STAGES="tiptoe coeus" bash datasets/run_all.sh nq    # the baselines (quality + communication)
```

`run_all.sh` chains the jobs with dependencies and sizes each configuration's memory/time request from earlier runs
(`slurm_common.sh`). The per-method arrays are `run_bins_low.slurm`, `run_bins_high.slurm`, `run_pacmann.slurm`,
`run_tree_pir_old.slurm`, `run_tiptoe.slurm` and `run_coeus.slurm`; each takes `--count` to print its array size.
The tree's parameter grid is defined once in `tree_sweep_grid.sh` (see [`tree/SWEEP.md`](tree/SWEEP.md)).
Add your cluster's `#SBATCH --account=` / partition and `module load` lines where your site needs them.

To run every configuration on one machine with simulated PIR: `bash local_run_go_simulated.sh plan|run`.

## Baselines

* Tiptoe and Coeus: [`baselines/README.md`](baselines/README.md). Both report exact retrieval quality and exact
  communication (and the LAN/WAN times derived from it).
* Non-private BM25 / dense / hybrid: `python baselines.py --dataset-root ../datasets --datasets scifact nq --k 10 100`.

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
