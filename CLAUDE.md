# bins-go: guide for Claude

PIR (private information retrieval) document-retrieval experiments for the PILLAR paper: two PILLAR methods (Bin,
Tree) and the baselines they are compared against, run as large parameter grids on the GMU Hopper cluster (slurm) and
plotted for the paper and rebuttal. `README.md` is the detailed manual (section numbers below refer to it); this file
is the map.

@AGENT.md

## Rules (also in AGENT.md)
- Commit locally, never `git push`.
- Hopper (`ssh hopper`, repo at `/scratch/dblackle/bins-go`) is read-only for agents: read files and logs, run nothing
  (no sbatch/squeue/sacct, no check_runs.sh/run_me.sh/run_all.sh, not even `DRY_RUN=1`), change nothing. Give the user
  the command instead.
- Baselines measure quality + communication only: never add the server-time benchmarks (`simplepir_bench`,
  Coeus `opt2`, `--bench-bin`, `--opt2-bin`) back to any script or command.
- Don't commit `pir_app` (machine-specific rpath).

## The methods
One Go binary, `pir_app` (`main.go`), runs every Piano-PIR method with `-t <method>`:

| method | paper name | code | slurm grid |
|---|---|---|---|
| `bins` | PILLAR-Bin | `bins/` (keyword bins, Bluge index), PIR in `pianopir/` | `run_bins_high.slurm` / `run_bins_low.slurm` |
| `tree` | PILLAR-Tree | `tree/` (Go in `tree/internal`, artifacts via `tree/build_tree_artifacts.py`), PIR in `pir-old/` | `run_tree_pir_old.slurm`, grid in `tree_sweep_grid.sh` |
| `pacmann` | PACMANN baseline | `Pacmann/` (NGT graph, cgo) | `run_pacmann.slurm` |
| `tiptoe` | Tiptoe baseline | `tiptoe/`, `baselines/tiptoe/` (built with `-tags tiptoe`) | `run_tiptoe*.slurm` |
| `coeus` | Coeus baseline | Python, `baselines/` | `run_coeus.slurm` |

Non-private BM25/dense/hybrid reference numbers: `baselines.py` (README 7.3). Client-hint sizes: `pirhint/`.
Low-RAM mode (`-noram`): `diskstore/` + `pianopir/noram.go`. `-simulate` skips hints and XOR: same answers, MRR,
bytes, LAN/WAN and client storage, but server timings are -1. **Simulated low-RAM runs are the default since 2026-11-10.**

## Running things (README 5)
- `datasets/run_all.sh <datasets>`: the whole pipeline, stage by stage (prep, embed, tree-db, bins-index, bins,
  pacmann, tree, tiptoe, coeus; `STAGES=...` picks some). It plans each config's memory/time and submits one job
  array per bucket. It skips configs that already have a `metadata.json` or are already queued.
- `run_sim_noram.sh` = `run_all.sh` with `SIMULATE=1 NORAM=1`. It does **not** `git pull`.
- `run_me.sh`: what the user runs after every fix. It pulls, fixes exec bits, rebuilds `pir_app` if needed, resubmits
  crashed and not-started configs with fresh estimates, cleans caches, and moves the "since" mark. `REAL=1` gives real runs.
- `check_runs.sh`: read-only status (queue, failed jobs, done/queued/crashed per method and dataset, common errors).
  It writes `logs/check/<timestamp>/<method>_<dataset>_not_done.tsv`; read those on hopper instead of running it.
  `EXPECTED_ONLY=1` lists every config the grids produce (works locally with `module` stubbed; see
  `plotting/save_current_grid.sh`).
- Resource model: `slurm_common.sh` has the dataset sizes, the usage log (`$RESULTS_DIR/resource_usage.tsv`, one
  line per finished/failed run), and the estimates (`estimate_resources`, `_estimate_resources`, `bins_model`,
  `bins_sim_noram_model`). A measured run replaces the model. An OOM is logged at 1.5x so the retry asks for more.
  For tree, a config never measured on a dataset asks for at least that shape's smallest run there.
- Local runs: `run_scifact_local.sh`, `datasets/run_local.sh`, `local_run_go_simulated.sh` (README 6).

## Where to change what
- **Bins grid**: `BS_LIST` / `DPB_LIST` / `BINS_VECS` near the top of `run_bins_high.slurm` **and**
  `run_bins_low.slurm` (keep them identical). The vec0 sweep is `vec0_sweep_param`.
- **Tree grid**: `tree_sweep_grid.sh` (B r s L q c). `SWEEP=legacy|oat|grid|lowmrr|all`. `lowmrr` holds extra
  configs only for the fixed-MRR plots. `c` is clamped to `L*B/s`. `--count` / `--list` / `--mem` inspect it.
  `tree/SWEEP.md` explains it. MRR knobs: q > L > c, while B/r/s barely matter.
- **PACMANN grid**: a `case` in `run_pacmann.slurm` (`PACMANN_GRID`, currently g3; old grids are kept so queued
  jobs can still be decoded).
- **New dataset**: `GetDatasets` in `main.go` (only if the layout isn't generic), `baselines/common.py`
  `DATASETS`, `DATASET_EMBEDS` in `baselines/tiptoe/preprocess_tiptoe.py`, and `DATASET_SIZES` in `slurm_common.sh`
  (README 9).
- **Memory/time estimates, buckets, partitions**: `slurm_common.sh`.
- **Result folder names**: README 7.1. The same names are rebuilt in `queued_dirs` (`slurm_common.sh`),
  `check_runs.sh` and `plotting/load_results.py`, so change all of them together.
- **Metrics written per run**: `main.go` (`calcMRR`, `calcRecall`, metadata) → `metadata.json`.

## Results and plotting (README 7)
- Each config writes one folder: `metadata.json` holds every metric (MRR, Recall, ClientStorageMB, TotalByteSent,
  TotalLAN/WANTime, ...), plus `results*.json` with the per-query answers. No `metadata.json` means crashed or unfinished.
- Roots: real runs go to `/scratch/dblackle/results`, simulated runs to `/scratch/dblackle/results_simulated`.
  Locally, results are rsync'd to `../../../datasets/results_simulated`, which the loader reads by default.
  Rsync only `metadata.json` + `results*.json`; see the top of `local_run_go_simulated.sh`.
- `plotting/`: `globals.py` holds every setting and env flag (datasets, methods, colours, filters).
  `load_results.py` loads the results into `{method: {dataset: [runs]}}`. `main.py` makes every figure; comment
  lines in or out there. Each `plot_*.py` also runs alone.
- Loader filters, all on by default:
  - `ONLY_SIMULATED`.
  - The client-storage filter (`FILTER_CLIENT_STORAGE`).
  - `ONLY_CURRENT_GRID`: only configs in `plotting/current_grid.txt` load. Regenerate the file with
    `bash plotting/save_current_grid.sh` when a grid change should show up everywhere.
  - The fixed-MRR scripts (`plot_fixed_mrr.py`, `fixed_recall.py`, `frozen_values.py`) and `main.py`'s
    `all_data` pass `only_grid=False`, so they see every run.
- Rebuttal/camera-ready outputs: `rebuttal_final.py`, `plot_fixed_mrr.py` (`TARGETS` per dataset, ±0.01 window),
  `plot_fixed_storage.py`, `scale_v2.py`.

## Gotchas (README 9)
- `build.sh` deletes old bins/pacmann results. Use `run_me.sh` to rebuild without deleting.
- There are three different default dataset roots (`../datasets`, `../../datasets`, `../../../datasets`). Set
  `DATASETS_ROOT` when in doubt.
- msmarco-v2 (full) tree uses Cuckoo `h3_lf0.88`, so its results sit under `h3_lf0.88/`.
- The bins Bluge index was rebuilt on 2026-11-13 with stemming. Bins results from before then are stale.
- Go 1.24. Only PACMANN needs cgo, while bins/tree build with `CGO_ENABLED=0`. Tests: `go test ./bins/ ./pianopir/
  ./pirhint/` (the Go packages with `_test.go` files).
