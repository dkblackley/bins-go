# Tree parameter sweep

The grid is defined once in `tree_sweep_grid.sh` and used by both
`datasets/prep_tree.sh` (builds the missing `B*_r*_s*` artifact dirs) and
`run_tree_pir_old.slurm` (runs the configs). It covers every dataset:
msmarco, scifact, nq, dbpedia-entity, fever, hotpotqa, miracl-en and
msmarco-v2 (+ 25m/50m/100m). The new datasets and msmarco-v2 use the msmarco grid.

```bash
# how big is it
DATASET=nq SWEEP=all bash tree_sweep_grid.sh --count       # 961 configs, 51 artifact dirs
DATASET=scifact SWEEP=all bash tree_sweep_grid.sh --count  # 1675 configs, 57 artifact dirs

# everything for every DB: artifacts, then the sweep (chunked past MaxArraySize), chained with dependencies
STAGES="tree-db tree" bash datasets/run_all.sh msmarco scifact nq dbpedia-entity fever miracl-en msmarco-v2 \
    msmarco-v2-25m msmarco-v2-50m msmarco-v2-100m
DRY_RUN=1 STAGES="tree-db tree" bash datasets/run_all.sh nq   # print the sbatch lines only

# or by hand for one dataset
DATASET=nq sbatch datasets/tree_artifacts.slurm             # existing combos are skipped
DATASET=nq sbatch --array=0-960 run_tree_pir_old.slurm      # finished configs are skipped
DATASET=scifact ARRAY_OFFSET=1000 sbatch --array=0-674 run_tree_pir_old.slurm   # second chunk
```

For msmarco and scifact, `prep_tree.sh` keeps the original index and stage-3
files and only adds the missing dirs under `data/<name>/pir/` (`--db-only`).
`SWEEP` selects the grid for both steps (default `all`).

Result folders are now `tree_grid_<ds>_b<B>_r<r>_s<s>_L<L>_q<q>_c<k_candidates>_k10`.
`plotting/load_results.py` reads the old names too.

## Sweeps

| SWEEP  | what | msmarco | scifact |
|--------|------|---------|---------|
| legacy | old Cartesian grid, `k_candidates` fixed (128 / 20) | 432 | 960 |
| oat    | one parameter at a time from its minimum, rest at the anchor | 43 | 37 |
| grid   | joint grid, small values first, `k_candidates = P / s` for pools P | 918 | 1638 |
| all    | oat + grid | 961 | 1675 |

Anchors: msmarco `B32 r32 s8 L100 q4 c128`, scifact `B16 r8 s4 L20 q8 c20`.

`k_candidates` is clamped to `L*B/s` (the number of sub-blocks Round 2 can
see), so very small L automatically keeps every sub-block.

## Which knobs should matter

Per query, roughly:

* Round 1 fetches `H * L * q` rows of `r` bytes, with `H = ceil(log_r(N/B))` levels (one PIR round each).
* Round 2 fetches `L * q` rows of `B/s` bytes.
* Stage 3 fetches `c * ceil(s/8)` blocks of 8 embeddings, and reranks `c * s` docs.

Expected effect (not measured yet):

* **L and c are the main quality/cost knobs.** The old grid never went below
  L=50 (msmarco) and kept c fixed, so Stage 3 cost and the rerank pool barely
  moved and cost scaled almost linearly with L. That is the likely cause of the
  "straight line". L in 1..32 and c in 1..64 should fill the cheap end of the
  curve.
* **q** cuts query terms. msmarco queries are short, so q=3..4 should be close
  to q=8 in quality at half the cost; q=1..2 shows the low end.
* **r** trades depth for row width. r=2 gives ~21 levels on msmarco (many
  rounds, tiny rows), r=256 gives 2 levels with 256-byte rows. It should move
  points mostly along the latency/rounds axis.
* **B and s** set Round 2 row width (B/s) and rerank granularity. s below 8
  wastes part of each 8-doc Stage 3 PIR block, so s=8 is probably the sweet
  spot; s=1 and B=4 are there to show the extreme.

## Artifact builder

`tree/build_tree_artifacts.py` takes explicit triples with `--combos B,r,s ...`
(which `prep_tree.sh` fills from `tree_sweep_grid.sh --shapes`). Only those
combos are built, not the product of their values. The output on synthetic
data is byte-identical to `build_small_test_artifacts.py`. Like the other
scripts, it uses avgdl-based bounds and no `min_dl_map`. If the original
msmarco artifacts were built with a `min_dl_map`, rebuild them with `FORCE=1`
for a like-for-like comparison.
