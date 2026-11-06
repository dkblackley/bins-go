# Tree parameter sweep

The grid is defined once in `tree_sweep_grid.sh` and used by both
`build_tree_artifacts.slurm` (builds missing `B*_r*_s*` artifact dirs) and
`run_tree_pir_old.slurm` (runs the configs).

```bash
# 1. how big is it
DATASET=msmarco SWEEP=all bash tree_sweep_grid.sh --count      # 961 configs, 51 artifact dirs
DATASET=scifact SWEEP=all bash tree_sweep_grid.sh --count      # 1675 configs, 57 artifact dirs

# 2. build the artifacts the sweep needs (one task per (B, r); existing dirs are skipped)
DATASET=msmarco SWEEP=all bash build_tree_artifacts.slurm --count
DATASET=msmarco SWEEP=all sbatch --array=0-21 build_tree_artifacts.slurm

# 3. run (finished configs are skipped, so resubmitting is safe)
DATASET=msmarco SWEEP=all sbatch --array=0-960 run_tree_pir_old.slurm
# arrays above MaxArraySize (often 1001): send the rest with an offset
DATASET=scifact SWEEP=all sbatch --array=0-999 run_tree_pir_old.slurm
DATASET=scifact SWEEP=all ARRAY_OFFSET=1000 sbatch --array=0-674 run_tree_pir_old.slurm
```

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

`tree/build_tree_artifacts.py` writes Stage 1 once per (B, r) and Stage 2 for
every s, in one pass over the postings, using the same maths as
`build_small_test_artifacts.py` (byte-identical on synthetic data). It uses
avgdl-based upper bounds with no `min_dl_map`, like the non-`--compute-leaf-mins`
path of `prepare_data_stage2.py`. Existing artifact dirs are left alone, so if the
original msmarco artifacts were built with a `min_dl_map`, rebuild them with
`FORCE=1` for a like-for-like comparison.
