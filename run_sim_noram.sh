#!/bin/bash
# run_sim_noram.sh: queue simulated, low-RAM bins + pacmann runs on Hopper (pir_app -simulate -noram), sized for
# what those runs really use: a few GB and minutes instead of the real runs' hundreds of GB and hours.
#
#   bash run_sim_noram.sh scifact trec-covid nq dbpedia-entity fever msmarco msmarco-v2-25m
#   DRY_RUN=1 bash run_sim_noram.sh nq         # print the sbatch lines (memory/time per group), submit nothing
#   STAGES=bins bash run_sim_noram.sh nq       # only bins (default "bins pacmann")
#   STAGES=tree bash run_sim_noram.sh nq fever # simulated tree runs (STAGES="bins pacmann tree" for all three)
#
# The tree artifacts (tree-db) and the bins' Bluge index (bins-index) are queued too, but only for a dataset where
# they are missing; the runs wait for them. prep/embed are never queued here (run those without SIMULATE).
# Simulated tree configs are sized from the real run of the same config when there is one
# (/scratch/$USER/results/resource_usage.tsv), else from the tree model (x0.9 memory, x0.5 time).
#
# It is datasets/run_all.sh with SIMULATE=1 NORAM=1, which switches slurm_common.sh to:
#   * bins_sim_noram_model: memory/time measured from simulated -noram runs (see its comment), not the real model
#   * a floor of SIM_MIN_GB=12 GB and SIM_MIN_MIN=30 min, buckets 12/16/20/.. GB and 30/60/90/.. min
#   * measured runs (results_simulated/resource_usage.tsv, written by every finished job) replace the model:
#     peak x 1.2 + 1 GB + NORAM_CACHE_GB (2), elapsed x 1.5 + 10 min, so a re-run of this script after a first wave
#     sizes the rest from real numbers. An OOM/timeout is logged as needing 1.5x, so just re-run to retry those.
#   * pacmann: pacmann_sim_noram_model per dataset size, instead of run_pacmann.slurm's fixed 120 GB / 5 h
# Everything run_all.sh takes still works (MAX_GB, FORCE, MAX_ARRAY, NORAM_DIR, ...).
#
# Each -noram run writes its PIR DB to NORAM_DIR (default temp_storage/ in the repo, deleted when the run ends):
# up to tens of GB for the biggest vec1 configs, so many at once need the /scratch quota for it.
set -euo pipefail
cd "$(dirname "$0")"
export SIMULATE=1 NORAM=1
export STAGES="${STAGES:-bins pacmann}"
exec bash datasets/run_all.sh "$@"
