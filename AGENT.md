# Notes for coding agents

## Git
- Commit locally, but never `git push`. The user pushes.

## Hopper (the cluster, `ssh hopper`)
- Read-only: only look at files and logs (`cat`, `less`, `tail`, `grep`, `ls`, `awk` on a log, `git log`/`git status`).
- Do not run anything that does real work or queries the scheduler in bulk: no `sbatch`, `squeue`/`sacct` loops,
  `check_runs.sh`, `run_me.sh`, `run_all.sh`, `run_sim_noram.sh` (not even with `DRY_RUN=1`), builds, or Python.
  Heavy or repeated commands on the login node can get the account banned.
- Do not change files there, not even temporarily. When something needs to run on hopper, give the user the exact
  command and let them run it.
- Useful read-only places: `/scratch/dblackle/bins-go/logs/` (job logs, `logs/check/<date>/*_not_done.tsv` written by
  the user's own `check_runs.sh`), `/scratch/dblackle/results_simulated/resource_usage.tsv`, and the results folders.
