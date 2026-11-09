#!/bin/bash
# Status of the bins / pacmann / tree / tiptoe / coeus runs. Read-only: never submits, cancels, moves or deletes.
#
#   bash check_runs.sh                              # scifact msmarco nq fever dbpedia-entity, every method
#   bash check_runs.sh nq fever                     # only these datasets
#   METHODS="bins tree" SINCE=2026-11-01 bash check_runs.sh nq
#
# Sections:
#   1. queue        your jobs in squeue, per job name / state / pending reason
#   2. failed jobs  sacct since $SINCE (default 14 days ago): FAILED, OUT_OF_MEMORY, TIMEOUT, CANCELLED, ...
#   3. configs      every config each method's grid should produce, per dataset: done (metadata.json) /
#                   crashed (folder but no metadata.json: failed, or still running) / not started.
#                   Tiptoe: how many results are SIMULATED (we want real ones). Coeus: how many have no server time.
#   4. errors       the most common error lines in live logs (logs/*.err) and in the results' error.log files
#   5. usage log    OOM / timeout entries in resource_usage.tsv (they make run_all.sh pick bigger buckets next time)
#   6. fail files   bins_low_fails, pacman_fails, ... (tasks that exited without finishing)
# The full lists of crashed / not started configs (with each crash's last error line) go to logs/check/<timestamp>/.
#
# Env: DATASETS_ROOT/DATASET_DIR (../datasets), RESULTS_DIR (/scratch/$USER/results: bins, pacmann, tree),
#      RESULTS_ROOT (/scratch/dblackle/results: tiptoe, coeus), SWEEP (tree grid, default all), SINCE, METHODS,
#      TREE_CUCKOO_HASHES/TREE_CUCKOO_LF (only if the tree runs used them), TIPTOE_DPC etc. are not needed.
set -uo pipefail
cd "$(dirname "$0")"
export DATASETS_ROOT="${DATASETS_ROOT:-../datasets}"
export DATASET_DIR="${DATASET_DIR:-$DATASETS_ROOT}"
RES="${RESULTS_DIR:-/scratch/$USER/results}"
BRES="${RESULTS_ROOT:-/scratch/dblackle/results}"
SINCE="${SINCE:-$(date -d '-14 days' +%F)}"
METHODS="${METHODS:-bins pacmann tree tiptoe coeus}"
DATASETS="${*:-scifact msmarco nq fever dbpedia-entity}"
OUT="logs/check/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
has() { [[ " $METHODS " == *" $1 "* ]]; }
hr() { printf '\n==== %s\n' "$*"; }

# ---------------------------------------------------------------------------------------------------- 1. queue
hr "1. queue (squeue, array tasks expanded)"
squeue -u "$USER" -h -r -o "%j|%T|%r" 2>/dev/null \
  | awk -F'|' '{ r = ($2 == "PENDING") ? $3 : ""; n[$1 "|" $2 "|" r]++ }
      END { for (k in n) { split(k, a, "|"); printf "  %-36s %-10s %-28s %6d\n", a[1], a[2], a[3], n[k] } }' | sort \
  || echo "  (squeue not available)"
echo "  DependencyNeverSatisfied = something it waits on failed: fix it, scancel these, re-run run_all.sh"

# ---------------------------------------------------------------------------------------------- 2. failed jobs
hr "2. jobs that didn't complete since $SINCE (sacct; up to 4 example ids each)"
sacct -u "$USER" -S "$SINCE" -X -n -P -o JobID,JobName,State,ExitCode 2>/dev/null \
  | awk -F'|' '{ st = $3; sub(/ .*/, "", st) }
      st !~ /^(COMPLETED|RUNNING|PENDING|REQUEUED|RESIZING)$/ {
        k = $2 "|" st; n[k]++; if (n[k] <= 4) ex[k] = ex[k] " " $1 }
      END { for (k in n) { split(k, a, "|"); printf "  %-36s %-14s %6d  e.g.%s\n", a[1], a[2], n[k], ex[k] } }' | sort \
  || echo "  (sacct not available)"
echo "  look at one: sacct -j <id> -o JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed,Timelimit ; seff <id>"
echo "  its log: logs/<Bins|Pacmann|Tree|Tiptoe|Coeus>_<arrayjob>_<task>.err (moved into the results folder when it finishes)"

# ------------------------------------------------------------------------------------- 3. expected vs finished
# one line per expected config: method dataset results_folder config...
{
for d in $DATASETS; do
    if has bins; then
        { BINS_DATASETS="$d" SLURM_SUBMIT_DIR="$PWD" bash run_bins_low.slurm --list-raw 2>/dev/null
          BINS_DATASETS="$d" SLURM_SUBMIT_DIR="$PWD" bash run_bins_high.slurm --list-raw 2>/dev/null; } |
        while read -r ds k bs dpb vec; do
            case $vec in true) t=_vec1;; false) t=_vec0;; text) t=_text;; esac
            sfx=""; [[ $vec == text && ( "${TEXT_FORMAT:-bow}" != bow || "${TEXT_ORDER:-3}" != 3 ) ]] && sfx="_${TEXT_FORMAT:-bow}o${TEXT_ORDER:-3}"
            echo "bins $ds $RES/bins${t}_${ds}_k${k}_bs${bs}_dpb${dpb}${sfx} vec=$vec k=$k bs=$bs dpb=$dpb"
        done
    fi
    if has pacmann; then
        for k in 10 100; do for st in 5 10 15 20 25 30; do for nb in 8 16 24 32 40 48; do
            echo "pacmann $d $RES/pacmann_${d}_k${k}_steps${st}_neighb${nb} k=$k steps=$st neighb=$nb"
        done; done; done
    fi
    if has tree; then
        tag=""; [[ "${TREE_CUCKOO_HASHES:-2}" != 2 || "${TREE_CUCKOO_LF:-0.4}" != 0.4 ]] && tag="h${TREE_CUCKOO_HASHES:-2}_lf${TREE_CUCKOO_LF:-0.4}/"
        DATASET="$d" SWEEP="${SWEEP:-all}" bash tree_sweep_grid.sh --list 2>/dev/null |
        while read -r _ ds B r s L q c; do
            B=${B#B=}; r=${r#r=}; s=${s#s=}; L=${L#L=}; q=${q#q=}; c=${c#c=}
            echo "tree $ds $RES/${tag}tree_grid_${ds}_b${B}_r${r}_s${s}_L${L}_q${q}_c${c}_k10 B=$B r=$r s=$s L=$L q=$q c=$c"
        done
    fi
    if has tiptoe; then
        case $d in msmarco) n=8841823;; scifact) n=5183;; trec-covid) n=171332;;
                   *) n=$(cat "$DATASET_DIR/$d/num_docs.txt" 2>/dev/null || echo 0);; esac
        # same cluster rule as preprocess_tiptoe.py; several DPC values can map to one folder, so dedupe
        python3 -I - "$d" "$n" "$BRES" <<'EOF'
import sys
d, n, root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seen = set()
for dpc in (250, 500, 1000, 2500, 5000, 10000, 40000):
    c = min(max(1, round(n / dpc)), 25196) if n else 0
    for ma in (1, 2):
        for b in (40, 80, 160, 320):
            for h in (125, 500):
                for k in (10, 100):
                    name = f"{root}/tiptoe_{d}_c{c}_m{min(ma, c)}_b{b}_h{h}_k{k}"
                    if name not in seen:
                        seen.add(name); print(f"tiptoe {d} {name} dpc={dpc} ma={ma} bundle={b} hint={h} k={k}")
EOF
    fi
    if has coeus; then
        for f in 8192 16384 32768 65536 131072 262144; do for tf in raw log; do for k in 10 100; do
            echo "coeus $d $BRES/coeus_${d}_k${k}_f${f}_tf${tf} F=$f tf=$tf k=$k"
        done; done; done
    fi
done
} > "$OUT/expected.txt"

hr "3. configs per method and dataset (expected = the grid; crashed = folder without metadata.json, or still running)"
python3 -I - "$OUT" <<'EOF'
import json, os, re, sys
from collections import defaultdict
out = sys.argv[1]
ERR = re.compile(r"panic:|Traceback|Error|error|Killed|oom|OUT_OF_MEMORY|TIME LIMIT|CANCELLED|Segmentation|fatal|No such file", re.I)

def last_error(folder):
    for name in ("error.log", "execution.log"):
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            continue
        with open(p, errors="replace") as f:
            lines = [l.strip() for l in f.readlines()[-300:] if l.strip()]
        hits = [l for l in lines if ERR.search(l)]
        if hits or lines:
            return (hits or lines)[-1][:200]
    return "(no log in the folder: task died before writing one, or is still running)"

stats = defaultdict(lambda: defaultdict(int))
files = {}
for line in open(f"{out}/expected.txt"):
    method, ds, folder, *cfg = line.split()
    s = stats[(method, ds)]
    s["expected"] += 1
    meta = os.path.join(folder, "metadata.json")
    key = f"{method}_{ds}"
    if key not in files:
        files[key] = open(f"{out}/{key}_not_done.tsv", "w")
    if os.path.exists(meta) and os.path.getsize(meta) > 0:
        s["done"] += 1
        if method in ("tiptoe", "coeus"):
            try:
                m = json.load(open(meta))
            except Exception:
                s["bad json"] += 1; files[key].write(f"bad_json\t{folder}\t{' '.join(cfg)}\n"); continue
            if method == "tiptoe" and str(m.get("TiptoeSimulated", "")).lower() == "true":
                s["SIMULATED"] += 1; files[key].write(f"simulated\t{folder}\t{' '.join(cfg)}\n")
            if method == "coeus" and "TotalAnswerTime" not in m:
                s["no server time"] += 1; files[key].write(f"no_server_time\t{folder}\t{' '.join(cfg)}\n")
    elif os.path.isdir(folder):
        s["crashed"] += 1
        files[key].write(f"crashed\t{folder}\t{' '.join(cfg)}\t{last_error(folder)}\n")
    else:
        s["not started"] += 1
        files[key].write(f"not_started\t{folder}\t{' '.join(cfg)}\n")

print(f"  {'method':8} {'dataset':15} {'expected':>8} {'done':>6} {'crashed':>8} {'not started':>12}  notes")
for (method, ds), s in sorted(stats.items()):
    notes = ", ".join(f"{s[k]} {k}" for k in ("SIMULATED", "no server time", "bad json") if s[k])
    if method == "coeus" and s["not started"]:
        notes += (", " if notes else "") + "F past the vocabulary writes nothing (expected on small sets)"
    if method == "bins" and s["not started"]:
        notes += (", " if notes else "") + "configs estimated > 1500 GB are never submitted"
    print(f"  {method:8} {ds:15} {s['expected']:>8} {s['done']:>6} {s['crashed']:>8} {s['not started']:>12}  {notes}")
for f in files.values():
    f.close()

# most common last-error lines of the crashed configs
crash = defaultdict(int)
for key in files:
    for l in open(f"{out}/{key}_not_done.tsv"):
        p = l.rstrip("\n").split("\t")
        if p[0] == "crashed":
            crash[(key.split("_")[0], re.sub(r"\d+", "N", p[-1])[:150])] += 1
if crash:
    print("\n  crashed configs, by their last error line (numbers replaced by N):")
    for (m, e), n in sorted(crash.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:6}  {m:8} {e}")
EOF

# ------------------------------------------------------------------------------------------------- 4. errors
hr "4. most common error lines since $SINCE (live logs/*.err + results */error.log; numbers replaced by N)"
{ find logs -maxdepth 1 -name '*.err' -newermt "$SINCE" -size +0 2>/dev/null
  find "$RES" "$BRES" -mindepth 2 -maxdepth 3 -name error.log -newermt "$SINCE" -size +0 2>/dev/null; } | sort -u > "$OUT/error_files.txt"
echo "  $(wc -l < "$OUT/error_files.txt") non-empty error logs (list: $OUT/error_files.txt)"
xargs -r -d '\n' grep -hiE 'panic:|Traceback|^[A-Za-z]*Error|error:|Killed|oom|out of memory|DUE TO TIME LIMIT|CANCELLED|Segmentation|fatal' \
    < "$OUT/error_files.txt" 2>/dev/null \
  | sed -E 's/[0-9]+/N/g' | cut -c1-150 | sort | uniq -c | sort -rn | head -20
echo "  which files: xargs -d '\\n' grep -l '<text>' < $OUT/error_files.txt"

# ------------------------------------------------------------------------------------------------ 5. usage log
hr "5. OOM / timeout entries in ${USAGE_FILE:-$RES/resource_usage.tsv}"
awk -F'\t' -v d=" $DATASETS " '$7 ~ /^failed/ && index(d, " " $2 " ") { n[$1 " " $2 " " $7]++ }
    END { for (k in n) printf "  %-50s %6d\n", k, n[k] }' "${USAGE_FILE:-$RES/resource_usage.tsv}" 2>/dev/null | sort
echo "  (each one is re-estimated at 1.5x memory / 2x time the next time run_all.sh plans that config)"

# ------------------------------------------------------------------------------------------------ 6. fail files
hr "6. fail files in the repo root (tasks that exited without finishing): total, since $SINCE"
for f in bins_low_fails bins_high_fails pacman_fails tree_fails tiptoe_fails tiptoe_v2_fails coeus_fails; do
    [[ -f $f ]] || continue
    printf '  %-18s %6d  %6d since\n' "$f" "$(wc -l < "$f")" "$(awk -v s="$SINCE" '$(NF-1) >= s' "$f" | wc -l)"
done

hr "lists written to $OUT/"
echo "  <method>_<dataset>_not_done.tsv : status  folder  config  [last error]   (status: crashed, not_started, simulated, no_server_time)"
echo "  e.g.  cut -f1 $OUT/bins_nq_not_done.tsv | sort | uniq -c"
echo "        awk -F'\\t' '\$1==\"crashed\"' $OUT/bins_nq_not_done.tsv | cut -f4 | sort | uniq -c | sort -rn | head"
