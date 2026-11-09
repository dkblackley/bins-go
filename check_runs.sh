#!/bin/bash
# Status of the bins / pacmann / tree / tiptoe / coeus runs. Read-only: never submits, cancels, moves or deletes.
#
#   bash check_runs.sh                              # scifact msmarco nq fever dbpedia-entity, every method
#   bash check_runs.sh nq fever                     # only these datasets
#   METHODS="bins tree" SINCE=2026-11-01 bash check_runs.sh nq
#   bash check_runs.sh --mark                       # from now on only show failures after this moment
#
# "Since": sections 2, 4, 5 and 6 only show failures after the time in logs/check/since. run_me.sh writes it every
# time it deploys a fix and resubmits, so failures that are already fixed drop out. No file: the last 14 days.
# SINCE=<date or date-time> overrides it for one run (SINCE=2026-10-01 to see everything again).
#
# Sections:
#   1. queue        your jobs in squeue, per job name / state / pending reason
#   2. failed jobs  sacct, jobs that ended since $SINCE: FAILED, OUT_OF_MEMORY, TIMEOUT, CANCELLED, ...
#   3. configs      every config each method's grid should produce, per dataset: done (metadata.json) /
#                   queued (a pending or running job will write it) / crashed (folder but no metadata.json and
#                   nothing queued for it) / not started. "crashed" + "not started" is what run_me.sh resubmits.
#                   Tiptoe and Coeus: quality + communication only (no server-time benchmark), so no extra flags.
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
SINCE_FILE="${SINCE_FILE:-logs/check/since}"
if [[ "${1:-}" == "--mark" ]]; then
    mkdir -p "$(dirname "$SINCE_FILE")"; date +%FT%T > "$SINCE_FILE"
    echo "check_runs.sh will now only show failures after $(cat "$SINCE_FILE") ($SINCE_FILE)"; exit 0
fi
if [[ -z "${SINCE:-}" ]]; then
    if [[ -s "$SINCE_FILE" ]]; then SINCE=$(<"$SINCE_FILE"); else SINCE=$(date -d '-14 days' +%F); fi
fi
SINCE_SP="${SINCE/T/ }"   # "2026-11-09 05:12:00" for find -newermt and the fail files
METHODS="${METHODS:-bins pacmann tree tiptoe coeus}"
DATASETS="${*:-scifact msmarco nq fever dbpedia-entity}"
OUT="logs/check/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
echo "Failures since $SINCE ($([[ -s "$SINCE_FILE" ]] && echo "set by run_me.sh / --mark in $SINCE_FILE" || echo "default: 14 days")). SINCE=... to change it."
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
hr "2. jobs that ended without completing since $SINCE (sacct; ALL_JOBS=1 also shows other projects' jobs)"
JOBS_RE='^(bins|pacmann|tree|treedb|binsidx|prep|emb|pca|merge|import|Bins|Pacmann|Tree|Tiptoe|Coeus)'
[[ -n "${ALL_JOBS:-}" ]] && JOBS_RE='.'
sacct -u "$USER" -S "$SINCE" -X -n -P -o JobID,JobName,State,ExitCode,End 2>/dev/null \
  | awk -F'|' -v re="$JOBS_RE" -v since="$SINCE" '$2 ~ re { st = $3; sub(/ .*/, "", st)
        if (st ~ /^(COMPLETED|RUNNING|PENDING|REQUEUED|RESIZING)$/) next
        if ($5 < since) next   # ended before the last fix (run_me.sh / --mark)
        k = $2 "|" st; n[k]++; if ($5 >= last[k]) { last[k] = $5; id[k] = $1 } }
      END { for (k in n) { split(k, a, "|"); printf "%s|%s|%d|%s|%s\n", a[1], a[2], n[k], id[k], last[k] } }' \
  | sort > "$OUT/failed_jobs.txt"
if [[ ! -s "$OUT/failed_jobs.txt" ]]; then
    echo "  none (or sacct not available)"
else
    printf '  %-34s %-14s %6s  %-16s %s\n' "job name" "state" "count" "newest" "ended"
    while IFS='|' read -r name st n id end; do
        printf '  %-34s %-14s %6d  %-16s %s\n' "$name" "$st" "$n" "$id" "$end"
    done < "$OUT/failed_jobs.txt"
    echo "  CANCELLED is normally you (scancel) or a dependency that failed; FAILED / OUT_OF_MEMORY / TIMEOUT are real."
    echo
    echo "  last lines of the newest FAILED / OUT_OF_MEMORY / TIMEOUT task of each group (its log in logs/):"
    while IFS='|' read -r name st n id end; do
        [[ $st == FAILED || $st == OUT_OF_MEMORY || $st == TIMEOUT ]] || continue
        case $name in bins_*) pre=Bins;; pacmann_*) pre=Pacmann;; tree_*) pre=Tree;; Tiptoe*) pre=Tiptoe;;
                      Coeus*) pre=Coeus;; *) continue;; esac
        [[ $id == *_* && $id != *\[* ]] || continue
        echo "  -- $name $st  (newest: $id, ended $end)"
        found=""
        for f in "logs/${pre}_${id}.err" "logs/${pre}_${id}.out"; do
            [[ -s $f ]] || continue; found=1
            tail -c 65536 "$f" | grep -vE '^\s*$|record_usage' | tail -n 6 | cut -c1-220 | sed "s|^|     ${f##*/}: |"
        done
        [[ -n $found ]] || echo "     (no logs/${pre}_${id}.{out,err}: it finished its cleanup, look in its results folder's error.log)"
    done < "$OUT/failed_jobs.txt"
fi
echo "  more on one task: sacct -j <id> -o JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed,Timelimit ; seff <id> ;"
echo "                    less logs/<Bins|Pacmann|Tree|Tiptoe|Coeus>_<id>.err"
echo "  Was it fixed since? Compare 'ended' with the last code change: git log -3 --format='%h %ci %s'"

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

# result folders a pending / running job will write (slurm_common.sh queued_dirs; run_me.sh passes QUEUED_FILE)
if [[ -n "${QUEUED_FILE:-}" && -f "$QUEUED_FILE" ]]; then cp "$QUEUED_FILE" "$OUT/queued.tsv"
else ( RESULTS_DIR="$RES"; source ./slurm_common.sh; queued_dirs ) > "$OUT/queued.tsv" 2>/dev/null; fi

hr "3. configs per method and dataset (expected = the grid; queued = a pending/running job covers it; crashed = folder without metadata.json, nothing queued)"
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
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 65536))
            lines = [l.strip() for l in f.read().decode("utf-8", "replace").splitlines()
                     if l.strip() and not l.startswith("record_usage")]
        hits = [l for l in lines if ERR.search(l)]
        if hits or lines:
            return (hits or lines)[-1][:200]
    return "(no log in the folder: it died before its cleanup, its log is still in logs/, see section 2)"

stats = defaultdict(lambda: defaultdict(int))
files = {}
queued, blocked = set(), set()
for l in open(f"{out}/queued.tsv"):
    p = l.rstrip("\n").split("\t")
    if p[0] == "?":
        blocked.add((p[1], p[2]))
    elif p[0]:
        queued.add(p[0])
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
            # simulated Tiptoe and Coeus without server time are what we want (quality + bytes / WAN only)
    elif folder in queued or (method, ds) in blocked:
        s["queued"] += 1
        files[key].write(f"queued\t{folder}\t{' '.join(cfg)}\n")
    elif os.path.isdir(folder):
        s["crashed"] += 1
        files[key].write(f"crashed\t{folder}\t{' '.join(cfg)}\t{last_error(folder)}\n")
    else:
        s["not started"] += 1
        files[key].write(f"not_started\t{folder}\t{' '.join(cfg)}\n")

print(f"  {'method':8} {'dataset':15} {'expected':>8} {'done':>6} {'queued':>7} {'crashed':>8} {'not started':>12}  notes")
for (method, ds), s in sorted(stats.items()):
    notes = f"{s['bad json']} bad json" if s["bad json"] else ""
    if method == "coeus" and s["not started"]:
        notes += (", " if notes else "") + "F past the vocabulary writes nothing (expected on small sets)"
    if method == "bins" and s["not started"]:
        notes += (", " if notes else "") + "configs estimated > 1500 GB are never submitted"
    if (method, ds) in blocked:
        notes += (", " if notes else "") + "queued array with unknown configs: every unfinished one counted as queued"
    print(f"  {method:8} {ds:15} {s['expected']:>8} {s['done']:>6} {s['queued']:>7} {s['crashed']:>8} {s['not started']:>12}  {notes}")
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
{ find logs -maxdepth 1 -name '*.err' -newermt "$SINCE_SP" -size +0 2>/dev/null
  find "$RES" "$BRES" -mindepth 2 -maxdepth 3 -name error.log -newermt "$SINCE_SP" -size +0 2>/dev/null; } | sort -u > "$OUT/error_files.txt"
echo "  $(wc -l < "$OUT/error_files.txt") non-empty error logs (list: $OUT/error_files.txt)"
while IFS= read -r f; do tail -c 65536 "$f" 2>/dev/null; echo; done < "$OUT/error_files.txt" \
  | grep -aiE 'panic:|Traceback|^[A-Za-z]*Error|error:|Killed|oom|out of memory|DUE TO TIME LIMIT|CANCELLED|Segmentation|fatal' \
  | sed -E 's/[0-9]+/N/g' | cut -c1-150 | sort | uniq -c | sort -rn | head -20
echo "  which files: xargs -d '\\n' grep -l '<text>' < $OUT/error_files.txt | head"

# ------------------------------------------------------------------------------------------------ 5. usage log
hr "5. OOM / timeout entries in ${USAGE_FILE:-$RES/resource_usage.tsv} since ${SINCE%%T*}"
awk -F'\t' -v d=" $DATASETS " -v s="${SINCE%%T*}" '$7 ~ /^failed/ && $6 >= s && index(d, " " $2 " ") { n[$1 " " $2 " " $7]++ }
    END { for (k in n) printf "  %-50s %6d\n", k, n[k] }' "${USAGE_FILE:-$RES/resource_usage.tsv}" 2>/dev/null | sort
echo "  (each one is re-estimated at 1.5x memory / 2x time the next time run_all.sh plans that config)"

# ------------------------------------------------------------------------------------------------ 6. fail files
hr "6. fail files in the repo root (tasks that exited without finishing): total, since $SINCE"
for f in bins_low_fails bins_high_fails pacman_fails tree_fails tiptoe_fails tiptoe_v2_fails coeus_fails; do
    [[ -f $f ]] || continue
    printf '  %-18s %6d  %6d since\n' "$f" "$(wc -l < "$f")" "$(awk -v s="$SINCE_SP" '($(NF-1) " " $NF) >= s' "$f" | wc -l)"
done

hr "lists written to $OUT/"
echo "  <method>_<dataset>_not_done.tsv : status  folder  config  [last error]   (status: crashed, queued, not_started, bad_json)"
echo "  e.g.  cut -f1 $OUT/bins_nq_not_done.tsv | sort | uniq -c"
echo "        awk -F'\\t' '\$1==\"crashed\"' $OUT/bins_nq_not_done.tsv | cut -f4 | sort | uniq -c | sort -rn | head"
echo "  After a fix: bash run_me.sh   (pulls, rebuilds pir_app if needed, resubmits crashed + not started, moves 'since')"
