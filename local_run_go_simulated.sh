#!/bin/bash
# local_run_go_simulated.sh: run every bins (vec1, vec0, text), pacmann and tree config with simulated PIR
# (pir_app -simulate) on this machine, then check the results against the real (Hopper) runs.
#
#   bash local_run_go_simulated.sh plan     what would run, per method and dataset (nothing runs)
#   bash local_run_go_simulated.sh run      build pir_app if needed, run every config that fits in RAM (default)
#   bash local_run_go_simulated.sh check    compare finished simulated runs with the real runs
#
# A simulated run builds no PIR hints and does no XOR, so it needs far less RAM and time than a real one. It still
# loads the DB rows, the embeddings (the client re-ranks with them) and the indexes, because the answers, and so
# MRR/Recall, come from them. Every server timing in its metadata.json is -1 and it has "Simulated": "true".
#
# Knobs (environment):
#   DATASETS_ROOT   the datasets folder (pir_app -dataset). Default: ~/Nextcloud/10TB-STHDD/datasets if it exists,
#                   else ../../../datasets (README 3.2)
#   DATASETS        default "scifact trec-covid nq dbpedia-entity fever msmarco msmarco-v2-25m msmarco-v2-50m msmarco-v2"
#   METHODS         default "bins-vec bins-text pacmann tree" (bins-vec = vec1 and vec0)
#   SIM_RESULTS     where the simulated runs go, default $DATASETS_ROOT/results_simulated (same folder names as Hopper)
#   REAL_RESULTS    the real runs to check against, default $DATASETS_ROOT/results_hopper. Fetch just the small files:
#                     rsync -avm --include='*/' --include='metadata.json' --include='results*.json' --exclude='*' \
#                       hopper:/scratch/dblackle/results/ "$REAL_RESULTS/"
#   MAX_GB          skip configs estimated above this, default 90% of this machine's RAM
#   SIM_MEM_FACTOR  simulated RAM = this x the real-run model in slurm_common.sh, default 0.6 (a guess until measured;
#                   every run logs its peak RSS to $SIM_RESULTS/resource_usage.tsv)
#   NORAM=1         also pass -noram: bins PIR DBs and every .npy embedding file are mmap'd from temp_storage/
#                   (NORAM_DIR) instead of held in RAM. Same answers, slower. Estimates use NORAM_MEM_FACTOR
#                   (default 0.3) instead of SIM_MEM_FACTOR. Needs disk space about the size of the DB
#   JOBS            runs at once, default 1. Above 1, runs only start while their estimates fit in MAX_GB together
#   SWEEP           tree grid, default all (tree_sweep_grid.sh)
#   FORCE=1         rerun configs that already have a metadata.json; RETRY_FAILED=1 retries configs that failed before
#   ONLY            only configs whose folder name matches this regex, e.g. ONLY='_k10_bs1000_' or ONLY='^pacmann'
#   TOL             check: relative tolerance for the non-exact numbers, default 0.02
#
# Tree inputs live in the repo (data/<dataset>_tree/, data/scifact, indexes/scifact, data/msmarco*), not in the
# datasets folder, so rsync them from your Hopper copy of the repo, or build them with datasets/prep_tree.sh.
# Configs with a missing input are skipped and listed by `plan`.
set -uo pipefail
cd "$(dirname "$0")"

MODE="${1:-run}"
case $MODE in plan|run|check) ;; *) sed -n 2,33p "$0"; exit 1 ;; esac

if [[ -z "${DATASETS_ROOT:-}" ]]; then
    for c in "$HOME/Nextcloud/10TB-STHDD/datasets" ../../../datasets; do [[ -d "$c" ]] && { DATASETS_ROOT=$c; break; }; done
fi
[[ -d "${DATASETS_ROOT:-}" ]] || { echo "No datasets folder found, set DATASETS_ROOT" >&2; exit 1; }
DATASETS_ROOT=$(cd "$DATASETS_ROOT" && pwd)
export DATASET_DIR="$DATASETS_ROOT"
SIM_RESULTS="${SIM_RESULTS:-$DATASETS_ROOT/results_simulated}"
REAL_RESULTS="${REAL_RESULTS:-$DATASETS_ROOT/results_hopper}"
DATASETS="${DATASETS:-scifact trec-covid nq dbpedia-entity fever msmarco msmarco-v2-25m msmarco-v2-50m msmarco-v2}"
METHODS="${METHODS:-bins-vec bins-text pacmann tree}"
JOBS="${JOBS:-1}"
SIM_MEM_FACTOR="${SIM_MEM_FACTOR:-0.6}"
EXTRA_ARGS="-simulate"
if [[ -n "${NORAM:-}" ]]; then
    SIM_MEM_FACTOR="${NORAM_MEM_FACTOR:-0.3}"
    EXTRA_ARGS="-simulate -noram -noram-dir ${NORAM_DIR:-$PWD/temp_storage}"
fi
MAX_GB="${MAX_GB:-$(awk '/^MemTotal:/ { printf "%d", $2 * 0.9 / 1048576 }' /proc/meminfo)}"
mkdir -p "$SIM_RESULTS"
case "$DATASETS_ROOT$SIM_RESULTS$PWD" in *" "*) echo "Paths with spaces are not supported" >&2; exit 1 ;; esac
echo "datasets: $DATASETS_ROOT"
echo "simulated results: $SIM_RESULTS"

# ============================================================================================== check
check() {
    echo "real results: $REAL_RESULTS"
    [[ -d "$REAL_RESULTS" ]] || { echo "No $REAL_RESULTS (see REAL_RESULTS at the top of this file)"; return 1; }
    SIM="$SIM_RESULTS" REAL="$REAL_RESULTS" TOL="${TOL:-0.02}" python3 - <<'PY'
import json, os, sys

sim_root, real_root, tol = os.environ["SIM"], os.environ["REAL"], float(os.environ["TOL"])
# identical in a simulated run by construction (same answers)
EXACT = ["MRR", "MRRPreReRank", "Recall", "RecallPreReRank", "NumQueries"]

def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def skip(k):
    if k.startswith("Pre") or k in ("Simulated", "noram", "NoRAMPeakAnonMB"):
        return True
    timing = "Time" in k or "_seconds" in k
    return timing and not ("LAN" in k or "WAN" in k)   # timings are -1 in simulated runs

rows, n_ok, n_diff, n_noreal = [], 0, 0, 0
for dirpath, _, files in sorted(os.walk(sim_root)):
    if "metadata.json" not in files:
        continue
    rel = os.path.relpath(dirpath, sim_root)
    sm = json.load(open(os.path.join(dirpath, "metadata.json")))
    if sm.get("Simulated") != "true":
        continue
    real_dir = os.path.join(real_root, rel)
    if not os.path.isfile(os.path.join(real_dir, "metadata.json")):
        n_noreal += 1
        continue
    rm = json.load(open(os.path.join(real_dir, "metadata.json")))
    exact_bad, approx_bad = [], []
    for k in EXACT:
        if k in sm and k in rm and sm[k] != rm[k]:
            exact_bad.append(f"{k} real={rm[k]} sim={sm[k]}")
    for f in ("results.json", "results_reRank.json"):
        a, b = os.path.join(real_dir, f), os.path.join(dirpath, f)
        if os.path.isfile(a) and os.path.isfile(b) and json.load(open(a)) != json.load(open(b)):
            exact_bad.append(f"{f} answers differ")
    for k in sorted(set(sm) & set(rm)):
        if k in EXACT or skip(k):
            continue
        a, b = num(rm[k]), num(sm[k])
        if a is None or b is None:
            if rm[k] != sm[k]:
                approx_bad.append(f"{k} real={rm[k]} sim={sm[k]}")
            continue
        if abs(a - b) > tol * max(abs(a), abs(b), 1e-12):
            approx_bad.append(f"{k} real={rm[k]} sim={sm[k]} ({100 * abs(a - b) / max(abs(a), abs(b)):.1f}%)")
    if exact_bad or approx_bad:
        n_diff += 1
        rows.append((rel, exact_bad, approx_bad))
    else:
        n_ok += 1

for rel, ex, ap in rows:
    print(f"DIFF {rel}")
    for x in ex:
        print(f"   EXACT  {x}")
    for x in ap:
        print(f"   >{tol:g}  {x}")
print(f"\n{n_ok} match, {n_diff} differ, {n_noreal} simulated runs have no real run to compare with")
print("MRR/Recall/answers must match exactly. Bytes, LAN/WAN and client storage of the bins vary by ~0.5% between")
print("two real runs too. A difference can also mean the real run was made by older code (git log on Hopper).")
sys.exit(1 if any(ex for _, ex, _ in rows) else 0)
PY
}
if [[ $MODE == check ]]; then check; exit $?; fi

# ============================================================================================== plan
export MEASURE_INTERVAL="${MEASURE_INTERVAL:-1}"   # peak RSS polling when GNU time is missing (measure_run.sh)
export RESULTS_DIR="$SIM_RESULTS" USAGE_FILE="$SIM_RESULTS/resource_usage.tsv" ROOT_DIR="$PWD"
source ./tree_sweep_grid.sh     # sources slurm_common.sh: dataset_docs, bins_model, bins_missing, tree_* helpers
has_method() { [[ " $METHODS " == *" $1 "* ]]; }
sim_gb() { awk -v g="$1" -v f="$SIM_MEM_FACTOR" 'BEGIN { x = g * f; printf "%d", (x == int(x)) ? x : int(x) + 1 }'; }

PLAN="$SIM_RESULTS/plan.tsv"; SKIPS="$SIM_RESULTS/skipped.tsv"
: > "$PLAN"; : > "$SKIPS"
declare -A FAILED=()
if [[ -f "$SIM_RESULTS/failed.tsv" && -z "${RETRY_FAILED:-}" ]]; then
    while IFS=$'\t' read -r dir _; do FAILED[$dir]=1; done < "$SIM_RESULTS/failed.tsv"
fi

# add <est GB> <method> <dataset> <out dir> <command...>: queue a config, or note why it is skipped
add() {
    local gb=$1 method=$2 d=$3 od=$4; shift 4
    if [[ -z "${FORCE:-}" && -s "$od/metadata.json" ]]; then printf '%s\t%s\tdone\n' "$method" "$d" >> "$SKIPS"; return; fi
    if [[ -n "${FAILED[$od]:-}" ]]; then printf '%s\t%s\tfailed before (RETRY_FAILED=1)\n' "$method" "$d" >> "$SKIPS"; return; fi
    if [[ -n "${ONLY:-}" ]] && ! [[ "$(basename "$od")" =~ $ONLY ]]; then return; fi
    if (( gb > MAX_GB )); then printf '%s\t%s\tover %s GB\n' "$method" "$d" "$MAX_GB" >> "$SKIPS"; return; fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$gb" "$method" "$d" "$od" "$*" >> "$PLAN"
}
pacmann_missing() {   # dataset -> first missing input of pir_app -t pacmann (paths as in main.go GetDatasets)
    local d=$1 r=$DATASET_DIR f pfx=$1 q="$1/queries.jsonl" qr="$1/qrels/test.tsv"
    case $d in
        msmarco) q=msmarco/queries.dev.small.jsonl; qr=msmarco/qrels/qrels.dev.tsv ;;
        trec-covid) pfx=trec ;;
    esac
    for f in "$d/${pfx}_corpus_embed_f32.npy" "$d/${pfx}_query_embed_f32.npy" "$q" "$qr"; do
        [[ -e "$r/$f" ]] || { echo "$r/$f"; return; }
    done
    [[ $d == msmarco || $d == scifact || $d == trec-covid || -e "$r/$d/num_docs.txt" ]] || echo "$r/$d/num_docs.txt"
}
missing() { printf '%s\t%s\tmissing %s\n' "$1" "$2" "$3" >> "$SKIPS"; }

for d in $DATASETS; do
    n=$(dataset_docs "$d")
    # ---------------------------------------------------------------- bins: the grids of run_bins_low/high.slurm
    vecs=""
    has_method bins-vec && vecs="true false"
    has_method bins-text && vecs="$vecs text"
    if [[ -n "$vecs" ]]; then
        while read -r dd k bs dpb vec; do
            lname=$(case $vec in true) echo bins-vec1 ;; false) echo bins-vec0 ;; *) echo bins-text ;; esac)
            m=$(bins_missing "$dd" "$vec")
            [[ -z "$m" ]] || { missing "$lname" "$dd" "$m"; continue; }
            read -r gb _ <<< "$(bins_model "$dd" "$vec" "$dpb" "$bs")"
            if [[ $vec == text ]]; then layout="-text -text-format ${TEXT_FORMAT:-bow} -text-order ${TEXT_ORDER:-3}"
            else layout="-vectors=$vec"; fi
            add "$(sim_gb "$gb")" "$lname" "$dd" "$(bins_out_dir "$dd" "$k" "$bs" "$dpb" "$vec")" \
                ./pir_app -n "$n" -t bins -name "$dd" -k "$k" -binSize "$bs" -docsPerBin "$dpb" \
                -dataset "$DATASET_DIR" -debug 1 $layout -dim 192
        done < <(for tier in low high; do
                     # the slurm scripts call `module` before listing; a no-op stands in for it here
                     BINS_DATASETS="$d" BINS_VECS="$vecs" \
                         bash -c 'module() { :; }; export -f module; bash "$0" --list-raw' "run_bins_$tier.slurm"
                 done 2>/dev/null)
    fi

    # ---------------------------------------------------------------- pacmann: the grid of run_pacmann.slurm
    if has_method pacmann; then
        m=$(pacmann_missing "$d")
        if [[ -n "$m" ]]; then missing pacmann "$d" "$m"
        else
            for k in 10 100; do for st in 5 10 15 20 25 30; do for nb in 8 16 24 32 40 48; do
                # rough model: embeddings + the NGT graph build (several copies of the vectors) + 2 GB
                gb=$(awk -v n="$n" -v nb="$nb" 'BEGIN { printf "%d", n * (768 * 4 + nb * 16) / 1e9 + 3 }')
                add "$(sim_gb "$gb")" pacmann "$d" "$SIM_RESULTS/pacmann_${d}_k${k}_steps${st}_neighb${nb}" \
                    ./pir_app -n "$n" -t pacmann -name "$d" -k "$k" -steps "$st" -neighb "$nb" \
                    -dataset "$DATASET_DIR" -debug 1 -dim 192
            done; done; done
        fi
    fi

    # ---------------------------------------------------------------- tree: the grid of tree_sweep_grid.sh
    if has_method tree && [[ " $TREE_DATASETS " == *" $d "* ]]; then
        DATA_ARR=(); L_ARR=(); KCAND_ARR=(); QWORDS_ARR=(); B_ARR=(); R_ARR=(); S_ARR=(); MAX_Q_ARR=(); _TREE_SEEN=()
        build_tree_grid "$d" "${SWEEP:-all}"
        declare -A tmiss=()
        for i in "${!DATA_ARR[@]}"; do
            B=${B_ARR[$i]} r=${R_ARR[$i]} s=${S_ARR[$i]} L=${L_ARR[$i]} q=${QWORDS_ARR[$i]} c=${KCAND_ARR[$i]}
            [[ -n "${tmiss[$B.$r.$s]+x}" ]] || tmiss[$B.$r.$s]=$(tree_missing "$d" "$B" "$r" "$s")
            [[ -z "${tmiss[$B.$r.$s]}" ]] || { missing tree "$d" "${tmiss[$B.$r.$s]}"; continue; }
            tree_paths "$d" "$B" "$r" "$s"
            read -r ch lf <<< "$(tree_cuckoo "$d")"
            kicks=$([[ "$ch" == 3 ]] && echo 5000 || echo 1000)
            od="$SIM_RESULTS/$(tree_cuckoo_tag "$d")tree_grid_${d}_b${B}_r${r}_s${s}_L${L}_q${q}_c${c}_k10"
            cache=""; [[ -n "$CACHE_SUBDIR" ]] && cache="--cuckoo-cache-dir $PWD/cuckoo_cache/$CACHE_SUBDIR/sim_L${L}_q${q}_c${c}"
            add "$(sim_gb "$(tree_mem_gb "$d" "$B" "$r" "$s")")" tree "$d" "$od" \
                ./pir_app -t tree -name "$d" -n "$DB_SIZE" -dim 192 -dataset "$DATASET_DIR" \
                --L "$L" --k_candidates "$c" --query-words "$q" --B "$B" --r "$r" --s "$s" --k 10 \
                --max-queries "${MAX_Q_ARR[$i]}" --k1 0.9 --b 0.4 --scale 10 --precompute --pir-batch-size 128 \
                --cuckoo-load-factor "$lf" --cuckoo-hashes "$ch" --cuckoo-max-kicks "$kicks" --cuckoo-build-attempts 8 \
                --layout-path "$LAYOUT_PATH" --index "$INDEX" --queries "$QUERIES" --qrels "$QRELS" --vocab "$VOCAB" \
                --stage1-data-bin "$STAGE1_DATA" --stage1-idmap-bin "$STAGE1_IDMAP" \
                --stage2-data-bin "$STAGE2_DATA" --stage2-idmap-bin "$STAGE2_IDMAP" \
                --doc-embed "$DOC_EMBED" --doc-id-map "$DOC_ID_MAP" --query-embed "$QUERY_EMBED" --embed-dim 192 \
                ${BLOCK_PERMS:+--block-permutation "$BLOCK_PERMS"} --pir-threads "$(nproc)" $cache
        done
        unset tmiss
    fi
done

sort -n -k1,1 -o "$PLAN" "$PLAN"   # smallest first: the most results soonest
echo
echo "== to run (estimated RAM, MAX_GB=$MAX_GB, memory factor $SIM_MEM_FACTOR, flags: $EXTRA_ARGS)"
awk -F'\t' '{ n[$2 " " $3]++; g[$2 " " $3] = $1 } END { for (k in n) printf "  %-12s %-16s %5d configs, largest ~%d GB\n", substr(k, 1, index(k, " ") - 1), substr(k, index(k, " ") + 1), n[k], g[k] }' "$PLAN" | sort
echo "== skipped"
awk -F'\t' '{ r = $3; if (r ~ /^missing/) r = "missing input"; n[$1 " " $2 " " r]++; if (!(($1 " " $2 " " r) in ex)) ex[$1 " " $2 " " r] = $3 }
     END { for (k in n) printf "  %-40s %5d  (%s)\n", k, n[k], ex[k] }' "$SKIPS" | sort
echo "($(wc -l < "$PLAN") to run; details in $PLAN and $SKIPS)"
[[ $MODE == plan ]] && exit 0

# ============================================================================================== run
if ! { [[ -x ./pir_app ]] && ./pir_app -h 2>&1 | grep -q -- '-simulate'; }; then
    echo "Building pir_app (needs the -simulate flag)"
    [[ -f "$HOME/opt/pacmann_env.sh" ]] && source "$HOME/opt/pacmann_env.sh"
    go build -o pir_app ./main.go || exit 1
fi
[[ -f "$HOME/opt/pacmann_env.sh" ]] && source "$HOME/opt/pacmann_env.sh"   # LD_LIBRARY_PATH for libngt/libhnsw

run_one() {   # <est GB> <method> <dataset> <out dir> <command line>
    local gb=$1 method=$2 d=$3 od=$4 cmd=$5 rc start=$SECONDS
    local -a argv; read -ra argv <<< "$cmd"
    mkdir -p "$od"
    echo "$(date '+%F %T') start $method $d ($gb GB est) -> $od"
    bash ./measure_run.sh "$od/time_v.txt" "${argv[@]}" $EXTRA_ARGS -outDir "$od" -outFile results \
        > "$od/execution.log" 2>&1
    rc=$?
    local kb; kb=$(awk -F': ' '/Maximum resident set size/ { print $2 }' "$od/time_v.txt" 2>/dev/null)
    local peak; peak=$(awk -v k="${kb:-0}" 'BEGIN { printf "%.2f", k / 1048576 }')
    if (( rc == 0 )) && [[ -s "$od/metadata.json" ]]; then
        printf '%s\t%s\t%s\t%s\t%d\t%s\n' "sim-$method" "$d" "$(basename "$od")" "$peak" $(( SECONDS - start )) "$(date +%F)" \
            >> "$USAGE_FILE"
        echo "$(date '+%F %T') done  $method $d: peak ${peak} GB (est $gb), $(( SECONDS - start )) s"
    else
        printf '%s\trc=%s\tpeak=%sGB\test=%sGB\t%s\n' "$od" "$rc" "$peak" "$gb" "$(date '+%F %T')" >> "$SIM_RESULTS/failed.tsv"
        echo "$(date '+%F %T') FAIL  $method $d rc=$rc peak ${peak} GB, see $od/execution.log"
    fi
    if [[ $method == pacmann ]]; then   # this config's NGT graph, as run_pacmann.slurm cleans it up
        local k st nb n=${argv[2]} name; name=$(basename "$od")
        k=${name##*_k}; k=${k%%_*}; st=${name##*_steps}; st=${st%%_*}; nb=${name##*_neighb}
        local prefix="$DATASET_DIR/$d/${d}_corpus_embed_f32_${n}_192_${nb}_neigh${nb}_step${st}_k${k}"
        rm -rf -- "${prefix}.ngt" "${prefix}_graph.npy"
    fi
    if [[ $method == tree && $cmd == *--cuckoo-cache-dir* ]]; then rm -rf -- "${cmd##*--cuckoo-cache-dir }"; fi
    return 0
}

declare -A PID_GB=()
running_gb=0
while IFS=$'\t' read -r gb method d od cmd; do
    if (( JOBS <= 1 )); then run_one "$gb" "$method" "$d" "$od" "$cmd"; continue; fi
    while (( ${#PID_GB[@]} >= JOBS || ( ${#PID_GB[@]} > 0 && running_gb + gb > MAX_GB ) )); do
        wait -n -p done_pid
        running_gb=$(( running_gb - ${PID_GB[$done_pid]:-0} )); unset "PID_GB[$done_pid]"
    done
    run_one "$gb" "$method" "$d" "$od" "$cmd" &
    PID_GB[$!]=$gb; running_gb=$(( running_gb + gb ))
done < "$PLAN"
wait

echo
echo "Finished. Failures (if any): $SIM_RESULTS/failed.tsv; peak RAM per run: $USAGE_FILE"
[[ -d "$REAL_RESULTS" ]] && check
