#!/bin/bash
# Shared helpers for the tree and bins run scripts and datasets/run_all.sh: `source slurm_common.sh`.
#
#   * dataset sizes                     dataset_docs <name>
#   * measured usage log                record_usage / load_usage / measured_usage
#   * memory + time estimate per config estimate_resources <method> <dataset> <config> <fallback GB> <fallback min>
#   * bucket rounding                   mem_bucket <GB>, time_bucket <minutes>, slurm_time <minutes>
#
# Every finished run appends one line to $USAGE_FILE (default $RESULTS_DIR/resource_usage.tsv):
#   method  dataset  config  max_rss_gb  elapsed_s  date  [failed:oom|timeout]
# so estimates become measurements as runs come back. Lookup order for a config:
#   1. measured on this dataset             -> peak RSS x MEM_HEADROOM, elapsed x TIME_HEADROOM
#   2. measured on another dataset          -> scaled by (docs ratio)^exp, with the larger headroom
#                                              (the closest measured dataset by size wins)
#   3. neither                              -> the method's model / fallback, passed in by the caller

USAGE_FILE="${USAGE_FILE:-${RESULTS_DIR:-/scratch/${USER:-$(id -un)}/results}/resource_usage.tsv}"
MEM_HEADROOM="${MEM_HEADROOM:-1.2}"      # measured on the same dataset
TIME_HEADROOM="${TIME_HEADROOM:-1.5}"
SCALED_HEADROOM="${SCALED_HEADROOM:-1.5}" # scaled from another dataset (mem and time)
# how memory / time grow with corpus size, per method (1 = linear)
declare -A MEM_SCALE_EXP=([tree]=1.0 [bins]=0.8)
declare -A TIME_SCALE_EXP=([tree]=1.0 [bins]=1.0)

MEM_BUCKETS="${MEM_BUCKETS:-16 32 64 100 250 500 1000 1500}"   # GB, <=100 runs on the normal partition
NORMAL_MAX_GB="${NORMAL_MAX_GB:-100}"
TIME_BUCKETS="${TIME_BUCKETS:-60 120 240 360 720 1440 2880}"  # minutes

#   name:docs:words_per_doc   (words/doc from the BEIR table in datasets/README.md; used by the tree model)
DATASET_SIZES="msmarco:8841823:56 scifact:5183:214 trec-covid:171332:160 nq:2681468:79 dbpedia-entity:4635922:50
fever:5416568:85 hotpotqa:5233329:46 miracl-en:32893221:100 msmarco-v2:138364198:70 msmarco-v2-25m:25000000:70
msmarco-v2-50m:50000000:70 msmarco-v2-100m:100000000:70"

declare -A _DOCS=()
dataset_docs() {   # name -> number of docs (num_docs.txt if it exists, else the table above)
    if [[ -z "${_DOCS[$1]:-}" ]]; then
        local f="${DATASET_DIR:-../../datasets}/$1/num_docs.txt" e n=0
        if [[ -s "$f" ]]; then n=$(<"$f")
        else for e in $DATASET_SIZES; do [[ "${e%%:*}" == "$1" ]] && { e=${e#*:}; n=${e%%:*}; break; }; done; fi
        _DOCS[$1]=$n
    fi
    echo "${_DOCS[$1]}"
}

mem_bucket() {   # GB (float ok) -> smallest bucket >= it, or "over"
    local b
    for b in $MEM_BUCKETS; do
        awk -v x="$1" -v b="$b" 'BEGIN { exit !(x <= b) }' && { echo "$b"; return; }
    done
    echo over
}

time_bucket() {   # minutes -> smallest bucket >= it (the largest bucket caps it)
    local b last
    for b in $TIME_BUCKETS; do
        last=$b
        awk -v x="$1" -v b="$b" 'BEGIN { exit !(x <= b) }' && { echo "$b"; return; }
    done
    echo "$last"
}

slurm_time() { printf '%d-%02d:%02d:00\n' $(( $1 / 1440 )) $(( $1 % 1440 / 60 )) $(( $1 % 60 )); }

partition_for() { (( $1 <= NORMAL_MAX_GB )) && echo normal || echo bigmem; }

# ------------------------------------------------------------------------------------------ usage log

# record_usage <method> <dataset> <config> <file written by measure_run.sh> <elapsed seconds>
record_usage() {
    local method=$1 dataset=$2 config=$3 tfile=$4 secs=$5 kb=""
    [[ -f "$tfile" ]] && kb=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$tfile")
    [[ -n "$kb" && "$kb" -gt 0 ]] || { echo "record_usage: no peak RSS in $tfile, not recorded" >&2; return 0; }
    mkdir -p "$(dirname "$USAGE_FILE")"
    local line
    line=$(printf '%s\t%s\t%s\t%.2f\t%d\t%s' "$method" "$dataset" "$config" "$(awk -v k="$kb" 'BEGIN{print k/1048576}')" \
        "$secs" "$(date +%F)")
    ( flock 9; echo "$line" >> "$USAGE_FILE" ) 9>>"${USAGE_FILE}.lock"
    echo "usage: $line"
}

# record_failure <method> <dataset> <config> <measure file> <exit code> <elapsed seconds> [timeout]
# A run killed for memory (exit 137/9, or peak RSS >= 90% of the request) or by the time limit is logged as needing
# 1.5x what it was given, so the next submission of that config lands in a bigger bucket instead of failing again.
record_failure() {
    local method=$1 dataset=$2 config=$3 tfile=$4 rc=$5 secs=$6 timeout=${7:-} kb=0
    local req_kb=$(( ${SLURM_MEM_PER_NODE:-0} * 1024 ))
    [[ -f "$tfile" ]] && kb=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$tfile")
    kb=${kb:-0}
    local oom=""
    [[ "$rc" == 137 || "$rc" == 9 ]] && oom=1
    (( req_kb > 0 && kb * 10 >= req_kb * 9 )) && oom=1
    [[ -n "$oom" || -n "$timeout" ]] || return 0
    local gb sec
    gb=$(awk -v k="$kb" -v r="$req_kb" -v o="$oom" 'BEGIN { m = (k > r ? k : r); if (o) m *= 1.5; printf "%.2f", m / 1048576 }')
    sec=$(( timeout ? secs * 2 : secs ))
    local line
    line=$(printf '%s\t%s\t%s\t%s\t%d\t%s\t%s' "$method" "$dataset" "$config" "$gb" "$sec" "$(date +%F)" \
        "failed:${oom:+oom}${timeout:+timeout}")
    mkdir -p "$(dirname "$USAGE_FILE")"
    ( flock 9; echo "$line" >> "$USAGE_FILE" ) 9>>"${USAGE_FILE}.lock"
    echo "usage (failed run, next estimate is bumped): $line"
}

declare -A _USAGE_MEM=() _USAGE_SEC=() _USAGE_DS=()
load_usage() {   # read $USAGE_FILE once; the latest line per (method, dataset, config) wins
    [[ -f "$USAGE_FILE" ]] || return 0
    local m d c gb s _
    while IFS=$'\t' read -r m d c gb s _; do
        [[ -n "$c" && "$gb" != max_rss_gb ]] || continue
        [[ -z "${_USAGE_MEM["$m|$d|$c"]:-}" ]] && _USAGE_DS["$m|$c"]+="$d "
        _USAGE_MEM["$m|$d|$c"]=$gb; _USAGE_SEC["$m|$d|$c"]=$s
    done < "$USAGE_FILE"
}

# estimate_resources <method> <dataset> <config> <fallback GB> <fallback minutes> -> "GB minutes source"
estimate_resources() {
    local m=$1 d=$2 c=$3 fb_gb=$4 fb_min=$5 key="$1|$2|$3"
    if [[ -n "${_USAGE_MEM[$key]:-}" ]]; then
        awk -v g="${_USAGE_MEM[$key]}" -v s="${_USAGE_SEC[$key]}" -v hm="$MEM_HEADROOM" -v ht="$TIME_HEADROOM" \
            'BEGIN { printf "%.1f %d measured\n", g * hm + 1, s * ht / 60 + 10 }'
        return
    fi
    local cands="" od
    for od in ${_USAGE_DS["$m|$c"]:-}; do
        cands+="$od:$(dataset_docs "$od"):${_USAGE_MEM["$m|$od|$c"]}:${_USAGE_SEC["$m|$od|$c"]} "
    done
    if [[ -n "$cands" && "$(dataset_docs "$d")" -gt 0 ]]; then
        # closest measured dataset by size, then scale by (docs ratio)^exp
        awk -v cands="$cands" -v n="$(dataset_docs "$d")" -v em="${MEM_SCALE_EXP[$m]:-1}" \
            -v et="${TIME_SCALE_EXP[$m]:-1}" -v h="$SCALED_HEADROOM" -v fg="$fb_gb" -v fm="$fb_min" '
        BEGIN {
            k = split(cands, C, " "); best = 0
            for (i = 1; i <= k; i++) {
                split(C[i], f, ":"); if (f[2] <= 0) continue
                r = n / f[2]; d = (r < 1) ? 1 / r : r
                if (!best || d < bd) { best = i; bd = d; bn = f[1]; br = r; bg = f[3]; bs = f[4] }
            }
            if (!best) { printf "%s %s model\n", fg, fm; exit }
            printf "%.1f %d scaled-from-%s\n", bg * br ^ em * h + 1, bs * br ^ et * h / 60 + 10, bn
        }'
        return
    fi
    echo "$fb_gb $fb_min model"
}

# ------------------------------------------------------------------------------------------ bins

bins_config_key() { echo "vec$([[ $1 == true ]] && echo 1 || echo 0)_k$2_bs$3_dpb$4"; }   # vec k bs dpb

bins_missing() {   # dataset -> the first missing input of pir_app -t bins (paths as in main.go GetDatasets), or nothing
    local d=$1 root="${DATASET_DIR:-../../datasets}" f
    local -a need
    case $d in
        msmarco) need=("$root/index_marco" "$root/msmarco/corpus.jsonl" "$root/msmarco/queries.dev.small.jsonl"
                       "$root/msmarco/qrels/qrels.dev.tsv" "$root/msmarco/msmarco_corpus_embed_f32.npy"
                       "$root/msmarco/msmarco_query_embed_f32.npy") ;;
        trec-covid) need=("$root/index_trec_covid" "$root/trec-covid/corpus.jsonl" "$root/trec-covid/queries.jsonl"
                       "$root/trec-covid/qrels/test.tsv" "$root/trec-covid/trec_corpus_embed_f32.npy"
                       "$root/trec-covid/trec_query_embed_f32.npy") ;;
        *)       need=("$root/index_$d" "$root/$d/corpus.jsonl" "$root/$d/queries.jsonl" "$root/$d/qrels/test.tsv"
                       "$root/$d/${d}_corpus_embed_f32.npy" "$root/$d/${d}_query_embed_f32.npy") ;;
    esac
    [[ $d == msmarco || $d == scifact || $d == trec-covid ]] || need+=("$root/$d/num_docs.txt")
    for f in "${need[@]}"; do [[ -e "$f" ]] || { echo "$f"; return; }; done
}

# Model until measured: bins keeps every vocab word's top-dpb doc IDs in RAM while building the bins
# (~30 B per ID; ~4x that with -vectors=true, matching where the old high tier started: dpb>=1000 vs >=250),
# plus the corpus vectors twice (fp32 load + PIR copy). Vocab ~2.9M words for msmarco, growing as docs^0.6.
# Calibrated so the worst old low-tier msmarco config (dpb 750) comes out just under 100 GB with headroom (it ran in 120 GB).
# Time: msmarco runs took minutes to 2 h, so 2 h scaled by the RAM estimate, x2 since unmeasured, >= 20 min.
bins_model() {   # dataset vec dpb -> "GB minutes"
    awk -v n="$(dataset_docs "$1")" -v vec="$2" -v dpb="$3" 'BEGIN {
        V = 2.9e6 * (n / 8841823) ^ 0.6; per = (vec == "true") ? 120 : 30
        gb = (V * dpb * per + n * 192 * 4 * 2) / 1e9 * 1.2 + 2
        m = 2 * 120 * gb / 100; if (m < 20) m = 20
        printf "%.0f %d\n", gb, m }'
}

# stdin: "dataset k bs dpb vec" per config -> stdout: the same + "GB minutes source missing" (missing = "-" if ready)
bins_plan() {
    load_usage
    declare -A model=() miss=()
    local d k bs dpb vec
    while read -r d k bs dpb vec; do
        [[ -n "$d" ]] || continue
        [[ -n "${model[$d.$vec.$dpb]:-}" ]] || model[$d.$vec.$dpb]=$(bins_model "$d" "$vec" "$dpb")
        local fb=(${model[$d.$vec.$dpb]})
        local est; est=$(estimate_resources bins "$d" "$(bins_config_key "$vec" "$k" "$bs" "$dpb")" "${fb[0]}" "${fb[1]}")
        local m="-"
        if [[ "${CHECK_INPUTS:-1}" == 1 ]]; then
            [[ -n "${miss[$d]+x}" ]] || miss[$d]=$(bins_missing "$d")
            m=${miss[$d]:--}
        fi
        echo "$d $k $bs $dpb $vec $est $m"
    done
}

# group_plan <plan file> <out dir> <prefix>: split the ready configs into one list per (memory bucket, time bucket).
# Prints "GB minutes count list_file" per group, then "missing <count> <first missing file>" and "over <count>".
group_plan() {
    awk -v out="$2" -v pre="$3" -v mb="$MEM_BUCKETS" -v tb="$TIME_BUCKETS" '
    BEGIN { nm = split(mb, M, " "); nt = split(tb, T, " ") }
    {
        if ($NF != "-") { miss++; if (!first) first = $NF; next }
        g = $(NF-3) + 0; t = $(NF-2) + 0; b = ""
        for (i = 1; i <= nm; i++) if (g <= M[i]) { b = M[i]; break }
        if (b == "") { over++; next }
        tt = T[nt]; for (i = 1; i <= nt; i++) if (t <= T[i]) { tt = T[i]; break }
        f = out "/" pre "_" b "G_" tt "min.tsv"
        if (!(f in n)) { printf "" > f; key[f] = b " " tt }
        line = $1; for (i = 2; i <= NF - 4; i++) line = line " " $i
        print line, $(NF-3), $(NF-2), $(NF-1) >> f; n[f]++
    }
    END {
        for (f in n) print key[f], n[f], f
        if (miss) print "missing", miss, first
        if (over) print "over", over, "-"
    }' "$1" | sort -k1,1n -k2,2n
}
