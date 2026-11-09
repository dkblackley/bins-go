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

# GB: 5G steps up to 100 (normal partition), then 50G steps up to 1500 (bigmem), so a request is at most 5G / 50G above
# its estimate (it used to round 260G up to 500G)
MEM_BUCKETS="${MEM_BUCKETS:-$(seq -s' ' 5 5 100) $(seq -s' ' 150 50 1500)}"
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
    if [[ -n "$timeout" ]]; then sec=$(( secs * 2 )); else sec=$secs; fi   # not $(( timeout ? .. )): bash would expand the word "timeout" recursively
    local line
    line=$(printf '%s\t%s\t%s\t%s\t%d\t%s\t%s' "$method" "$dataset" "$config" "$gb" "$sec" "$(date +%F)" \
        "failed:${oom:+oom}${timeout:+timeout}")
    mkdir -p "$(dirname "$USAGE_FILE")"
    ( flock 9; echo "$line" >> "$USAGE_FILE" ) 9>>"${USAGE_FILE}.lock"
    echo "usage (failed run, next estimate is bumped): $line"
}

declare -A _USAGE_MEM=() _USAGE_SEC=() _USAGE_DS=() _USAGE_FAIL=()
load_usage() {   # read $USAGE_FILE once; the latest line per (method, dataset, config) wins
    [[ -f "$USAGE_FILE" ]] || return 0
    local m d c gb s _date st
    while IFS=$'\t' read -r m d c gb s _date st; do
        [[ -n "$c" && "$gb" != max_rss_gb ]] || continue
        [[ -z "${_USAGE_MEM["$m|$d|$c"]:-}" ]] && _USAGE_DS["$m|$c"]+="$d "
        _USAGE_MEM["$m|$d|$c"]=$gb; _USAGE_SEC["$m|$d|$c"]=$s
        if [[ "$st" == failed* ]]; then _USAGE_FAIL["$m|$d|$c"]=1; else unset '_USAGE_FAIL["$m|$d|$c"]'; fi
    done < "$USAGE_FILE"
}

# estimate_resources <method> <dataset> <config> <fallback GB> <fallback minutes> -> "GB minutes source"
estimate_resources() {
    local m=$1 d=$2 c=$3 fb_gb=$4 fb_min=$5 key="$1|$2|$3"
    # A failed run (OOM / time limit) only tells us a lower bound: its elapsed time is how far it got before it was
    # killed, so it never lowers the estimate below the model (e.g. an OOM after 2 min must not ask for 13 min).
    if [[ -n "${_USAGE_MEM[$key]:-}" ]]; then
        awk -v g="${_USAGE_MEM[$key]}" -v s="${_USAGE_SEC[$key]}" -v hm="$MEM_HEADROOM" -v ht="$TIME_HEADROOM" \
            -v f="${_USAGE_FAIL[$key]:-}" -v fg="$fb_gb" -v fm="$fb_min" '
            BEGIN { g = g * hm + 1; t = s * ht / 60 + 10
                    if (f) { if (fg > g) g = fg; if (fm > t) t = fm; printf "%.1f %d failed-bumped\n", g, t }
                    else printf "%.1f %d measured\n", g, t }'
        return
    fi
    local cands="" od
    for od in ${_USAGE_DS["$m|$c"]:-}; do
        cands+="$od:$(dataset_docs "$od"):${_USAGE_MEM["$m|$od|$c"]}:${_USAGE_SEC["$m|$od|$c"]}:${_USAGE_FAIL["$m|$od|$c"]:-0} "
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
                if (!best || d < bd) { best = i; bd = d; bn = f[1]; br = r; bg = f[3]; bs = f[4]; bf = f[5] }
            }
            if (!best) { printf "%s %s model\n", fg, fm; exit }
            g = bg * br ^ em * h + 1; t = bs * br ^ et * h / 60 + 10
            if (bf + 0) { if (fg > g) g = fg; if (fm > t) t = fm }   # scaled from a failed run: a lower bound only
            printf "%.1f %d scaled-from-%s\n", g, t, bn
        }'
        return
    fi
    echo "$fb_gb $fb_min model"
}

# ------------------------------------------------------------------------------------------ tree msmarco layout
# msmarco's tree normally uses the collaborator's original files (data/msmarco_reordered, data/stage3_*). On a machine
# where they are incomplete (no data/stage3_query.npy), it is built like every new dataset instead, in data/msmarco_tree
# (datasets/prep_tree.sh: own Lucene index, stage 3 from ../datasets/msmarco's embeddings). TREE_MSMARCO=original|
# generic forces one. prep_tree.sh and tree_paths both use this, so they always agree.
tree_msmarco_layout() {
    case "${TREE_MSMARCO:-}" in original|generic) echo "$TREE_MSMARCO"; return ;; esac
    if [[ -e "${ROOT_DIR:-$PWD}/data/stage3_query.npy" ]]; then echo original; else echo generic; fi
}

# ------------------------------------------------------------------------------------------ tree Cuckoo settings
# msmarco-v2 (138M) needs the smaller Cuckoo tables (3 hashes, load factor 0.88, ~half of Stage 1/2), so that is its
# default; every other dataset uses 2 / 0.4. TREE_CUCKOO_HASHES / TREE_CUCKOO_LF override it for every dataset.
tree_cuckoo() {   # dataset -> "hashes load_factor"
    local h="${TREE_CUCKOO_HASHES:-}" lf="${TREE_CUCKOO_LF:-}"
    if [[ -z "$h" && -z "$lf" && "$1" == msmarco-v2 ]]; then h=3; lf=0.88; fi
    echo "${h:-2} ${lf:-0.4}"
}
tree_cuckoo_tag() {   # dataset -> "h3_lf0.88/" (results sub-folder) or nothing for the default 2 / 0.4
    local h lf; read -r h lf <<< "$(tree_cuckoo "$1")"
    if [[ "$h" != 2 || "$lf" != 0.4 ]]; then echo "h${h}_lf${lf}/"; fi
}

# ------------------------------------------------------------------------------------------ queued configs
# Which result folders a job in the queue (pending or running) will write, so datasets/run_all.sh and run_me.sh
# don't submit them a second time and check_runs.sh can tell "queued" apart from "crashed".
#   queued_dirs > file        one "folder<TAB>jobid_task" line per queued bins/tree/pacmann task, plus
#                             "?<TAB>method<TAB>dataset<TAB>jobid" for an array whose config list can't be found
# A task of a run_all.sh array runs line ARRAY_OFFSET+task+1 of its CONFIG_LIST. Both come from logs/lists/submitted.tsv
# (run_all.sh appends "jobid list offset" for every array it submits) or, for older arrays, from sacct's SubmitLine.
# Pacmann arrays are one dataset's grid, task i = config i (k, steps, neighbours as in run_pacmann.slurm).
SUBMITTED_FILE="${SUBMITTED_FILE:-logs/lists/submitted.tsv}"
queued_dirs() {
    command -v squeue >/dev/null 2>&1 || return 0
    local res="${RESULTS_DIR:-/scratch/${USER:-$(id -un)}/results}" jid task name _st m d list off
    declare -A jl=() jo=()
    if [[ -f "$SUBMITTED_FILE" ]]; then
        while IFS=$'\t' read -r jid list off; do jl[$jid]=$list; jo[$jid]=$off; done < "$SUBMITTED_FILE"
    fi
    squeue -u "${USER:-$(id -un)}" -h -r -o '%F|%K|%j|%T' 2>/dev/null | while IFS='|' read -r jid task name _st; do
        m=${name%%_*}; d=${name#*_}; d=${d%%_*}
        [[ $m == bins || $m == tree || $m == pacmann ]] || continue
        [[ $task =~ ^[0-9]+$ ]] || task=0
        if [[ $m == pacmann ]]; then
            echo "P	$d	$task	${jid}_$task"; continue
        fi
        if [[ -z "${jl[$jid]+x}" ]]; then   # not submitted by this version of run_all.sh: ask sacct how it was submitted
            local sl; sl=$(sacct -j "$jid" -X -n -P -o SubmitLine 2>/dev/null | head -1 || true)
            jl[$jid]=$(sed -nE 's/.*CONFIG_LIST=([^, ]+).*/\1/p' <<< "$sl")
            jo[$jid]=$(sed -nE 's/.*ARRAY_OFFSET=([0-9]+).*/\1/p' <<< "$sl")
        fi
        list=${jl[$jid]}; off=${jo[$jid]:-0}
        if [[ -z "$list" || ! -f "$list" ]]; then echo "?	$m	$d	$jid"; continue; fi
        echo "L	$m	$list	$(( off + task + 1 ))	${jid}_$task"
    done | awk -F'\t' -v res="$res" -v tf="${TEXT_FORMAT:-bow}" -v to="${TEXT_ORDER:-3}" \
                -v ch="${TREE_CUCKOO_HASHES:-}" -v clf="${TREE_CUCKOO_LF:-}" '
    function ttag(d,   h, l) {   # same rule as tree_cuckoo_tag
        h = ch; l = clf
        if (h == "" && l == "" && d == "msmarco-v2") { h = 3; l = 0.88 }
        if (h == "") h = 2; if (l == "") l = 0.4
        return (h != 2 || l != 0.4) ? "h" h "_lf" l "/" : "" }
    BEGIN { split("5 10 15 20 25 30", ST, " "); split("8 16 24 32 40 48", NB, " ") }
    $1 == "?" { print; next }
    $1 == "P" { i = $3 + 0; k = (i < 36) ? 10 : 100; i %= 36
                printf "%s/pacmann_%s_k%d_steps%d_neighb%d\t%s\n", res, $2, k, ST[int(i / 6) + 1], NB[i % 6 + 1], $4; next }
    $1 == "L" {
        if (!($3 in loaded)) { loaded[$3] = 1; n = 0; while ((getline l < $3) > 0) L[$3, ++n] = l; close($3) }
        l = L[$3, $4 + 0]; if (l == "") next
        split(l, f, " ")
        if ($2 == "bins") {
            t = (f[5] == "true") ? "_vec1" : (f[5] == "text") ? "_text" : "_vec0"
            sfx = (f[5] == "text" && (tf != "bow" || to != 3)) ? "_" tf "o" to : ""
            printf "%s/bins%s_%s_k%s_bs%s_dpb%s%s\t%s\n", res, t, f[1], f[2], f[3], f[4], sfx, $5
        } else
            printf "%s/%stree_grid_%s_b%s_r%s_s%s_L%s_q%s_c%s_k10\t%s\n", res, ttag(f[1]), f[1], f[2], f[3], f[4], f[5], f[6], f[7], $5
    }'
}

# load_queued: fill _QUEUED[folder]=jobid_task and _QBLOCK[method|dataset]=jobid (arrays we can't map: everything of
# that method + dataset counts as queued, so nothing is submitted twice). Reads $QUEUED_FILE if set (written once by
# run_me.sh / run_all.sh), else asks squeue. QUEUED_CHECK=0 turns it off.
declare -A _QUEUED=() _QBLOCK=()
_QUEUED_LOADED=""
load_queued() {
    [[ -n "$_QUEUED_LOADED" || "${QUEUED_CHECK:-1}" == 0 ]] && return 0
    _QUEUED_LOADED=1
    local a b c dd
    while IFS=$'\t' read -r a b c dd; do
        if [[ "$a" == "?" ]]; then _QBLOCK["$b|$c"]=$dd; else _QUEUED["$a"]=$b; fi
    done < <(if [[ -n "${QUEUED_FILE:-}" && -f "$QUEUED_FILE" ]]; then cat "$QUEUED_FILE"; else queued_dirs; fi)
}
is_queued() {   # is_queued <method> <dataset> <folder>
    [[ -n "${_QUEUED[$3]:-}" || -n "${_QBLOCK["$1|$2"]:-}" ]]
}

# ------------------------------------------------------------------------------------------ bins

# vec is true (1 DB of embeddings), false (ID DB + embedding DB) or text (1 DB of compressed text, -text)
bins_config_key() {   # vec k bs dpb
    case $1 in
        text) echo "text_k$2_bs$3_dpb$4" ;;
        true) echo "vec1_k$2_bs$3_dpb$4" ;;
        *)    echo "vec0_k$2_bs$3_dpb$4" ;;
    esac
}

bins_missing() {   # dataset [vec] -> the first missing input of pir_app -t bins (paths as in main.go GetDatasets), or nothing
    local d=$1 vec=${2:-} root="${DATASET_DIR:-../../datasets}" f
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
    [[ $vec == text ]] && need=("${need[@]/*_embed_f32.npy/}")   # -text reads no embeddings
    for f in "${need[@]}"; do
        [[ -n "$f" ]] || continue
        [[ -e "$f" ]] || { echo "$f"; return; }
    done
}

# Vocabulary size: VocabSize from any finished bins run of this dataset (every run writes it to metadata.json),
# else extrapolated from nq (824,171 words at 2.68M docs; msmarco has 1.16M at 8.8M, i.e. ~docs^0.3).
declare -A _VOCAB=()
bins_vocab() {   # dataset -> vocabulary size
    if [[ -z "${_VOCAB[$1]:-}" ]]; then
        local f v=""
        f=$(ls -d "${RESULTS_DIR:-/scratch/${USER:-$(id -un)}/results}"/bins_*_"$1"_k*/metadata.json 2>/dev/null | head -1)
        [[ -n "$f" ]] && v=$(grep -oE '"VocabSize": *"?[0-9]+' "$f" | grep -oE '[0-9]+$' | head -1)
        [[ -n "$v" ]] || v=$(awk -v n="$(dataset_docs "$1")" 'BEGIN { printf "%d", 824171 * (n / 2681468) ^ 0.3 }')
        _VOCAB[$1]=$v
    fi
    echo "${_VOCAB[$1]}"
}

# Model until measured, fitted to 427 nq + msmarco runs (2026-11-08; model/peak 0.8-2.1, median 1.2; with the 1.3x
# below it covers every successful run and 22 of the 23 OOMs of the old model). Peak RSS in GB:
#   base     2.3 + 1.4 KB/doc (corpus, index reader), + 1.15 KB/doc more with embeddings (vec0/vec1)
#   topdocs  every word's top-dpb doc IDs, ~27 B each: V x dpb x 27 B (x2 for text)
#   bins     ~1 KB per bin (bs)
#   vec1 DB  3.3x the PIR DB of 768 B entries. Entries = bs x dpb until the bins run out of distinct (word, doc) pairs,
#            T = 1.66 V sqrt(dpb) (x sqrt of docs-per-word relative to nq): E = S (1 - exp(-T/S)), S = bs x dpb.
#            This is what the old model missed: it ignored bs, so vec1 at bs >= 500k OOMed, while big dpb was ~6x over.
#   text     ~30 B per (bin, doc) slot (padded slices)
#   vec0     the ID DB is small; base + topdocs + bins covers it
# Time: measured runs took <= 18 s/GB (vec) and <= 42 s/GB (text), so 2x that per GB of the estimate, >= 30 min.
bins_model() {   # dataset vec dpb [bs] -> "GB minutes"
    awk -v n="$(dataset_docs "$1")" -v V="$(bins_vocab "$1")" -v vec="$2" -v dpb="$3" -v bs="${4:-0}" 'BEGIN {
        gb = 2.3 + n * 1.4e-6 + (vec != "text" ? n * 1.15e-6 : 0)
        gb += V * dpb * 27e-9 * (vec == "text" ? 2 : 1) + bs * 1e-6
        S = bs * dpb; dpw = (n / V) / 3.25; if (dpw < 1) dpw = 1
        T = 1.66 * V * sqrt(dpb) * sqrt(dpw)
        x = (S > 0) ? T / S : 0   # exp(-x) underflows past ~700 (gawk warns); E is S to double precision by then
        E = (S <= 0) ? 0 : (x > 700 ? S : S * (1 - exp(-x)))
        if (vec == "true") gb += 3.3 * E * 768e-9
        if (vec == "text") gb += S * 30e-9
        gb *= 1.3
        m = gb * (vec == "text" ? 1.4 : 0.6); if (m < 30) m = 30
        printf "%.0f %d\n", gb, m }'
}

# stdin: "dataset k bs dpb vec" per config -> stdout: the same + "GB minutes source missing" (missing = "-" if ready)
# results folder of a bins config: the same name run_bins_low/high.slurm use (a metadata.json there = done)
bins_out_dir() {   # dataset k bs dpb vec
    local t=_vec0 sfx=""
    case $5 in
        true) t=_vec1 ;;
        text) t=_text; [[ "${TEXT_FORMAT:-bow}" != bow || "${TEXT_ORDER:-3}" != 3 ]] && sfx="_${TEXT_FORMAT:-bow}o${TEXT_ORDER:-3}" ;;
    esac
    echo "${RESULTS_DIR:-/scratch/${USER:-$(id -un)}/results}/bins${t}_$1_k$2_bs$3_dpb$4${sfx}"
}

# Configs that already have a metadata.json come out as "... 0 0 - done" (group_plan counts them, nothing is
# submitted for them); FORCE=1 plans them again. Configs a queued job will run come out as "... 0 0 - queued".
bins_plan() {
    load_usage; load_queued
    declare -A model=() miss=()
    local d k bs dpb vec
    while read -r d k bs dpb vec; do
        [[ -n "$d" ]] || continue
        local od; od=$(bins_out_dir "$d" "$k" "$bs" "$dpb" "$vec")
        if [[ -z "${FORCE:-}" && -s "$od/metadata.json" ]]; then
            echo "$d $k $bs $dpb $vec 0 0 - done"; continue
        fi
        if is_queued bins "$d" "$od"; then echo "$d $k $bs $dpb $vec 0 0 - queued"; continue; fi
        local mk="$d.$vec.$dpb.$bs"   # the model depends on bs for every mode
        [[ -n "${model[$mk]:-}" ]] || model[$mk]=$(bins_model "$d" "$vec" "$dpb" "$bs")
        local fb=(${model[$mk]})
        local est; est=$(estimate_resources bins "$d" "$(bins_config_key "$vec" "$k" "$bs" "$dpb")" "${fb[0]}" "${fb[1]}")
        local m="-"
        if [[ "${CHECK_INPUTS:-1}" == 1 ]]; then
            local mv="$d"; [[ $vec == text ]] && mv+=".text"
            [[ -n "${miss[$mv]+x}" ]] || miss[$mv]=$(bins_missing "$d" "$vec")
            m=${miss[$mv]:--}
        fi
        echo "$d $k $bs $dpb $vec $est $m"
    done
}

# group_plan <plan file> <out dir> <prefix>: split the ready configs into one list per (memory bucket, time bucket).
# Prints "GB minutes count list_file" per group, then "missing <count> <first missing file>", "over <count>",
# "done <count>" and "queued <count>".
group_plan() {
    awk -v out="$2" -v pre="$3" -v mb="$MEM_BUCKETS" -v tb="$TIME_BUCKETS" '
    BEGIN { nm = split(mb, M, " "); nt = split(tb, T, " ") }
    {
        if ($NF == "done") { done++; next }
        if ($NF == "queued") { queued++; next }
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
        if (done) print "done", done, "-"
        if (queued) print "queued", queued, "-"
    }' "$1" | sort -k1,1n -k2,2n
}
