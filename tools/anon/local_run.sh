#!/bin/bash
# local_run.sh: every experiment of the paper on one machine, one run after another (no job scheduler).
#
#   bash local_run.sh                       # every stage, every dataset (weeks of compute and TBs of RAM for the
#                                           # largest configs: see DATASETS and ONLY to run a subset)
#   bash local_run.sh --count               # how many configs each sweep has, per dataset; runs nothing
#   DRY_RUN=1 bash local_run.sh bins        # print the bins commands instead of running them
#   DATASETS=scifact bash local_run.sh      # the whole pipeline on SciFact (minutes, a few GB of RAM)
#   DATASETS="nq fever" bash local_run.sh prep embed index treedb tree
#   SIMULATE=1 bash local_run.sh bins pacmann tree   # simulated PIR: same answers, bytes, LAN/WAN and client
#                                                    # storage as a real run, far less RAM; server timings are -1
#
# Stages, in order (default: all). Each skips work whose output already exists (FORCE=1 redoes it).
#   prep      download + convert to <DATASETS_ROOT>/<name>/{corpus.jsonl,queries.jsonl,qrels/test.tsv,num_docs.txt}
#   embed     e5-base-v2 + a per-dataset PCA to 192 dims (GPU recommended): <name>_{corpus,query}_embed_f32.npy
#   index     the keyword index PILLAR-Bin needs (pir_app -t index)
#   treedb    PILLAR-Tree's artifacts for every (B, r, s) of the tree grid (datasets/prep_tree.sh -> data/)
#   bins      PILLAR-Bin: vec1 / vec0 / text x k x binSize x docsPerBin, + the vec0 batch/rounds and query-term sweeps
#   pacmann   PACMANN: k x steps x neighbours
#   tree      PILLAR-Tree: the grid in tree_sweep_grid.sh (SWEEP=legacy|oat|grid|all)
#   tiptoe    Tiptoe (simulated: exact quality and communication): docs/cluster x multi-assign x bundle x hint x k x
#             embedding precision
#   coeus     Coeus (exact TF-IDF quality and communication): features x tf weighting x k
#   docfetch  cost of privately downloading the k returned documents (one Piano PIR DB of compressed docs)
#   plain     non-private BM25 / dense / hybrid baselines (cmd/plain_baselines -> $RESULTS_DIR/baselines.json)
#
# Environment:
#   DATASETS_ROOT  datasets folder (pir_app -dataset), default ../datasets
#   RESULTS_DIR    results folder, default $DATASETS_ROOT/results ($DATASETS_ROOT/results_simulated with SIMULATE=1)
#   DATASETS       default "scifact nq dbpedia-entity fever msmarco msmarco-v2 msmarco-v2-25m msmarco-v2-50m";
#                  also: miracl-en, msmarco-v2-100m, trec-covid, hotpotqa. msmarco-v2-* subsamples reuse
#                  msmarco-v2's embeddings, so list msmarco-v2 first.
#   SIMULATE=1     pir_app -simulate (bins, pacmann, tree, docfetch)
#   NORAM=1        pir_app -noram: PIR DBs, client hints and embeddings mmap'd from disk (NORAM_DIR, default
#                  temp_storage). Same results, slower
#   ONLY           only configs whose results folder name matches this regex, e.g. ONLY='_k10_bs1000_'
#   SWEEP          tree grid, default all;   TIPTOE_PRECISIONS  default "fp32 fp8 fp1"
#   FORCE=1, DRY_RUN=1, FP16=1 (embedding in fp16; always on for corpora of 25M+ docs), ENCODE_BATCH (128)
#
# Every run writes <RESULTS_DIR>/<config>/ with metadata.json (quality, bytes, LAN/WAN, client storage, timings),
# per-query answers, execution.log and time_v.txt (peak RSS, wall time). Failures are listed in
# $RESULTS_DIR/failed.tsv and the sweep moves on. Plot with: RESULTS_DIR=... python plotting/main.py
#
# MS MARCO v1 is not downloaded automatically: put collection, queries.dev.small and qrels.dev (from
# https://microsoft.github.io/msmarco/) in <DATASETS_ROOT>/msmarco as corpus.jsonl, queries.dev.small.jsonl and
# qrels/qrels.dev.tsv (BEIR format), then run the other stages.
set -uo pipefail
cd "$(dirname "$0")"
ROOT_DIR=$PWD

DATASETS_ROOT="${DATASETS_ROOT:-../datasets}"
mkdir -p "$DATASETS_ROOT"; DATASETS_ROOT=$(cd "$DATASETS_ROOT" && pwd)
export DATASET_DIR="$DATASETS_ROOT" ROOT_DIR
if [[ "${SIMULATE:-0}" == 1 ]]; then RESULTS_DIR="${RESULTS_DIR:-$DATASETS_ROOT/results_simulated}"
else RESULTS_DIR="${RESULTS_DIR:-$DATASETS_ROOT/results}"; fi
export RESULTS_DIR
DATASETS="${DATASETS:-scifact nq dbpedia-entity fever msmarco msmarco-v2 msmarco-v2-25m msmarco-v2-50m}"
ALL_STAGES="prep embed index treedb bins pacmann tree tiptoe coeus docfetch plain"
COUNT=0
[[ "${1:-}" == "--count" ]] && { COUNT=1; DRY_RUN=1; shift; }
CHECK=1; [[ -n "${DRY_RUN:-}" ]] && CHECK=0   # skip configs with missing inputs (not when only listing them)
STAGES="${*:-$ALL_STAGES}"
for s in $STAGES; do [[ " $ALL_STAGES " == *" $s "* ]] || { echo "unknown stage $s (stages: $ALL_STAGES)" >&2; exit 1; }; done
has() { [[ " $STAGES " == *" $1 "* ]]; }

# tree grid + helpers: build_tree_grid, tree_paths, tree_missing, tree_cuckoo, dataset_docs, bins_missing, bins_out_dir
source ./tree_sweep_grid.sh

PIR_FLAGS=()
[[ "${SIMULATE:-0}" == 1 ]] && PIR_FLAGS+=(-simulate)
[[ "${NORAM:-0}" == 1 ]] && PIR_FLAGS+=(-noram -noram-dir "${NORAM_DIR:-$ROOT_DIR/temp_storage}")
mkdir -p "$RESULTS_DIR" logs
declare -A NCFG=()

log() { echo "$(date '+%F %T') $*"; }
# run <method> <dataset> <out dir> <command...>: one config, skipped if done / filtered; never stops the sweep
run() {
    local method=$1 d=$2 od=$3; shift 3
    if [[ -n "${ONLY:-}" ]] && ! [[ "$(basename "$od")" =~ $ONLY ]]; then return 0; fi
    NCFG[$method $d]=$(( ${NCFG[$method $d]:-0} + 1 ))
    (( COUNT )) && return 0
    if [[ -z "${FORCE:-}" ]] && [[ -s "$od/metadata.json" || -e "$od/.done" ]]; then return 0; fi
    if [[ -n "${DRY_RUN:-}" ]]; then echo "$*"; return 0; fi
    mkdir -p "$od"
    log "start $method $d -> $od"
    local start=$SECONDS rc
    bash ./measure_run.sh "$od/time_v.txt" "$@" > "$od/execution.log" 2>&1
    rc=$?
    if (( rc == 0 )); then log "done  $method $d ($(( SECONDS - start )) s)"; touch "$od/.done"
    else
        printf '%s\t%s\trc=%s\t%s\n' "$method" "$od" "$rc" "$(date '+%F %T')" >> "$RESULTS_DIR/failed.tsv"
        log "FAIL  $method $d rc=$rc, see $od/execution.log"
    fi
    return 0
}
# step <description> <command...>: a preparation step (stops the script on failure: later stages need it)
step() {
    local what=$1; shift
    if [[ -n "${DRY_RUN:-}" ]]; then (( COUNT )) || echo "$*"; return 0; fi
    log "$what"
    "$@" || { echo "FAILED: $what" >&2; exit 1; }
}
emb_prefix() { [[ $1 == trec-covid ]] && echo trec || echo "$1"; }
big() { (( $(dataset_docs "$1") >= 25000000 )); }

if ! (( COUNT )) && [[ -z "${DRY_RUN:-}" ]] && { has index || has bins || has pacmann || has tree || has docfetch || has plain; }; then
    log "building pir_app, docfetch, plain_baselines"
    source ./go_env.sh
    go build -o pir_app ./main.go && go build -o bin/docfetch ./cmd/docfetch &&
        CGO_ENABLED=0 go build -o bin/plain_baselines ./cmd/plain_baselines || { echo "build failed" >&2; exit 1; }
fi

for d in $DATASETS; do
    D="$DATASETS_ROOT/$d"; RAW="$DATASETS_ROOT/raw"
    (( COUNT )) || echo "================ $d"

    # ------------------------------------------------------------------------------------------ prep
    if has prep && ! (( COUNT )) && [[ -n "${FORCE:-}" || ! -s "$D/corpus.jsonl" ]]; then
        case $d in
            scifact)   # the BEIR files already have the layout pir_app expects for scifact
                [[ -d "$RAW/scifact" ]] || step "download scifact" bash datasets/download.sh scifact "$RAW"
                step "copy scifact" cp -r "$RAW/scifact" "$D" ;;
            nq|hotpotqa|fever|trec-covid|dbpedia-entity)
                [[ -d "$RAW/$d" ]] || step "download $d" bash datasets/download.sh "$d" "$RAW"
                step "convert $d" python3 datasets/prepare_dataset.py --format beir --src "$RAW/$d" --out "$D" --split test ;;
            msmarco-v2|miracl-en)
                fmt=$([[ $d == msmarco-v2 ]] && echo msmarco-v2 || echo miracl)
                [[ -d "$RAW/$d" ]] || step "download $d" bash datasets/download.sh "$d" "$RAW"
                step "convert $d" python3 datasets/prepare_dataset.py --format "$fmt" --src "$RAW/$d" --out "$D" --split dev ;;
            msmarco-v2-25m|msmarco-v2-50m|msmarco-v2-100m)   # all relevant docs + random fill (seed 42), v2's embeddings
                n="${d##*-}"; n="${n%m}"; F="$DATASETS_ROOT/msmarco-v2"
                step "subsample $d" python3 datasets/prepare_dataset.py --format beir --src "$F" --out "$D" \
                    --max-docs "$(( n * 1000000 ))" --src-embed "$F/msmarco-v2_corpus_embed_f32.npy" \
                    --src-query-embed "$F/msmarco-v2_query_embed_f32.npy" ;;
            msmarco) echo "msmarco: see the note at the top of this file" ;;
            *) echo "unknown dataset $d" >&2; exit 1 ;;
        esac
    fi

    # ------------------------------------------------------------------------------------------ embed
    P=$(emb_prefix "$d")
    if has embed && [[ $d != msmarco-v2-*m ]] && [[ -n "${FORCE:-}" || ! -s "$D/${P}_query_embed_f32.npy" ]]; then
        fp16=""; { [[ -n "${FP16:-}" ]] || big "$d"; } && fp16=--fp16
        nd=$(dataset_docs "$d")
        step "embed $d corpus" python3 collection_preproc.py --documents "$D/corpus.jsonl" \
            --output "$D/${P}_corpus_embed_f32.npy" --pca "$D/pca_e5_768_to_192.faiss" --num-docs "$nd" \
            --encode-batch "${ENCODE_BATCH:-128}" $fp16 --skip-if-done ${FORCE:+--overwrite}
        q="$D/queries.jsonl"; [[ $d == msmarco ]] && q="$D/queries.dev.small.jsonl"
        step "embed $d queries" python3 query_preproc.py --queries "$q" --output "$D/${P}_query_embed_f32.npy" \
            --pca "$D/pca_e5_768_to_192.faiss"
    fi

    # ------------------------------------------------------------------------------------------ index
    idx="$DATASETS_ROOT/index_$d"; [[ $d == msmarco ]] && idx="$DATASETS_ROOT/index_marco"
    [[ $d == trec-covid ]] && idx="$DATASETS_ROOT/index_trec_covid"
    if has index && [[ -n "${FORCE:-}" || ! -d "$idx" ]]; then
        step "bins index $d" ./pir_app -t index -name "$d" -dataset "$DATASETS_ROOT"
    fi

    # ------------------------------------------------------------------------------------------ treedb
    if has treedb; then
        step "tree artifacts $d (SWEEP=${SWEEP:-all})" env SWEEP="${SWEEP:-all}" bash datasets/prep_tree.sh "$d"
    fi

    n=$(dataset_docs "$d")

    # ------------------------------------------------------------------------------------------ bins
    if has bins; then
        BS_LIST=(100 1000 10000 100000 500000 1000000 5000000 10000000)
        DPB_LIST=(10 25 50 100 250 500 1000 2500)
        [[ $d =~ ^msmarco-v2(-25m|-50m)?$ ]] && BS_LIST+=(20000000 50000000)
        huge=0; [[ $d == msmarco-v2* || $d == miracl-en ]] && huge=1   # < 100k bins hold a tiny part of 25M+ docs
        bins_cfgs=()
        for vec in true false text; do for k in 10 100; do for bs in "${BS_LIST[@]}"; do for dpb in "${DPB_LIST[@]}"; do
            (( huge && bs < 100000 )) && continue
            bins_cfgs+=("$k $bs $dpb $vec")
        done; done; done; done
        # vec0 client-storage sweep: -vecBatch B x -vecRounds R with -idBatch I, pruned where it would be all padding
        if [[ $d == scifact ]]; then vk=10; vbs="100 1000 10000 100000"; vvb="4 16 50"; vvr="3 10 30"; vib=2
        else vk=10; vbs="100000 1000000 10000000"; vvb="10 50 100 2500"; vvr="1 3 10 30"; vib=24; fi
        for bs in $vbs; do for dpb in "${DPB_LIST[@]}"; do for vb in $vvb; do for vr in $vvr; do
            (( huge && bs < 100000 )) && continue
            cap=$(( 8 * dpb ))
            (( vb <= cap && vb * (vr - 1) < cap && vb <= 2 * n / 16 )) || continue
            bins_cfgs+=("$vk $bs $dpb vb${vb}r${vr}i${vib}")
        done; done; done; done
        # query-term sweep: -maxTerms N on each dataset's best vec0 (bs, dpb) family
        case $d in
            scifact) base="1000 1000" ;;           msmarco|msmarco-v2-25m) base="1000000 2500" ;;
            nq) base="1000000 500" ;;              fever) base="10000000 500" ;;
            dbpedia-entity) base="1000000 1000" ;; msmarco-v2-50m|msmarco-v2) base="10000000 2500" ;;
            *) base="" ;;
        esac
        if [[ -n "$base" ]]; then
            read -r bs dpb <<< "$base"
            for vr in 1 10 30; do for mt in 0 1 2 3 4 6 8; do bins_cfgs+=("10 $bs $dpb vb100r${vr}i24m${mt}"); done; done
        fi
        m=$(bins_missing "$d"); mt_=$(bins_missing "$d" text)
        for cfg in "${bins_cfgs[@]}"; do
            read -r k bs dpb vec <<< "$cfg"
            if (( CHECK )); then
                [[ $vec == text && -n "$mt_" ]] && continue
                [[ $vec != text && -n "$m" ]] && continue
            fi
            case $vec in
                true|false) layout=(-vectors="$vec") ;;
                text) layout=(-text -text-format bow -text-order 3) ;;
                *) [[ $vec =~ ^vb([0-9]+)r([0-9]+)i([0-9]+)(m([0-9]+))?$ ]]
                   layout=(-vectors=false -vecBatch "${BASH_REMATCH[1]}" -vecRounds "${BASH_REMATCH[2]}"
                           -idBatch "${BASH_REMATCH[3]}" -vecOrder rr)
                   [[ -n "${BASH_REMATCH[4]}" ]] && layout+=(-maxTerms "${BASH_REMATCH[5]}") ;;
            esac
            od=$(bins_out_dir "$d" "$k" "$bs" "$dpb" "$vec")
            run bins "$d" "$od" ./pir_app -t bins -name "$d" -n "$n" -k "$k" -binSize "$bs" -docsPerBin "$dpb" \
                -dim 192 -dataset "$DATASETS_ROOT" "${layout[@]}" "${PIR_FLAGS[@]}" -outDir "$od" -outFile results
        done
        [[ -n "$m" ]] && (( CHECK )) && echo "bins $d: skipped configs with missing inputs (first: $m)"
    fi

    # ------------------------------------------------------------------------------------------ pacmann
    if has pacmann; then
        export PACMANN_GRAPH_DIR="${PACMANN_GRAPH_DIR:-$D}"   # NGT graph cache, one per config
        for k in 10 100; do for st in 5 10 15 20 25 30; do for nb in 8 16 24 32 40 48; do
            od="$RESULTS_DIR/pacmann_${d}_k${k}_steps${st}_neighb${nb}"
            run pacmann "$d" "$od" ./pir_app -t pacmann -name "$d" -n "$n" -k "$k" -steps "$st" -neighb "$nb" \
                -dim 192 -dataset "$DATASETS_ROOT" "${PIR_FLAGS[@]}" -outDir "$od" -outFile results
        done; done; done
    fi

    # ------------------------------------------------------------------------------------------ tree
    if has tree; then
        DATA_ARR=(); L_ARR=(); KCAND_ARR=(); QWORDS_ARR=(); B_ARR=(); R_ARR=(); S_ARR=(); MAX_Q_ARR=(); _TREE_SEEN=()
        build_tree_grid "$d" "${SWEEP:-all}"
        read -r ch lf <<< "$(tree_cuckoo "$d")"
        kicks=$([[ $ch == 3 ]] && echo 5000 || echo 1000)
        for i in "${!DATA_ARR[@]}"; do
            B=${B_ARR[$i]} r=${R_ARR[$i]} s=${S_ARR[$i]} L=${L_ARR[$i]} q=${QWORDS_ARR[$i]} c=${KCAND_ARR[$i]}
            od="$RESULTS_DIR/$(tree_cuckoo_tag "$d")tree_grid_${d}_b${B}_r${r}_s${s}_L${L}_q${q}_c${c}_k10"
            if (( CHECK )); then
                miss=$(tree_missing "$d" "$B" "$r" "$s")
                [[ -z "$miss" ]] || { echo "tree $d B=$B r=$r s=$s: missing $miss (run the treedb stage)"; continue; }
            fi
            tree_paths "$d" "$B" "$r" "$s"
            cache="$ROOT_DIR/cuckoo_cache/$CACHE_SUBDIR/shared"; [[ -n "${DRY_RUN:-}" ]] || mkdir -p "$cache"   # tables shared by a shape's configs
            run tree "$d" "$od" ./pir_app -t tree -name "$d" -n "$DB_SIZE" -dim 192 -dataset "$DATASETS_ROOT" \
                --L "$L" --k_candidates "$c" --query-words "$q" --B "$B" --r "$r" --s "$s" --k 10 \
                --max-queries "${MAX_Q_ARR[$i]}" --k1 0.9 --b 0.4 --scale 10 --precompute --pir-batch-size 128 \
                --cuckoo-load-factor "$lf" --cuckoo-hashes "$ch" --cuckoo-max-kicks "$kicks" --cuckoo-build-attempts 8 \
                --layout-path "$LAYOUT_PATH" --index "$INDEX" --queries "$QUERIES" --qrels "$QRELS" --vocab "$VOCAB" \
                --stage1-data-bin "$STAGE1_DATA" --stage1-idmap-bin "$STAGE1_IDMAP" \
                --stage2-data-bin "$STAGE2_DATA" --stage2-idmap-bin "$STAGE2_IDMAP" \
                --doc-embed "$DOC_EMBED" --doc-id-map "$DOC_ID_MAP" --query-embed "$QUERY_EMBED" --embed-dim 192 \
                ${BLOCK_PERMS:+--block-permutation "$BLOCK_PERMS"} --pir-threads "$(nproc)" \
                --cuckoo-cache-dir "$cache" "${PIR_FLAGS[@]}" -outDir "$od" -outFile results
        done
    fi

    # ------------------------------------------------------------------------------------------ tiptoe
    if has tiptoe; then
        read -ra precs <<< "${TIPTOE_PRECISIONS:-fp32 fp8 fp1}"
        fmt=(); big "$d" && fmt=(--format npy)
        for dpc in 250 2500 40000; do for ma in 1 2; do for bundle in 40 320; do
            TT="$DATASETS_ROOT/tiptoe/${d}_dpc${dpc}_m${ma}_b${bundle}"
            if ! (( COUNT )) && [[ ! -f "$TT/tiptoe_meta.json" ]]; then
                step "tiptoe DB $d dpc=$dpc ma=$ma bundle=$bundle" python3 baselines/tiptoe/preprocess_tiptoe.py \
                    --dataset "$DATASETS_ROOT" --name "$d" --docs-per-cluster "$dpc" --multi-assign "$ma" \
                    --bundle-size "$bundle" --out "$TT" --kmeans-cache "$DATASETS_ROOT/tiptoe/kmeans_cache" "${fmt[@]}"
            fi
            C=$dpc; MA=$ma
            [[ -f "$TT/tiptoe_meta.json" ]] && read -r C MA < <(python3 -c \
                "import json;m=json.load(open('$TT/tiptoe_meta.json'));print(m['num_clusters'],m['multi_assign'])")
            for hint in 125 500; do for k in 10 100; do for prec in "${precs[@]}"; do
                sfx=""; [[ $prec == fp32 ]] || sfx="_$prec"
                # several docs/cluster values can give the same cluster count: the first one's results stand
                od="$RESULTS_DIR/tiptoe_${d}_c${C}_m${MA}_b${bundle}_h${hint}_k${k}${sfx}"
                run tiptoe "$d" "$od" python3 baselines/tiptoe/simulate_tiptoe.py --dataset "$DATASETS_ROOT" \
                    --name "$d" --tiptoe-dir "$TT" --k "$k" --emb-hint-mb "$hint" --results-root "$RESULTS_DIR" \
                    --precision "$prec" --parallelism "$(nproc)"
            done; done; done
        done; done; done
    fi

    # ------------------------------------------------------------------------------------------ coeus
    if has coeus; then
        tok="$DATASETS_ROOT/coeus_cache"
        for f in 8192 32768 262144; do for tf in raw log; do for k in 10 100; do
            od="$RESULTS_DIR/coeus_${d}_k${k}_f${f}_tf${tf}"
            run coeus "$d" "$od" python3 baselines/coeus/run_coeus.py --dataset "$DATASETS_ROOT" --name "$d" \
                --k "$k" --features "$f" --tf "$tf" --results-root "$RESULTS_DIR" --token-cache "$tok" \
                --parallelism "$(nproc)"
        done; done; done
    fi

    # ------------------------------------------------------------------------------------------ docfetch, plain
    if has docfetch; then
        df=(-name "$d" -dataset "$DATASETS_ROOT" -out "$RESULTS_DIR" -k 10,100 -compress flate)
        [[ -n "${FORCE:-}" ]] && df+=(-force)
        [[ "${SIMULATE:-0}" == 1 ]] && df+=(-simulate)
        run docfetch "$d" "$RESULTS_DIR/docfetch_${d}_k10_flate" ./bin/docfetch "${df[@]}"
    fi
    if has plain; then
        run plain "$d" "$RESULTS_DIR/plain_baselines_$d" ./bin/plain_baselines -name "$d" -dataset "$DATASETS_ROOT" \
            -out "$RESULTS_DIR/baselines.json" -threads "$(nproc)"
    fi
done

if (( COUNT )); then
    echo "configs per sweep and dataset:"
    for key in "${!NCFG[@]}"; do printf '  %-10s %-18s %6d\n' ${key} "${NCFG[$key]}"; done | sort
    total=0; for v in "${NCFG[@]}"; do total=$(( total + v )); done
    echo "  total: $total"
fi
