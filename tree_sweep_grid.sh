#!/bin/bash
# Shared tree parameter grid, sourced by run_tree_pir_old.slurm (runs) and
# datasets/prep_tree.sh (artifacts). Edit the lists here, both pick it up.
#
#   DATASET=msmarco|scifact|nq|dbpedia-entity|fever|hotpotqa|miracl-en|msmarco-v2[-25m|-50m|-100m]
#   SWEEP=legacy|oat|grid|all   (default all = oat + grid)
#
#   bash tree_sweep_grid.sh --count    number of run configs (for --array=0-N)
#   bash tree_sweep_grid.sh --list     every run config
#   bash tree_sweep_grid.sh --shapes   the B/r/s artifact dirs those configs need
#   bash tree_sweep_grid.sh --plan     every config with its RAM/time estimate and first missing input ("-" = ready)
#   bash tree_sweep_grid.sh --summary  how many configs are ready / missing per memory bucket
#   bash tree_sweep_grid.sh --mem      the RAM model per B/r/s (before any measurement)
#
# Parameters (one PIR stage each, see tree/internal/bm25/search.go):
#   B  docs per leaf block          r  tree arity (Round 1 rows are r bytes, height = ceil(log_r(N/B)))
#   s  docs per sub-block (Round 2 rows are B/s bytes, B % s == 0)
#   L  beam width (leaves kept per level; Round 1 and 2 fetch L x q rows per level)
#   q  padded query terms           c  k_candidates, sub-blocks kept for Stage 3 (c x s docs reranked)
#
# Rules applied to every config: s divides B, c is clamped to L*B/s (there are
# only that many sub-blocks to pick from), duplicates after clamping are dropped.
#
# SWEEP modes:
#   legacy  the old Cartesian grid (k_candidates fixed), kept for reproducing past runs
#   oat     one-at-a-time: every parameter from its minimum up, the others held at an anchor
#   grid    joint grid over all six parameters, geometric steps starting at small values;
#           c is set from a target rerank pool P = c*s so it scales with s

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

DATA_ARR=(); L_ARR=(); KCAND_ARR=(); QWORDS_ARR=(); B_ARR=(); R_ARR=(); S_ARR=(); MAX_Q_ARR=()
declare -A _TREE_SEEN=()

add_cfg() {   # dataset B r s L q c max_q
    local d=$1 B=$2 r=$3 s=$4 L=$5 q=$6 c=$7 max_q=$8
    (( s >= 1 && s <= B && B % s == 0 && r >= 2 && L >= 1 && q >= 1 && c >= 1 )) || return 0
    local cmax=$(( L * B / s ))
    (( c > cmax )) && c=$cmax
    local key="$d|$B|$r|$s|$L|$q|$c"
    [[ -n "${_TREE_SEEN[$key]:-}" ]] && return 0
    _TREE_SEEN[$key]=1
    DATA_ARR+=("$d"); B_ARR+=("$B"); R_ARR+=("$r"); S_ARR+=("$s")
    L_ARR+=("$L"); QWORDS_ARR+=("$q"); KCAND_ARR+=("$c"); MAX_Q_ARR+=("$max_q")
}

# Cartesian product with a fixed c (old behaviour).
add_cartesian() {   # dataset c max_q "B list" "r list" "s list" "q list" "L list"
    local d=$1 c=$2 max_q=$3 B r s q L
    for B in $4; do for r in $5; do for s in $6; do for q in $7; do for L in $8; do
        add_cfg "$d" "$B" "$r" "$s" "$L" "$q" "$c" "$max_q"
    done; done; done; done; done
}

# Cartesian product where c = P / s for each target rerank pool P.
add_pool_grid() {   # dataset max_q "B list" "r list" "s list" "L list" "q list" "P list"
    local d=$1 max_q=$2 B r s L q P
    for B in $3; do for r in $4; do for s in $5; do for L in $6; do for q in $7; do for P in $8; do
        (( P >= s )) || continue
        add_cfg "$d" "$B" "$r" "$s" "$L" "$q" $(( P / s )) "$max_q"
    done; done; done; done; done; done
}

# Vary one parameter over a list, everything else at the anchor.
add_oat() {   # dataset max_q anchor"B r s L q c"  name  "values"
    local d=$1 max_q=$2 name=$4 v
    local aB aR aS aL aQ aC
    read -r aB aR aS aL aQ aC <<< "$3"
    for v in $5; do
        local B=$aB R=$aR S=$aS L=$aL Q=$aQ C=$aC
        case $name in
            B) B=$v; (( S > B )) && S=$B ;;   # keep s <= B so small B is still valid
            r) R=$v ;;
            s) S=$v ;;
            L) L=$v ;;
            q) Q=$v ;;
            c) C=$v ;;
        esac
        add_cfg "$d" "$B" "$R" "$S" "$L" "$Q" "$C" "$max_q"
    done
}

# Large-corpus grid (msmarco and every new dataset, 2.7M-138M docs).
_big_grid() {   # dataset max_q sweep
    local d=$1 MQ=$2 sweep=$3 ANCHOR="32 32 8 100 4 128"
    if [[ $sweep == legacy ]]; then
        #                    c   max_q  B          r          s        q         L
        add_cartesian "$d" 128 "$MQ" "16 32 64" "8 32 128" "4 8 16" "2 4 6 8" "50 100 200 400"
    fi
    if [[ $sweep == oat || $sweep == all ]]; then
        add_oat "$d" "$MQ" "$ANCHOR" B "4 8 16 32 64 128"
        add_oat "$d" "$MQ" "$ANCHOR" r "2 4 8 16 32 64 128 256"
        add_oat "$d" "$MQ" "$ANCHOR" s "1 2 4 8 16 32"
        add_oat "$d" "$MQ" "$ANCHOR" L "1 2 4 8 16 32 64 100 200 400 800"
        add_oat "$d" "$MQ" "$ANCHOR" q "1 2 3 4 6 8 12"
        add_oat "$d" "$MQ" "$ANCHOR" c "1 2 4 8 16 32 64 128 256 400"
    fi
    if [[ $sweep == grid || $sweep == all ]]; then
        #                  max_q  B             r         s        L                  q       P = c*s
        add_pool_grid "$d" "$MQ" "8 16 32 64" "4 16 64" "2 4 8"  "2 8 32 128 512"   "2 4 8" "128 1024"
    fi
}

TREE_DATASETS="msmarco scifact nq dbpedia-entity fever hotpotqa miracl-en msmarco-v2 msmarco-v2-25m msmarco-v2-50m msmarco-v2-100m"

build_tree_grid() {
    local dataset=$1 sweep=${2:-all}
    case $sweep in legacy|oat|grid|all) ;; *) echo "Unknown SWEEP: $sweep (legacy|oat|grid|all)" >&2; return 1 ;; esac
    case $dataset in
    msmarco)
        _big_grid msmarco 6980 "$sweep" ;;
    nq|hotpotqa|fever|dbpedia-entity|miracl-en|msmarco-v2*)
        _big_grid "$dataset" 10000 "$sweep" ;;
    scifact)
        local MQ=300 ANCHOR="16 8 4 20 8 20"
        if [[ $sweep == legacy ]]; then
            add_cartesian scifact 20 $MQ "8 16 32 64" "4 8 32 128" "2 4 8" "4 8 12 16" "5 10 20 40 80"
        fi
        if [[ $sweep == oat || $sweep == all ]]; then
            add_oat scifact $MQ "$ANCHOR" B "2 4 8 16 32 64"
            add_oat scifact $MQ "$ANCHOR" r "2 4 8 16 32 64"
            add_oat scifact $MQ "$ANCHOR" s "1 2 4 8 16"
            add_oat scifact $MQ "$ANCHOR" L "1 2 3 5 10 20 40 80 160"
            add_oat scifact $MQ "$ANCHOR" q "1 2 4 6 8 12 16 24"
            add_oat scifact $MQ "$ANCHOR" c "1 2 4 8 16 20 40 80"
        fi
        if [[ $sweep == grid || $sweep == all ]]; then
            add_pool_grid scifact $MQ "4 8 16 32" "2 4 16"   "1 2 4 8" "1 3 10 30 100" "2 8 16"  "16 64 256"
        fi
        ;;
    *) echo "Unknown DATASET: $dataset (one of: $TREE_DATASETS)" >&2; return 1 ;;
    esac
}

# ------------------------------------------------------------------------------------------ resources
# Model (used until a config has been measured, see slurm_common.sh): rough server RAM for one tree run, per (dataset, B, r, s); L, q and c barely matter.
# (term, node) row counts come from the measured msmarco B32 r32 s8 layout (pirhint/results/msmarco.md):
# rows per node as a function of docs under the node, log-log interpolated, scaled by words/doc.
#   tables = rows * (8-byte key + r or B/s bytes, rounded up to 8) / Cuckoo load factor (TREE_CUCKOO_LF, default 0.4)
#   mem    = TREE_MEM_CAL * (1.5 * (stage1 + stage2 tables + fp32 stage3 embeddings) + raw data/idmap files) + 4 GB
# TREE_MEM_CAL=0.75 is calibrated so the worst shape of the old msmarco grid (B16 r128) comes out at ~107 GB, since
# every old msmarco config ran in 120 GB. Then msmarco B32 r32 s8 is ~62 GB. The peak RSS of every run is now logged
# (MaxRSS in the run log, via /usr/bin/time -v), so set TREE_MEM_CAL from real numbers once some runs are back.
#   name:docs:words_per_doc
TREE_MEM_CAL="${TREE_MEM_CAL:-0.75}"
declare -A _TREE_MEM=()

tree_mem_gb() {   # dataset B r s -> estimated GB (integer)
    local key="$1|$2|$3|$4|${TREE_CUCKOO_LF:-0.4}"
    if [[ -z "${_TREE_MEM[$key]:-}" ]]; then
        _TREE_MEM[$key]=$(awk -v d="$1" -v B="$2" -v r="$3" -v s="$4" -v sizes="$DATASET_SIZES" -v cal="$TREE_MEM_CAL" -v lf="${TREE_CUCKOO_LF:-0.4}" '
        function ppn(x,   i, lx) {   # (term,node) rows per node holding x docs, msmarco-like text
            lx = log(x)
            if (lx <= LX[1]) return exp(LY[1] + (lx - LX[1]) * (LY[2] - LY[1]) / (LX[2] - LX[1]))
            for (i = 1; i < 6; i++) if (lx <= LX[i+1]) return exp(LY[i] + (lx - LX[i]) * (LY[i+1] - LY[i]) / (LX[i+1] - LX[i]))
            return exp(LY[6] + (lx - LX[6]) * (LY[6] - LY[5]) / (LX[6] - LX[5]))
        }
        function up8(x) { return int((x + 7) / 8) * 8 }
        BEGIN {
            split("1 32 1024 32768 1048576 8841823", X, " "); split("35 796 17370 297900 2160000 2600000", Y, " ")
            for (i = 1; i <= 6; i++) { LX[i] = log(X[i]); LY[i] = log(Y[i]) }
            n = split(sizes, S, /[ \n]+/)
            for (i = 1; i <= n; i++) { split(S[i], f, ":"); if (f[1] == d) { N = f[2]; w = f[3] / 56 } }
            if (!N) { print 1500; exit }
            nl = int((N + B - 1) / B); H = (nl > 1) ? int(log(nl) / log(r) + 0.999999999) : 0
            s1 = 0; s1raw = 0
            for (l = 0; l < H; l++) {
                span = B * r ^ (H - l); nodes = int((nl + r ^ (H - l) - 1) / r ^ (H - l))
                rows = nodes * w * ppn(span < N ? span : N)
                s1 += rows * (8 + up8(r)) / lf; s1raw += rows * (r + 16)
            }
            rows2 = nl * w * ppn(B)
            s2 = rows2 * (8 + up8(B / s)) / lf; s2raw = rows2 * (B / s + 16)
            s3 = N * 192 * 4
            printf "%d\n", cal * (1.5 * (s1 + s2 + s3) + s1raw + s2raw) / 1e9 + 4
        }')
    fi
    echo "${_TREE_MEM[$key]}"
}

# Model time: msmarco runs took ~45 min (never above 2 h), so scale that by the RAM estimate (both follow DB size),
# x2 because it is unmeasured, at least 30 min.
tree_model_minutes() { awk -v g="$1" 'BEGIN { m = 2 * 45 * g / 62; printf "%d\n", (m < 30 ? 30 : m) }'; }

tree_config_key() {   # B r s L q c -> config name used in the usage log (cuckoo settings included)
    local tag=""
    [[ "${TREE_CUCKOO_HASHES:-2}" != 2 || "${TREE_CUCKOO_LF:-0.4}" != 0.4 ]] && tag="_h${TREE_CUCKOO_HASHES:-2}lf${TREE_CUCKOO_LF:-0.4}"
    echo "b$1_r$2_s$3_L$4_q$5_c$6${tag}"
}

# Paths for one config. Sets DB_SIZE INDEX QUERIES QRELS VOCAB LAYOUT_PATH STAGE1_DATA STAGE1_IDMAP STAGE2_DATA
# STAGE2_IDMAP DOC_EMBED DOC_ID_MAP QUERY_EMBED BLOCK_PERMS CACHE_SUBDIR (relative to the repo root / DATASET_DIR).
tree_paths() {   # dataset B r s
    local d=$1 shape="B$2_r$3_s$4" root="${ROOT_DIR:-$PWD}" art
    case $d in
    msmarco)
        DB_SIZE=8841823; art="$root/data/msmarco/pir/$shape"
        INDEX="$root/data/msmarco_reordered"; QUERIES="$root/data/queries.dev.small.tsv"; QRELS="$root/data/qrels.dev.small.tsv"
        DOC_EMBED="$root/data/stage3_doc_aligned_shuffle.npy"; DOC_ID_MAP="$root/data/stage3_doc_aligned.npy.ids"
        QUERY_EMBED="$root/data/stage3_query.npy"; BLOCK_PERMS="data/block_permutations.json"; CACHE_SUBDIR="" ;;
    scifact)
        DB_SIZE=5183; local dr="$root/data/scifact"; art="$dr/pir/$shape"
        INDEX="$root/indexes/scifact"; QUERIES="$dr/queries.qrels.tsv"; QRELS="$dr/qrels.trec"
        DOC_EMBED="$dr/clustered/stage3_e5_pca192_doc_embeddings.npy"
        DOC_ID_MAP="$dr/clustered/stage3_e5_pca192_doc_embeddings.npy.ids"
        QUERY_EMBED="$dr/stage3_e5_pca192_query_embeddings.npy"; BLOCK_PERMS="$dr/block_permutations.json"
        CACHE_SUBDIR="scifact/$shape" ;;
    *)  # layout written by datasets/prep_tree.sh
        DB_SIZE=$(dataset_docs "$d"); local dr="$root/data/${d}_tree"; art="$dr/$shape"
        INDEX="$dr/index_$d"; QUERIES="$dr/queries.qrels.tsv"; QRELS="$dr/qrels.trec"
        DOC_EMBED="$dr/stage3_doc.npy"; DOC_ID_MAP="$dr/stage3_doc.npy.ids"
        # rows follow queries.jsonl order; tree_main matches them to the analyzed queries by ID
        QUERY_EMBED="${DATASET_DIR:-../../datasets}/$d/${d}_query_embed_f32.npy"; BLOCK_PERMS=""
        CACHE_SUBDIR="$d/$shape" ;;
    esac
    LAYOUT_PATH="$art/layout.json"; VOCAB="$art/stage1_vocab.json"
    STAGE1_DATA="$art/stage1_data.bin"; STAGE1_IDMAP="$art/stage1_idmap.bin"
    STAGE2_DATA="$art/stage2/data.bin"; STAGE2_IDMAP="$art/stage2/idmap.bin"
}

tree_missing() {   # dataset B r s -> the first missing input file, or nothing if the config can run
    tree_paths "$@"
    local f
    for f in "$INDEX" "$QUERIES" "$QRELS" "$DOC_EMBED" "$DOC_ID_MAP" "$QUERY_EMBED" "$LAYOUT_PATH" "$VOCAB" \
             "$STAGE1_DATA" "$STAGE1_IDMAP" "$STAGE2_DATA" "$STAGE2_IDMAP"; do
        [[ -e "$f" ]] || { echo "$f"; return; }
    done
    [[ "$DB_SIZE" -gt 0 ]] || echo "${DATASET_DIR:-../../datasets}/$1/num_docs.txt"
}

# One line per config of the current arrays: "d B r s L q c max_q GB minutes source missing"
# (missing = first missing input, "-" if ready). Only checked when CHECK_INPUTS=1 (default), cached per shape.
tree_plan() {
    load_usage
    local i; declare -A miss=()
    for i in "${!DATA_ARR[@]}"; do
        local d=${DATA_ARR[$i]} B=${B_ARR[$i]} r=${R_ARR[$i]} s=${S_ARR[$i]}
        local gb; gb=$(tree_mem_gb "$d" "$B" "$r" "$s")
        local fb_gb; fb_gb=$(awk -v g="$gb" 'BEGIN { printf "%.0f", g * 1.2 }')
        local est; est=$(estimate_resources tree "$d" "$(tree_config_key "$B" "$r" "$s" "${L_ARR[$i]}" "${QWORDS_ARR[$i]}" "${KCAND_ARR[$i]}")" \
                         "$fb_gb" "$(tree_model_minutes "$gb")")
        local m="-"
        if [[ "${CHECK_INPUTS:-1}" == 1 ]]; then
            [[ -n "${miss[$d$B.$r.$s]+x}" ]] || miss[$d$B.$r.$s]=$(tree_missing "$d" "$B" "$r" "$s")
            m=${miss[$d$B.$r.$s]:--}
        fi
        echo "$d $B $r $s ${L_ARR[$i]} ${QWORDS_ARR[$i]} ${KCAND_ARR[$i]} ${MAX_Q_ARR[$i]} $est $m"
    done
}

# Unique "B r s" triples the current arrays need.
tree_grid_shapes() {
    local i
    for i in "${!B_ARR[@]}"; do echo "${B_ARR[$i]} ${R_ARR[$i]} ${S_ARR[$i]}"; done | sort -n -k1,1 -k2,2 -k3,3 -u
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    build_tree_grid "${DATASET:?set DATASET (see the top of this file)}" "${SWEEP:-all}"
    case "${1:---count}" in
        --count)  echo "${#DATA_ARR[@]} configs, $(tree_grid_shapes | wc -l) B/r/s artifact dirs for ${DATASET} SWEEP=${SWEEP:-all} (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))" ;;
        --list)   for i in "${!DATA_ARR[@]}"; do echo "$i ${DATA_ARR[$i]} B=${B_ARR[$i]} r=${R_ARR[$i]} s=${S_ARR[$i]} L=${L_ARR[$i]} q=${QWORDS_ARR[$i]} c=${KCAND_ARR[$i]}"; done ;;
        --shapes) tree_grid_shapes ;;
        --plan)   tree_plan ;;
        --mem)    for sh in $(tree_grid_shapes | tr ' ' ','); do IFS=, read -r b r s <<< "$sh"; gb=$(tree_mem_gb "$DATASET" "$b" "$r" "$s"); echo "B=$b r=$r s=$s model ~${gb} GB"; done ;;
        --summary) tree_plan | awk '{ ready = ($12 == "-") ? "ready" : "missing"; g = $9 + 0
                     b = "over"; split(ENVIRON["MEM_BUCKETS"] ? ENVIRON["MEM_BUCKETS"] : "16 32 64 100 250 500 1000 1500", B, " ")
                     for (i = 1; i in B; i++) if (g <= B[i]) { b = B[i] "G"; break }
                     n[ready " " b]++ } END { for (k in n) print k, n[k] }' | sort -k1,1 -k2,2n ;;
        *) echo "usage: $0 [--count|--list|--shapes|--plan|--mem|--summary]" >&2; exit 1 ;;
    esac
fi
