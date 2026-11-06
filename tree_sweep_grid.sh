#!/bin/bash
# Shared tree parameter grid, sourced by run_tree_pir_old.slurm (runs) and
# build_tree_artifacts.slurm (artifacts). Edit the lists here, both pick it up.
#
#   DATASET=msmarco|scifact   SWEEP=legacy|oat|grid|all   (default all = oat + grid)
#
#   bash tree_sweep_grid.sh --count    number of run configs (for --array=0-N)
#   bash tree_sweep_grid.sh --list     every run config
#   bash tree_sweep_grid.sh --shapes   the B/r/s artifact dirs those configs need
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

build_tree_grid() {
    local dataset=$1 sweep=${2:-all}
    case $dataset in
    msmarco)
        local MQ=6980 ANCHOR="32 32 8 100 4 128"
        if [[ $sweep == legacy ]]; then
            #                       c   max_q  B          r          s        q         L
            add_cartesian msmarco 128 $MQ "16 32 64" "8 32 128" "4 8 16" "2 4 6 8" "50 100 200 400"
        fi
        if [[ $sweep == oat || $sweep == all ]]; then
            add_oat msmarco $MQ "$ANCHOR" B "4 8 16 32 64 128"
            add_oat msmarco $MQ "$ANCHOR" r "2 4 8 16 32 64 128 256"
            add_oat msmarco $MQ "$ANCHOR" s "1 2 4 8 16 32"
            add_oat msmarco $MQ "$ANCHOR" L "1 2 4 8 16 32 64 100 200 400 800"
            add_oat msmarco $MQ "$ANCHOR" q "1 2 3 4 6 8 12"
            add_oat msmarco $MQ "$ANCHOR" c "1 2 4 8 16 32 64 128 256 400"
        fi
        if [[ $sweep == grid || $sweep == all ]]; then
            #                    max_q  B             r         s        L                  q       P = c*s
            add_pool_grid msmarco $MQ "8 16 32 64" "4 16 64" "2 4 8"  "2 8 32 128 512"   "2 4 8" "128 1024"
        fi
        ;;
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
    *) echo "Unknown DATASET: $dataset" >&2; return 1 ;;
    esac
}

# Unique "B r s" triples the current arrays need.
tree_grid_shapes() {
    local i
    for i in "${!B_ARR[@]}"; do echo "${B_ARR[$i]} ${R_ARR[$i]} ${S_ARR[$i]}"; done | sort -n -k1,1 -k2,2 -k3,3 -u
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    build_tree_grid "${DATASET:?set DATASET to msmarco or scifact}" "${SWEEP:-all}"
    case "${1:---count}" in
        --count)  echo "${#DATA_ARR[@]} configs, $(tree_grid_shapes | wc -l) B/r/s artifact dirs for ${DATASET} SWEEP=${SWEEP:-all} (use --array=0-$(( ${#DATA_ARR[@]} - 1 )))" ;;
        --list)   for i in "${!DATA_ARR[@]}"; do echo "$i ${DATA_ARR[$i]} B=${B_ARR[$i]} r=${R_ARR[$i]} s=${S_ARR[$i]} L=${L_ARR[$i]} q=${QWORDS_ARR[$i]} c=${KCAND_ARR[$i]}"; done ;;
        --shapes) tree_grid_shapes ;;
        *) echo "usage: $0 [--count|--list|--shapes]" >&2; exit 1 ;;
    esac
fi
