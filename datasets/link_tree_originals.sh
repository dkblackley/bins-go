#!/bin/bash
# Link the tree's original msmarco + scifact files (the collaborator's bm25-tree data, which tree_paths in
# tree_sweep_grid.sh expects under the repo's data/ and indexes/) from an older checkout into this repo.
# Symlinks only: nothing is copied, moved or deleted, and paths that already exist here are left alone.
#
#   bash datasets/link_tree_originals.sh /scratch/dblackle/bm25-tree-new [more roots to search ...]
#
# For each file it tries <root>/<path> and then <root>/<path without its leading data/>, root by root.
# Then it links the B*_r*_s* artifact dirs already built there (data/<msmarco|scifact>/pir/), so tree-db only builds
# the missing ones. Afterwards: STAGES="tree-db tree" bash datasets/run_all.sh msmarco scifact
set -uo pipefail
cd "$(dirname "$0")/.."
[[ $# -gt 0 ]] || { sed -n 2,10p "$0"; exit 1; }

NEED=(
    data/msmarco_reordered data/queries.dev.small.tsv data/queries.dev.small.analyzed.json data/qrels.dev.small.tsv
    data/stage3_doc_aligned_shuffle.npy data/stage3_doc_aligned.npy.ids data/stage3_query.npy
    data/block_permutations.json
    indexes/scifact data/scifact/queries.qrels.tsv data/scifact/queries.qrels.analyzed.json data/scifact/qrels.trec
    data/scifact/clustered/stage3_e5_pca192_doc_embeddings.npy
    data/scifact/clustered/stage3_e5_pca192_doc_embeddings.npy.ids
    data/scifact/stage3_e5_pca192_query_embeddings.npy data/scifact/block_permutations.json
)
missing=0
for rel in "${NEED[@]}"; do
    if [[ -e "$rel" ]]; then echo "ok       $rel"; continue; fi
    src=""
    for root in "$@"; do
        for c in "$root/$rel" "$root/${rel#data/}"; do [[ -e "$c" ]] && { src=$(readlink -f "$c"); break 2; }; done
    done
    if [[ -z "$src" ]]; then echo "MISSING  $rel (not in: $*)"; missing=1; continue; fi
    mkdir -p "$(dirname "$rel")" && ln -s "$src" "$rel" && echo "linked   $rel -> $src"
done

# already-built tree artifact dirs (one per B/r/s shape)
for d in msmarco scifact; do
    mkdir -p "data/$d/pir"
    for root in "$@"; do
        for shape in "$root/data/$d/pir"/B*_r*_s* "$root/$d/pir"/B*_r*_s*; do
            [[ -d "$shape" && -f "$shape/layout.json" ]] || continue
            dst="data/$d/pir/$(basename "$shape")"
            [[ -e "$dst" ]] || { ln -s "$(readlink -f "$shape")" "$dst" && echo "linked   $dst"; }
        done
    done
    echo "$d: $(ls -d data/$d/pir/B*_r*_s* 2>/dev/null | wc -l) artifact dirs (tree-db builds the rest)"
done
[[ $missing == 0 ]] && echo "All there. Next: STAGES=\"tree-db tree\" bash datasets/run_all.sh msmarco scifact" \
                    || { echo "Some files are missing: add another root to search, or ask for them to be rebuilt"; exit 1; }
