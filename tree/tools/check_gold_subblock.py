#!/usr/bin/env python3
"""
Given a leaf node and the Go NodeToDocRange calculation,
find which sub-blocks the gold doc falls into and check if those sub-blocks have zero bounds.
"""

import argparse
import json
import math
import struct
from typing import Dict, List, Tuple


def node_to_doc_range(node: int, B: int, r: int, N: int) -> Tuple[int, int]:
    """Replicate Go's NodeToDocRange logic."""
    if r > 1:
        level = math.floor(math.log(max(1, node - 1)) / math.log(r))
    else:
        level = 0
    num_leaves = math.ceil(N / B)
    h = math.ceil(math.log(num_leaves) / math.log(r)) if r > 1 else 1
    leaves_per_node = math.pow(r, h - level)
    if r > 1:
        first_node_at_level = int(math.pow(r, level))
    else:
        first_node_at_level = 1
    idx_in_level = node - first_node_at_level
    leaf_start = idx_in_level * leaves_per_node
    doc_start = int(leaf_start) * B
    doc_end = int(min(N, doc_start + leaves_per_node * B))
    return doc_start, doc_end


def ext_to_internal(docmap_path: str, external: int):
    with open(docmap_path, "rb") as f:
        idx = 0
        chunk = f.read(8)
        while chunk:
            if struct.unpack("<Q", chunk)[0] == external:
                return idx
            idx += 1
            chunk = f.read(8)
    return None


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {term: int(tid_str) for term, tid_str in raw.items()}


def load_idmap(path: str) -> Dict[int, Dict[int, int]]:
    lookup: Dict[int, Dict[int, int]] = {}
    with open(path, "rb") as f:
        chunk = f.read(16)
        while chunk:
            if len(chunk) < 16:
                break
            term_id = struct.unpack("<I", chunk[0:4])[0]
            node_id = struct.unpack("<I", chunk[4:8])[0]
            row_idx = struct.unpack("<Q", chunk[8:16])[0]
            lookup.setdefault(term_id, {})[node_id] = row_idx
            chunk = f.read(16)
    return lookup


def read_row(data_path: str, row_idx: int, r: int) -> List[int]:
    offset = row_idx * r
    with open(data_path, "rb") as f:
        f.seek(offset)
        row = f.read(r)
    return list(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=int, required=True, help="External doc id")
    ap.add_argument("--leaf-node", type=int, required=True)
    ap.add_argument("--docmap", required=True)
    ap.add_argument("--stage2-data", required=True)
    ap.add_argument("--stage2-idmap", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--s", type=int, default=8)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--N", type=int, default=8841823)
    args = ap.parse_args()

    internal = ext_to_internal(args.docmap, args.gold)
    if internal is None:
        print(f"Gold {args.gold} not found in docmap")
        return
    print(f"Gold: external={args.gold}, internal={internal}")

    r = args.B // args.s
    a, b = node_to_doc_range(args.leaf_node, args.B, r, args.N)
    print(f"Leaf {args.leaf_node}: doc range [{a}, {b})")

    # Which sub-blocks does the gold doc fall into?
    if not (a <= internal < b):
        print(f"ERROR: gold {internal} is NOT in range [{a}, {b})")
        return

    # Offset within the leaf
    offset = internal - a
    # Which sub-block(s) does this offset cover?
    sub_block_idx = offset // args.s
    print(f"Gold offset in leaf: {offset}, falls in sub-block index: {sub_block_idx}")

    # Load Stage2 data
    vocab = load_vocab(args.vocab)
    lookup = load_idmap(args.stage2_idmap)

    # Check which query terms have non-zero scores in this sub-block
    terms = ["what", "paula", "deen", "brother"]
    print(f"\nSub-block {sub_block_idx} scores for each term:")

    for term in terms:
        tid = vocab.get(term) or vocab.get(term.lower())
        if tid is None:
            print(f"  {term}: NOT in vocab")
            continue
        term_rows = lookup.get(tid, {})
        if args.leaf_node not in term_rows:
            print(f"  {term} (id={tid}): no row for leaf")
            continue
        row_idx = term_rows[args.leaf_node]
        row_bytes = read_row(args.stage2_data, row_idx, r)
        score_ub = row_bytes[sub_block_idx] / args.scale
        print(f"  {term} (id={tid}): row_idx={row_idx}, sub-block score={score_ub}")

    # Check aggregated score
    print(f"\nAggregated scores for sub-block {sub_block_idx}:")
    agg_score = 0.0
    for term in terms:
        tid = vocab.get(term) or vocab.get(term.lower())
        if tid is None:
            continue
        term_rows = lookup.get(tid, {})
        if args.leaf_node not in term_rows:
            continue
        row_idx = term_rows[args.leaf_node]
        row_bytes = read_row(args.stage2_data, row_idx, r)
        agg_score += row_bytes[sub_block_idx]

    agg_score_ub = agg_score / args.scale
    print(f"Aggregated (raw): {agg_score}, (scaled): {agg_score_ub}")

    # Key question: if agg_score_ub > 0, why was this sub-block not in hitSubs?
    if agg_score_ub > 0:
        print(f"\nALERT: sub-block {sub_block_idx} has non-zero score {agg_score_ub}")
        print("It should have been included in hitSubs (unless filtered by k_candidates limit)")
    else:
        print(f"\nSub-block has zero aggregated score -> correctly filtered out by Go logic")


if __name__ == "__main__":
    main()
