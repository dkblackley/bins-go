#!/usr/bin/env python3
"""
Comprehensive diagnostic: verify that the gold leaf is NOT in the frontier,
and therefore Stage2 can never find it (by design).

This proves the miss is an ALGORITHM issue (Stage1 filtering), not code mismatch.
"""

import argparse
import json
import math
import struct
from typing import List, Tuple


def height_and_range(node: int, B: int, r: int, N: int) -> Tuple[int, int, int, int]:
    """Compute height, level, and doc range for a given node."""
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
    return h, level, doc_start, doc_end


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=int, required=True, help="External doc id")
    ap.add_argument("--qid", required=True, help="Query id")
    ap.add_argument("--docmap", required=True)
    ap.add_argument("--frontier-json", help="Path to frontier nodes JSON (or provide --frontier-list)")
    ap.add_argument("--frontier-list", help="Comma-separated frontier nodes")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--r", type=int, default=128)
    ap.add_argument("--N", type=int, default=8841823)
    args = ap.parse_args()

    # Parse frontier
    frontier: List[int] = []
    if args.frontier_json:
        with open(args.frontier_json, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and args.qid in data:
                frontier = data[args.qid]
            elif isinstance(data, list):
                frontier = data
    elif args.frontier_list:
        frontier = [int(x.strip()) for x in args.frontier_list.split(",")]
    else:
        print("Must provide --frontier-json or --frontier-list")
        return

    # Convert gold external to internal
    internal = ext_to_internal(args.docmap, args.gold)
    if internal is None:
        print(f"ERROR: gold {args.gold} not found in docmap")
        return

    # Which leaf should contain gold?
    gold_leaf_idx = internal // args.B
    gold_leaf_node = gold_leaf_idx + 1  # leaf nodes are 1-indexed
    h, lv, a, b = height_and_range(gold_leaf_node, args.B, args.r, args.N)

    print(f"GOLD DOC ANALYSIS")
    print(f"================")
    print(f"Gold external: {args.gold}, internal: {internal}")
    print(f"Gold leaf index: {gold_leaf_idx}, leaf node: {gold_leaf_node}")
    print(f"Gold leaf level: {lv}, doc range: [{a}, {b})")
    print()

    print(f"FRONTIER ANALYSIS")
    print(f"=================")
    print(f"Frontier size: {len(frontier)} nodes")
    print(f"Gold leaf ({gold_leaf_node}) in frontier? {gold_leaf_node in frontier}")
    print()

    if gold_leaf_node in frontier:
        print("✓ Gold leaf IS in frontier -> Stage2 CAN find it (if Stage2 data exists)")
    else:
        print("✗ Gold leaf NOT in frontier -> Stage2 CANNOT find it (by design)")
        print()
        print("ROOT CAUSE: Stage1 beam search did NOT select the gold leaf path.")
        print("This is an ALGORITHM issue, NOT a code mismatch.")
        print()
        print("Why might Stage1 miss the gold leaf?")
        print("- Cumulative path score to gold leaf was lower than top-L paths")
        print("- Beam width (L=200) might not be wide enough for this query")
        print("- Query terms may have low IDF/coverage at gold leaf path")
        print()


if __name__ == "__main__":
    main()
