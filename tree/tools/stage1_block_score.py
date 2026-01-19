#!/usr/bin/env python3
"""
Compute Stage1 child-score for the block containing a given gold doc.

It maps gold external doc -> internal (using docmap.bin), computes the leaf node id,
and then sums Stage1 per-term child scores along the path (root -> ... -> leaf parent)
using the Stage1 precomputed data/idmap.

Example (run from golang/tools):
  python stage1_block_score.py \
    --qid 1048585 --gold 7187158 \
    --queries ../queries.dev.small.analyzed.json \
    --docmap ../../msmarco_reordered/docmap.bin \
    --stage1-data ../stage1_data.bin \
    --stage1-idmap ../stage1/stage1_idmap.bin \
    --stage1-vocab ../stage1_vocab.json \
    --B 32 --r 128

Output:
  - Internal doc id, leaf node id, leaf index
  - Per-level child index and aggregated Stage1 score (sum over query terms of the child byte)
  - If a term lacks a row for a node, it contributes 0
"""

import argparse
import json
import math
import os
import struct
from typing import Dict, List, Set, Tuple


def load_docmap(path: str) -> List[int]:
    size = os.path.getsize(path)
    cnt = size // 8
    res = []
    with open(path, "rb") as f:
        for _ in range(cnt):
            b = f.read(8)
            if len(b) < 8:
                break
            res.append(struct.unpack("<Q", b)[0])
    return res


def ext_to_internal(docmap: List[int], external: int) -> int:
    for i, v in enumerate(docmap):
        if v == external:
            return i
    return -1


def load_query_terms(path: str, qid: str) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    for q in data:
        if q.get("id") == qid:
            return q.get("terms", [])
    return []


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        raw = json.load(f)
    vocab = {}
    for term, tid in raw.items():
        try:
            vocab[term] = int(tid)
        except Exception:
            continue
    return vocab


def read_stage1_rows_for_nodes(idmap_path: str, target_nodes: Set[int], target_terms: Set[int]) -> Dict[Tuple[int,int], int]:
    """Return map (termID,nodeID) -> row_index for the selected nodes/terms."""
    rec_size = 16
    buf = bytearray(rec_size)
    out = {}
    with open(idmap_path, "rb") as f:
        while True:
            n = f.readinto(buf)
            if n != rec_size:
                break
            term_id = struct.unpack_from("<I", buf, 0)[0]
            node_id = struct.unpack_from("<I", buf, 4)[0]
            row_idx = struct.unpack_from("<Q", buf, 8)[0]
            if node_id in target_nodes and term_id in target_terms:
                out[(term_id, node_id)] = row_idx
    return out


def read_row(data_path: str, r: int, row_idx: int) -> bytes:
    with open(data_path, "rb") as f:
        f.seek(row_idx * r)
        return f.read(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", required=True)
    ap.add_argument("--gold", required=True, help="External doc id (qrels)")
    ap.add_argument("--queries", required=True, help="Analyzed queries JSON path")
    ap.add_argument("--docmap", required=True, help="docmap.bin (internal->external)")
    ap.add_argument("--stage1-data", required=True, help="stage1_data.bin")
    ap.add_argument("--stage1-idmap", required=True, help="stage1_idmap.bin")
    ap.add_argument("--stage1-vocab", required=True, help="stage1_vocab.json")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--r", type=int, default=128)
    args = ap.parse_args()

    gold_ext = int(args.gold)
    # load docmap and resolve internal id
    docmap = load_docmap(args.docmap)
    internal = ext_to_internal(docmap, gold_ext)
    if internal < 0:
        print(f"gold external {gold_ext} not in docmap")
        return

    N = len(docmap)
    num_leaves = math.ceil(N / args.B)
    height = math.ceil(math.log(num_leaves, args.r)) if num_leaves > 1 else 0
    leaf_idx = internal // args.B
    leaf_node_start = args.r ** height
    leaf_node = leaf_node_start + leaf_idx

    # build ancestor chain (root->...->leaf parent)
    path = []  # list of (parent_node, child_idx)
    node = leaf_node
    while node > 1:
        parent = node // args.r
        child_idx = node - parent * args.r
        path.append((parent, child_idx))
        node = parent
    path.append((1, None))  # root marker (child_idx unused)
    path = list(reversed(path))

    # load query terms and vocab ids
    terms = load_query_terms(args.queries, args.qid)
    vocab = load_vocab(args.stage1_vocab)
    term_ids = {vocab[t] for t in terms if t in vocab}

    if not term_ids:
        print(f"No terms from qid {args.qid} found in stage1 vocab")
        return

    target_nodes = {p for p, _ in path if p is not None}
    row_map = read_stage1_rows_for_nodes(args.stage1_idmap, target_nodes, term_ids)

    print(f"gold external={gold_ext} -> internal={internal}, leaf_idx={leaf_idx}, leaf_node={leaf_node}, height={height}")
    print(f"query terms ({len(term_ids)} matched): {sorted(term_ids)[:10]}{' ...' if len(term_ids)>10 else ''}")

    # compute per-level aggregate score
    for lvl, (parent, child_idx) in enumerate(path[1:], start=1):  # skip root marker
        agg = 0
        missing = 0
        for tid in term_ids:
            row_idx = row_map.get((tid, parent))
            if row_idx is None:
                missing += 1
                continue
            row = read_row(args.stage1_data, args.r, row_idx)
            if child_idx is not None and child_idx < len(row):
                agg += row[child_idx]
        print(f"Level {lvl}: parent={parent} child_idx={child_idx} agg_score={agg} (missing_terms={missing})")


if __name__ == "__main__":
    main()
