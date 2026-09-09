#!/usr/bin/env python3
"""
Inspect Stage2 precomputed rows for specific terms and a leaf node.

Example:
  ./stage2_row_inspect.py \
      --leaf-node 2174210 \
      --terms what,paula,deen,brother \
      --stage2-data ../../stage2/data.bin \
      --stage2-idmap ../../stage2/idmap.bin \
      --vocab ../../stage1_vocab.json \
      --B 32 --s 8 --scale 10
"""

import argparse
import json
import os
import struct
from typing import Dict, List, Tuple


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        raw = json.load(f)
    vocab: Dict[str, int] = {}
    for term, tid_str in raw.items():
        try:
            vocab[term] = int(tid_str)
        except ValueError:
            continue
    return vocab


def load_idmap(path: str) -> Dict[int, Dict[int, int]]:
    lookup: Dict[int, Dict[int, int]] = {}
    rec_size = 16  # uint32 term_id, uint32 node_id, uint64 row_idx
    with open(path, "rb") as f:
        chunk = f.read(rec_size)
        while chunk:
            if len(chunk) < rec_size:
                break
            term_id = struct.unpack("<I", chunk[0:4])[0]
            node_id = struct.unpack("<I", chunk[4:8])[0]
            row_idx = struct.unpack("<Q", chunk[8:16])[0]
            lookup.setdefault(term_id, {})[node_id] = row_idx
            chunk = f.read(rec_size)
    return lookup


def read_row(data_path: str, row_idx: int, r: int) -> List[int]:
    offset = row_idx * r
    with open(data_path, "rb") as f:
        f.seek(offset)
        row = f.read(r)
    return list(row)


def maybe_terms_from_qid(qid: str, queries_path: str) -> List[str]:
    if not qid:
        return []
    with open(queries_path, "r") as f:
        queries = json.load(f)
    for q in queries:
        if q.get("id") == qid:
            return q.get("terms", [])
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf-node", type=int, required=True, help="Leaf node id to inspect")
    ap.add_argument("--terms", help="Comma-separated terms; overrides --qid")
    ap.add_argument("--qid", help="Query id to fetch terms from analyzed JSON")
    ap.add_argument("--queries", default="../../queries.dev.small.analyzed.json", help="Analyzed queries JSON path")
    ap.add_argument("--stage2-data", required=True, help="Path to Stage2 data.bin")
    ap.add_argument("--stage2-idmap", required=True, help="Path to Stage2 idmap.bin")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json")
    ap.add_argument("--B", type=int, default=32, help="Block size B")
    ap.add_argument("--s", type=int, default=8, help="Sub-block size s")
    ap.add_argument("--scale", type=float, default=10.0, help="Scale used during precompute")
    args = ap.parse_args()

    terms: List[str] = []
    if args.terms:
        terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    else:
        terms = maybe_terms_from_qid(args.qid, args.queries)
    if not terms:
        print("No terms provided or found for qid")
        return

    vocab = load_vocab(args.vocab)
    lookup = load_idmap(args.stage2_idmap)
    r = args.B // args.s

    print(f"Leaf node: {args.leaf_node}, B={args.B}, s={args.s}, r={r}, scale={args.scale}")
    print(f"Terms: {terms}")

    for term in terms:
        tid = vocab.get(term)
        if tid is None:
            tid = vocab.get(term.lower())
        if tid is None:
            print(f"Term '{term}': NOT in vocab")
            continue

        term_rows = lookup.get(tid, {})
        if args.leaf_node not in term_rows:
            print(f"Term '{term}' (id={tid}): no row for leaf {args.leaf_node}")
            continue

        row_idx = term_rows[args.leaf_node]
        row_bytes = read_row(args.stage2_data, row_idx, r)
        scores = [b / args.scale for b in row_bytes]
        print(f"Term '{term}' (id={tid}): row_idx={row_idx}, bytes={row_bytes}, scores={scores}")


if __name__ == "__main__":
    main()
