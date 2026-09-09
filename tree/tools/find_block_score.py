#!/usr/bin/env python3
"""
Find the Stage2 hitSub block (range) that contains a given gold doc and report its ScoreUB.

Example:
  ./find_block_score.py --qid 1048585 --gold 7187158 \
      --stage2-hits ../stage2_hits.json \
      --docmap ../../msmarco_reordered/docmap.bin

Inputs:
  --qid: query id (string)
  --gold: external doc id (int or string)
  --stage2-hits: path to saved Stage2 hitSubs JSON (map qid -> []HitSubBlock)
  --docmap: path to Lucene docmap.bin (internal -> external, uint64 little endian)
  --B: block size (docs per leaf), optional, used only for reporting leaf index

Output:
  - Internal doc id for the gold external id (if found in docmap)
  - Leaf index (internal // B)
  - Whether the internal id appears in any hitSub range for the qid and, if so, the block index and ScoreUB
"""

import argparse
import json
import os
import struct
from typing import Optional


def ext_to_internal(docmap_path: str, external: int) -> Optional[int]:
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
    ap.add_argument("--qid", required=True)
    ap.add_argument("--gold", required=True, help="External doc id (from qrels)")
    ap.add_argument("--stage2-hits", required=True)
    ap.add_argument("--docmap", required=True)
    ap.add_argument("--B", type=int, default=32)
    args = ap.parse_args()

    gold_external = int(args.gold)

    if not os.path.exists(args.stage2_hits):
        print(f"stage2 hits file not found: {args.stage2_hits}")
        return
    if not os.path.exists(args.docmap):
        print(f"docmap not found: {args.docmap}")
        return

    with open(args.stage2_hits, "r") as f:
        hits = json.load(f)

    if args.qid not in hits:
        print(f"qid {args.qid} not in stage2 hits")
        return

    internal = ext_to_internal(args.docmap, gold_external)
    if internal is None:
        print(f"gold external {gold_external} not found in docmap")
        return

    leaf_idx = internal // args.B
    print(f"gold external={gold_external} -> internal={internal}, leaf_idx={leaf_idx}")

    blocks = hits[args.qid]
    for i, b in enumerate(blocks):
        start = int(b["Start"])
        end = int(b["End"])
        if start <= internal < end:
            print(f"FOUND in block #{i}: Start={start}, End={end}, ScoreUB={b.get('ScoreUB')}")
            break
    else:
        print("NOT FOUND in any hitSub block for this qid")


if __name__ == "__main__":
    main()
