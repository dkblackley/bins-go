#!/usr/bin/env python3
"""Concatenate the shards written by `collection_preproc.py --num-shards N` into one .npy (+ .ids).

    python datasets/merge_embed_shards.py --output ../datasets/msmarco-v2/msmarco-v2_corpus_embed_f32.npy --num-shards 16
"""
import argparse
import os

import numpy as np
from numpy.lib.format import open_memmap

ap = argparse.ArgumentParser()
ap.add_argument("--output", required=True, help="Same --output you gave collection_preproc.py")
ap.add_argument("--num-shards", type=int, required=True)
ap.add_argument("--delete-shards", action="store_true")
args = ap.parse_args()

shards = [f"{args.output}.shard{i:03d}of{args.num_shards:03d}.npy" for i in range(args.num_shards)]
missing = [s for s in shards if not os.path.exists(s) or not os.path.exists(s + ".ids")]
if missing:
    raise SystemExit(f"Missing shards: {missing}")

arrs = [np.load(s, mmap_mode="r") for s in shards]
dim = arrs[0].shape[1]
total = sum(a.shape[0] for a in arrs)
print(f"Merging {len(arrs)} shards -> {total:,} x {dim}")

out = open_memmap(args.output, mode="w+", dtype=np.float32, shape=(total, dim))
row = 0
with open(args.output + ".ids", "w") as fid:
    for s, a in zip(shards, arrs):
        step = 1_000_000
        for i in range(0, a.shape[0], step):
            e = min(i + step, a.shape[0])
            out[row + i: row + e] = a[i:e]
        with open(s + ".ids") as f:
            n_ids = 0
            for line in f:
                fid.write(line)
                n_ids += 1
        assert n_ids == a.shape[0], f"{s}: {a.shape[0]} rows but {n_ids} ids"
        row += a.shape[0]
        print(f"  {s}: {a.shape[0]:,} rows")
del out

if args.delete_shards:
    for s in shards:
        os.remove(s)
        os.remove(s + ".ids")
print(f"Wrote {args.output} and {args.output}.ids")
