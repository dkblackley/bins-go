#!/usr/bin/env python3
"""
Shrink the Stage3 document embeddings (and so the Stage3 PIR entry and client
hint, which are proportional to it) by storing them as int8 or float16.

    python tree/tools/quantize_embeddings.py data/stage3_doc_aligned_shuffle.npy --dtype int8
    -> data/stage3_doc_aligned_shuffle_int8.npy   (+ a copy of the .ids file if present)

Pass the output to pir_app with --doc-embed; the query embeddings stay float32.

int8 uses per-vector absmax scaling: v_q = round(127 * v / max|v|). Stage3 ranks
by cosine similarity, which ignores a per-vector scale, so the scale is not
stored and the reranker uses the int8 values directly. The script reports how
much the cosine against random other rows moves, as a sanity check.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np


def quantize(x, dtype):
    if dtype == "float16":
        return x.astype(np.float16)
    scale = np.abs(x).max(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return np.clip(np.rint(x / scale * 127.0), -127, 127).astype(np.int8)


def cosine(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings", type=Path, help="(N, D) float32/float64 .npy")
    ap.add_argument("--dtype", choices=["int8", "float16"], default="int8")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chunk", type=int, default=1_000_000, help="rows converted at a time")
    args = ap.parse_args()

    src = np.load(args.embeddings, mmap_mode="r")
    if src.ndim != 2:
        raise SystemExit(f"expected a 2-D array, got shape {src.shape}")
    out = args.out or args.embeddings.with_name(f"{args.embeddings.stem}_{args.dtype}.npy")
    dst = np.lib.format.open_memmap(out, mode="w+", dtype=np.dtype(args.dtype), shape=src.shape)
    for start in range(0, src.shape[0], args.chunk):
        dst[start:start + args.chunk] = quantize(np.asarray(src[start:start + args.chunk], dtype=np.float32), args.dtype)
    dst.flush()

    ids = Path(str(args.embeddings) + ".ids")
    if ids.exists():
        shutil.copyfile(ids, str(out) + ".ids")

    rng = np.random.default_rng(0)
    n = min(10_000, src.shape[0])
    a, b = rng.integers(0, src.shape[0], n), rng.integers(0, src.shape[0], n)
    err = np.abs(cosine(src[a], src[b]) - cosine(dst[a], dst[b]))
    print(f"wrote {out}: {src.shape} {src.dtype} -> {dst.dtype}, "
          f"{src.nbytes / 2**20:.1f} MB -> {dst.nbytes / 2**20:.1f} MB")
    print(f"cosine change over {n} random pairs: mean {err.mean():.2e}, max {err.max():.2e}")


if __name__ == "__main__":
    main()
