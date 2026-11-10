#!/usr/bin/env python3
"""Bring embeddings made on another machine (datasets/embed_local.sh, stored as float16) into a server dataset dir.

Copy the local files into <dataset dir>/local_import/ first (see datasets/README.md "Embedding locally"), then
    python datasets/import_local_embeds.py --dataset-dir ../datasets/nq --name nq
(datasets/import_local_embeds.sh does this for several datasets, also as an sbatch job).

local_import/ holds either
  * the whole corpus:  <name>_corpus_embed_f16.npy(.ids)  + <name>_query_embed_f32.npy + pca_e5_768_to_192.faiss
    -> <name>_corpus_embed_f32.npy(.ids), the query file and the PCA are put in place
  * ALL shards of one split (e.g. 48/48 from datasets/colab_msmarco_v2.ipynb) + the PCA they were made with
    -> merged straight into <name>_corpus_embed_f32.npy(.ids); the PCA is put in place and, if no query file came
       with them, the queries are embedded here with that PCA (query_preproc.py; a few minutes on CPU)
  * some shards:       <name>_corpus_embed_f16.npy.shardXXXofYYY.npy(.ids)  (+ the PCA they were made with, which
    must be the server's: shards from different PCAs can't be merged)
    -> <name>_corpus_embed_f32.npy.shardXXXofYYY.npy(.ids) + .done, so the server's shard jobs skip them

float16 -> float32 is exact (every fp16 value is a fp32 value). Rows and ids are checked against the server's
corpus.jsonl, so a corpus prepared differently on the two machines is caught before anything is replaced.

It joins the same race as the slurm embed jobs: it claims <output>.done first, so if a server job already finished
the output nothing is touched (exit 3), and a server job still running will throw its copy away when it ends.
"""
import argparse
import filecmp
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from numpy.lib.format import open_memmap

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--src", default=None, help="Where the local files are (default <dataset dir>/local_import)")
ap.add_argument("--no-check", action="store_true", help="Skip comparing ids with corpus.jsonl")
ap.add_argument("--keep-src", action="store_true", help="Keep the float16 files after a successful import")
args = ap.parse_args()

D, name = args.dataset_dir, args.name
prefix = "trec" if name == "trec-covid" else name
src = args.src or f"{D}/local_import"
f32 = f"{D}/{prefix}_corpus_embed_f32.npy"
pca_dst = f"{D}/pca_e5_768_to_192.faiss"
pca_src = f"{src}/pca_e5_768_to_192.faiss"
ALREADY_DONE = 3


def corpus_ids(path, lo=0, hi=None):
    """_id of corpus lines [lo, hi), without json-parsing every line (prepare_dataset.py writes `{"_id": "..."`)."""
    with open(path, encoding="utf-8", buffering=16 << 20) as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if hi is not None and i >= hi:
                break
            if line.startswith('{"_id": "'):
                yield line[9:line.index('"', 9)]
            else:
                yield json.loads(line)["_id"]


def check_ids(ids_path, lo, hi):
    with open(ids_path) as f:
        for n, (a, b) in enumerate(zip(f, corpus_ids(f"{D}/corpus.jsonl", lo, hi))):
            if a.rstrip("\n") != b:
                sys.exit(f"{ids_path} line {n}: id {a.strip()!r} but the server's corpus has {b!r} at row {lo + n}. "
                         f"The corpus was prepared differently here; re-embed on the server instead.")


def claim(path):
    try:
        os.close(os.open(path + ".done", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def widen(f16_path, dst, rows):
    """float16 .npy -> float32 .npy, 1M rows at a time, via a temp file + rename."""
    a = np.load(f16_path, mmap_mode="r")
    assert a.shape[0] == rows, f"{f16_path} has {a.shape[0]:,} rows, expected {rows:,}"
    tmp = f"{dst}.import{os.getpid()}.npy"
    out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=a.shape)
    for i in range(0, rows, 1_000_000):
        out[i:i + 1_000_000] = a[i:i + 1_000_000]
    del out
    shutil.copyfile(f16_path + ".ids", tmp + ".ids")
    return tmp


def complete_split(paths):
    """N if paths hold every shard 0..N-1 of one N-way split, else None."""
    seen = {}
    for p in paths:
        i, n = (int(x) for x in p.split(".npy.shard", 1)[1][:-4].split("of"))
        seen.setdefault(n, set()).add(i)
    return next((n for n, ii in seen.items() if ii == set(range(n))), None)


num_docs = int(open(f"{D}/num_docs.txt").read())
whole = f"{src}/{prefix}_corpus_embed_f16.npy"
shards = sorted(p for p in glob.glob(f"{src}/{prefix}_corpus_embed_f16.npy.shard*of*.npy") if ".partial-" not in p)

if os.path.exists(f32 + ".done"):
    print(f"{f32} is already done on the server; nothing imported")
    sys.exit(ALREADY_DONE)

if os.path.exists(whole):
    for f in (whole + ".ids", f"{src}/{prefix}_query_embed_f32.npy", pca_src):
        if not os.path.exists(f):
            sys.exit(f"missing {f}: the corpus embeddings are useless without the queries + PCA they were made with")
    if not args.no_check:
        print(f"checking {num_docs:,} ids against {D}/corpus.jsonl ...")
        check_ids(whole + ".ids", 0, num_docs)
    print(f"widening {whole} -> {f32}")
    tmp = widen(whole, f32, num_docs)
    if not claim(f32):
        os.remove(tmp); os.remove(tmp + ".ids")
        print(f"a server job finished {f32} meanwhile; keeping the server's copy")
        sys.exit(ALREADY_DONE)
    shutil.copyfile(pca_src, pca_dst + ".import")
    os.replace(pca_dst + ".import", pca_dst)
    shutil.copyfile(f"{src}/{prefix}_query_embed_f32.npy", f"{D}/{prefix}_query_embed_f32.npy.import")
    os.replace(f"{D}/{prefix}_query_embed_f32.npy.import", f"{D}/{prefix}_query_embed_f32.npy")
    os.replace(tmp + ".ids", f32 + ".ids")
    os.replace(tmp, f32)
    print(f"imported {name}: {f32} ({num_docs:,} x 192 float32), queries and PCA")
    if not args.keep_src:
        for f in (whole, whole + ".ids", whole + ".done"):
            if os.path.exists(f):
                os.remove(f)

elif shards and (split := complete_split(shards)):
    n = split
    if not os.path.exists(pca_src):
        sys.exit(f"missing {pca_src}: the shards are useless without the PCA they were made with")
    q_src = f"{src}/{prefix}_query_embed_f32.npy"
    if not os.path.exists(q_src):   # embed the queries with the shards' PCA (before claiming, so nothing is half-done)
        print("no query embeddings came with the shards; embedding the queries here with their PCA")
        subprocess.run([sys.executable, "query_preproc.py", "--queries", f"{D}/queries.jsonl", "--output", q_src,
                        "--pca", pca_src], check=True)
    per = (num_docs + n - 1) // n
    ids_gen = None if args.no_check else corpus_ids(f"{D}/corpus.jsonl")
    tmp = f"{f32}.import{os.getpid()}.npy"
    out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(num_docs, 192))
    with open(tmp + ".ids", "w") as fid:
        for i in range(n):
            s_path = f"{src}/{prefix}_corpus_embed_f16.npy.shard{i:03d}of{n:03d}.npy"
            lo, hi = i * per, min(num_docs, (i + 1) * per)
            a = np.load(s_path, mmap_mode="r")
            assert a.shape == (hi - lo, 192), f"{s_path} is {a.shape}, expected {(hi - lo, 192)}"
            for j in range(0, hi - lo, 1_000_000):
                out[lo + j: lo + min(j + 1_000_000, hi - lo)] = a[j:j + 1_000_000]
            with open(s_path + ".ids") as f:
                for k, line in enumerate(f):
                    if ids_gen is not None and line.rstrip("\n") != (want := next(ids_gen)):
                        sys.exit(f"{s_path}.ids line {k}: {line.strip()!r} but the server's corpus has {want!r} "
                                 f"at row {lo + k}; the corpus differs, not importing")
                    fid.write(line)
            print(f"  shard {i + 1}/{n}")
    del out
    if not claim(f32):
        os.remove(tmp); os.remove(tmp + ".ids")
        print(f"a server job finished {f32} meanwhile; keeping the server's copy")
        sys.exit(ALREADY_DONE)
    shutil.copyfile(pca_src, pca_dst + ".import")
    os.replace(pca_dst + ".import", pca_dst)
    shutil.copyfile(q_src, f"{D}/{prefix}_query_embed_f32.npy.import")
    os.replace(f"{D}/{prefix}_query_embed_f32.npy.import", f"{D}/{prefix}_query_embed_f32.npy")
    os.replace(tmp + ".ids", f32 + ".ids")
    os.replace(tmp, f32)
    print(f"imported {name} from {n} shards: {f32} ({num_docs:,} x 192 float32), queries and PCA")
    if not args.keep_src:
        for s_path in shards:
            for f in (s_path, s_path + ".ids", s_path + ".done"):
                if os.path.exists(f):
                    os.remove(f)

elif shards:
    if not os.path.exists(pca_dst) or not os.path.exists(pca_src) or not filecmp.cmp(pca_src, pca_dst, shallow=False):
        sys.exit(f"shards need the server's PCA: copy {pca_dst} to the local dataset dir before embedding the shards, "
                 f"and put that same file in {src}/")
    for s in shards:
        tail = s.split(".npy.shard", 1)[1]                 # XXXofYYY.npy
        i, n = (int(x) for x in tail[:-4].split("of"))
        per = (num_docs + n - 1) // n
        lo, hi = i * per, min(num_docs, (i + 1) * per)
        dst = f"{f32}.shard{tail}"
        if os.path.exists(dst + ".done"):
            print(f"{dst} already done on the server, skipping"); continue
        if not args.no_check:
            check_ids(s + ".ids", lo, hi)
        tmp = widen(s, dst, hi - lo)
        if not claim(dst):
            os.remove(tmp); os.remove(tmp + ".ids"); print(f"{dst} finished on the server meanwhile"); continue
        os.replace(tmp + ".ids", dst + ".ids")
        os.replace(tmp, dst)
        print(f"imported shard {i}/{n} -> {dst}")
        if not args.keep_src:
            for f in (s, s + ".ids", s + ".done"):
                if os.path.exists(f):
                    os.remove(f)
else:
    sys.exit(f"nothing to import in {src} (expected {os.path.basename(whole)} or its shards)")
