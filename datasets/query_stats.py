#!/usr/bin/env python3
"""Query and doc length (whitespace words) for prepared datasets, e.g. to check queries stay under ~10-15 words.

    python datasets/query_stats.py ../datasets/msmarco ../datasets/scifact ../datasets/nq ../datasets/dbpedia-entity

Reads <dir>/queries.jsonl (or queries.dev.small.tsv / any *.tsv given directly) and, unless --no-docs, samples
<dir>/corpus.jsonl. Only queries with qrels are counted when <dir>/qrels/test.tsv exists.
"""
import argparse, glob, json, os, random
import numpy as np


def stats(n):
    n = np.asarray(n)
    return (f"mean {n.mean():5.1f}  median {np.median(n):4.0f}  p90 {np.percentile(n, 90):4.0f}  max {n.max():4d}  "
            f"<=10: {100 * (n <= 10).mean():5.1f}%  <=15: {100 * (n <= 15).mean():5.1f}%  <=20: {100 * (n <= 20).mean():5.1f}%")


def load_queries(path):
    if path.endswith(".tsv"):
        return {l.split("\t", 1)[0]: l.rstrip("\n").split("\t", 1)[1] for l in open(path) if "\t" in l}
    return {d["_id"]: d["text"] for d in map(json.loads, open(path))}


ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--no-docs", action="store_true")
ap.add_argument("--doc-sample", type=int, default=100000)
args = ap.parse_args()

for p in args.paths:
    qfile = p if os.path.isfile(p) else next(iter(glob.glob(f"{p}/queries.jsonl") + glob.glob(f"{p}/queries*.tsv")), None)
    if not qfile:
        print(f"{p}: no queries file"); continue
    qs = load_queries(qfile)
    qrels = f"{p}/qrels/test.tsv"
    if os.path.isdir(p) and os.path.exists(qrels):
        want = {l.split()[0] for l in open(qrels)}
        qs = {k: v for k, v in qs.items() if k in want} or qs
    print(f"{p}\n  queries ({len(qs):,}): {stats([len(t.split()) for t in qs.values()])}")
    corpus = f"{p}/corpus.jsonl"
    if not args.no_docs and os.path.isdir(p) and os.path.exists(corpus):
        lens = []
        with open(corpus) as f:
            for i, l in enumerate(f):
                if i >= args.doc_sample:
                    break
                d = json.loads(l)
                lens.append(len((d.get("title", "") + " " + d.get("text", "")).split()))
        print(f"  docs (first {len(lens):,}): {stats(lens)}")
