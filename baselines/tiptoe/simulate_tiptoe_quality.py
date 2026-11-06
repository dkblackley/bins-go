#!/usr/bin/env python3
"""
Plaintext replay of what the Tiptoe client ends up with, on a corpus made by
preprocess_tiptoe.py. Tiptoe's crypto is exact (SimplePIR inner products and
PIR both decrypt to the true values), so this gives the same ranking as the
real run in tiptoe/tiptoe.go, minus the order of exactly-tied scores.

Use it to (a) get Tiptoe's MRR/Recall without SEAL, and (b) sanity-check a
real -t tiptoe run: MRRPreReRank should match to ~3 decimals.

  python baselines/tiptoe/simulate_tiptoe_quality.py --dataset ../datasets --name scifact \
      --tiptoe-dir ../datasets/tiptoe/scifact_dpc500 --k 10
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402

SUBCLUSTER_DELIM = "-------------------------"


def read_cluster(path):
    """[(row, emb int8[dim], bundle_index)] in file order (= Tiptoe's within-cluster doc order)."""
    out, bundle = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == SUBCLUSTER_DELIM:
                bundle += 1
                continue
            row, emb, _ = line.split(" | ")
            out.append((int(row), np.array(emb.split(","), dtype=np.int64), bundle))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../datasets")
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--tiptoe-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=0)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.tiptoe_dir, "tiptoe_meta.json")))
    cents = np.load(os.path.join(args.tiptoe_dir, "centroids.npy"))
    paths = common.dataset_paths(args.dataset, args.name)
    doc_ids = [d[0] for d in common.load_corpus(paths["corpus"])]
    queries = common.load_queries(paths["queries"])
    qvecs = np.load(paths["query_vec"], mmap_mode="r")
    qrels = common.load_qrels(paths["qrels"])
    if args.max_queries:
        queries = queries[:args.max_queries]

    lo, hi = -(1 << (meta["slot_bits"] - 1)), (1 << (meta["slot_bits"] - 1)) - 1
    cache, results = {}, {}
    for i, (qid, _) in enumerate(queries):
        v = np.asarray(qvecs[i], dtype=np.float32)
        c = int(np.argmin(((cents - v) ** 2).sum(1)))                    # nearestCluster
        q = np.clip(np.round(v.astype(np.float64) * meta["scale"]), lo, hi).astype(np.int64)  # quantise
        if c not in cache:
            cache[c] = read_cluster(os.path.join(args.tiptoe_dir, "clusters", f"cluster_{c}.txt"))
        docs = cache[c]
        scores = np.array([int(e @ q) for _, e, _ in docs])
        order = np.argsort(-scores, kind="stable")
        best_bundle = docs[order[0]][2]                                   # bundle of the best doc
        ids, seen = [], set()
        for j in order:                                                   # that bundle's docs, by score
            row, _, b = docs[j]
            if b != best_bundle or doc_ids[row] in seen:
                continue
            seen.add(doc_ids[row])
            ids.append(doc_ids[row])
            if len(ids) == args.k:
                break
        results[qid] = ids

    print(json.dumps({
        "tiptoe_dir": args.tiptoe_dir, "num_clusters": meta["num_clusters"], "k": args.k,
        "NumQueries": len(results),
        "MRRPreReRank": round(common.calc_mrr(results, qrels), 4),
        "RecallPreReRank": round(common.calc_recall(results, qrels), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
