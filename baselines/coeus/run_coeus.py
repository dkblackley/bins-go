#!/usr/bin/env python3
"""
Coeus (Ahmad et al., SOSP'21) baseline, written to the same metadata.json format
as the Go runs so plotting/load_results.py picks it up as method "coeus".

Why this is not a Go wrapper like Tiptoe
-----------------------------------------
Coeus' artifact (github.com/ishtiyaque/Coeus_artifact) is C++ on SEAL 3.2/3.5
and only benchmarks its homomorphic pieces on synthetic matrices: it never
ranks real documents, and the end-to-end runs need a master + up to 96 AWS
workers. So the baseline is split into the two things that actually decide
Coeus' numbers:

1. QUALITY, exact. Coeus scores every document as  score(d) = q . tfidf(d)
   with q a 0/1 keyword vector, homomorphically but without approximation
   (BFV is exact for this product), then the client takes the top-k. So its
   MRR/Recall are exactly those of plaintext top-k TF-IDF over the same
   keyword dictionary, which is what this script computes on the real corpus.

2. COST, from Coeus' own code. Communication comes from the encryption
   parameters hard-coded in the artifact (step 1 = matmult/opt2.cpp:
   N=8192, 3x60-bit primes; step 2 = step2/common/globals.h: SealPIR with
   N=2048, logt=15, d=2, 1.5k cuckoo buckets). Server time comes from running
   the artifact's own single-machine benchmark `matmult/bin/opt2 -r R -c C`
   on a few small (R, C) grids on YOUR machine (--opt2-bin) and fitting
   T(R, C) = a*R*C + b*R + c*C + d, then evaluating it at the real
   R = ceil(N_docs/8192), C = ceil(F/8192). (One 8192x8192 block is ~1.6 GB of
   NTT plaintexts, so MS MARCO's full matrix would need terabytes of RAM; the
   paper spreads it over many machines for the same reason.)
   Without --opt2-bin the time keys are left out (NaN in the plots), never guessed.

Step 3 of Coeus (fetching the documents themselves) is not counted, because
none of the other methods in this repo count document download either.

Outputs <results-root>/coeus_<dataset>_k<K>_f<F>/{metadata.json,results.json,results_reRank.json}

Usage
-----
  python baselines/coeus/run_coeus.py --dataset ../datasets --name scifact --k 10
  python baselines/coeus/run_coeus.py --dataset ../datasets --name msmarco --k 10 \
      --opt2-bin ~/Coeus_artifact/matmult/bin/opt2 --parallelism 16
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402

# ---------------- Coeus parameters, copied from the artifact ----------------
# step 1 (matmult/opt2.cpp): BFV, N=8192, coeff modulus {60,60,60}, batching
S1_N = 8192
S1_PRIMES = 3
S1_CT_BYTES = 2 * S1_N * S1_PRIMES * 8          # 2 polys, 64-bit words: 393,216 B
S1_PT_NTT_BYTES = S1_N * S1_PRIMES * 8          # one NTT plaintext (server memory)
S1_GALOIS_STEPS = 1 + int(math.log2(S1_N // 2))  # steps 0,1,2,...,2048 -> 13 keys
S1_GALOIS_BYTES = S1_GALOIS_STEPS * (S1_PRIMES - 1) * S1_CT_BYTES  # one key-switch ct per decomposed prime
S1_SK_BYTES = S1_N * S1_PRIMES * 8
# step 2 (step2/common/globals.h + pir.cpp): SealPIR, N=2048, logt=15, d=2, one 60-bit prime
S2_CT_BYTES = 32841                              # CIPHER_SIZE
S2_D = 2
S2_EXPANSION = 2 * math.ceil(60 / 15)            # pir_params.expansion_ratio = 8
S2_BUCKET_FACTOR = 1.5                           # FACTOR
S2_GALOIS_BYTES = int(math.log2(2048)) * S2_CT_BYTES  # SealPIR query-expansion keys, one-time

STOPWORDS = set("""a an and are as at be but by for if in into is it no not of on or such that the their then there
these they this to was will with""".split())   # Lucene's default English stop set


def make_tokenizer(kind):
    if kind in ("auto", "lucene"):
        try:
            from pyserini.analysis import Analyzer, get_lucene_analyzer
            an = Analyzer(get_lucene_analyzer())
            return an.analyze, "lucene"
        except Exception as err:  # pyserini/java not available
            if kind == "lucene":
                raise SystemExit(f"pyserini analyzer unavailable: {err}")
    tok = re.compile(r"[a-z0-9]+")
    return (lambda s: [t for t in tok.findall(s.lower()) if t not in STOPWORDS]), "regex"


def build_tfidf(docs, features, tokenize, tf_mode):
    """CSC-style term -> (doc rows, weights) for the top-`features` terms by document frequency."""
    vocab, doc_terms = {}, []
    t0 = time.time()
    for i, (_, title, text) in enumerate(docs):
        ids = [vocab.setdefault(t, len(vocab)) for t in tokenize(f"{title} {text}")]
        doc_terms.append(np.asarray(ids, dtype=np.int32))
        if i % 500_000 == 0 and i:
            print(f"  tokenised {i}/{len(docs)} ({time.time() - t0:.0f}s)", flush=True)
    df = np.zeros(len(vocab), dtype=np.int64)
    for ids in doc_terms:
        df[np.unique(ids)] += 1
    keep = np.argsort(-df, kind="stable")[:features]
    remap = np.full(len(vocab), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    idf = np.log(len(docs) / np.maximum(df[keep], 1)).astype(np.float64)

    rows, terms, tfs = [], [], []
    for r, ids in enumerate(doc_terms):
        ids = remap[ids]
        ids = ids[ids >= 0]
        if not len(ids):
            continue
        u, c = np.unique(ids, return_counts=True)
        rows.append(np.full(len(u), r, dtype=np.int32))
        terms.append(u.astype(np.int32))
        tfs.append(c.astype(np.float32))
    rows, terms, tfs = np.concatenate(rows), np.concatenate(terms), np.concatenate(tfs)
    if tf_mode == "log":
        tfs = 1 + np.log(tfs)
    w = tfs * idf[terms]
    order = np.argsort(terms, kind="stable")
    rows, terms, w = rows[order], terms[order], w[order]
    ptr = np.searchsorted(terms, np.arange(len(keep) + 1))
    term_id = {t: remap[i] for t, i in vocab.items() if remap[i] >= 0}
    return term_id, ptr, rows, w.astype(np.float64)


def coeus_topk(query, k, n_docs, tokenize, term_id, ptr, rows, w):
    """Exactly what the client gets after decrypting Coeus' scores: the top-k of q . tfidf(d)."""
    scores = np.zeros(n_docs)
    q = {term_id[t] for t in tokenize(query) if t in term_id}   # binary query vector
    for t in q:
        s, e = ptr[t], ptr[t + 1]
        scores[rows[s:e]] += w[s:e]
    if not q:
        return []
    k = min(k, n_docs)
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.lexsort((top, -scores[top]))]   # by score, ties by doc order
    return [int(i) for i in top if scores[i] > 0]


# ---------------- cost model ----------------

def calibrate_opt2(binary, cache, grid=((1, 1), (1, 2), (2, 1), (2, 2)), reps=1):
    """Runs Coeus' own single-machine benchmark and fits T(R,C) = a*R*C + b*R + c*C + d (seconds)."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            cal = json.load(f)
        print(f"opt2 calibration from {cache}: {cal['coef']}")
        return cal
    pts = []
    for r, c in grid:
        for _ in range(reps):
            out = subprocess.run([binary, "-r", str(r), "-c", str(c)], capture_output=True, text=True, check=True).stdout
            m = re.search(r"Total time:\s*\n?\s*(\d+)", out)
            if not m:
                raise SystemExit(f"could not parse opt2 output:\n{out}")
            us = int(m.group(1))
            pts.append((r, c, us / 1e6))
            print(f"  opt2 -r {r} -c {c}: {us / 1e6:.2f}s", flush=True)
    A = np.array([[r * c, r, c, 1.0] for r, c, _ in pts])
    y = np.array([t for _, _, t in pts])
    coef = np.linalg.lstsq(A, y, rcond=None)[0].tolist()
    cal = {"binary": binary, "points": pts, "coef": coef, "model": "a*R*C + b*R + c*C + d"}
    if cache:
        with open(cache, "w") as f:
            json.dump(cal, f, indent=2)
    return cal


def coeus_cost(n_docs, features, k):
    R = math.ceil(n_docs / S1_N)        # response ciphertexts (8192 scores each)
    C = math.ceil(features / S1_N)      # query ciphertexts (8192 keywords each)
    buckets = math.ceil(S2_BUCKET_FACTOR * k)
    s1_up, s1_down = C * S1_CT_BYTES, R * S1_CT_BYTES
    s2_up = buckets * S2_D * S2_CT_BYTES
    s2_down = buckets * (S2_EXPANSION ** (S2_D - 1)) * S2_CT_BYTES
    return {
        "R": R, "C": C, "buckets": buckets,
        "s1_up": s1_up, "s1_down": s1_down, "s2_up": s2_up, "s2_down": s2_down,
        "per_query": s1_up + s1_down + s2_up + s2_down,
        "setup_upload": S1_GALOIS_BYTES + S2_GALOIS_BYTES,
        "client_state": S1_SK_BYTES + S1_GALOIS_BYTES + S2_GALOIS_BYTES,
        "server_db": R * C * S1_N * S1_PT_NTT_BYTES,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../datasets")
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--features", type=int, default=65536,
                    help="keyword dictionary size F (Coeus pads to a multiple of 8192; paper used 65,536)")
    ap.add_argument("--tf", choices=["raw", "log"], default="raw")
    ap.add_argument("--tokenizer", choices=["auto", "lucene", "regex"], default="auto",
                    help="auto = pyserini's Lucene analyzer (same as our BM25) if installed")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--results-root", default="../results")
    ap.add_argument("--opt2-bin", default="", help="Coeus_artifact/matmult/bin/opt2, to measure server time")
    ap.add_argument("--opt2-cache", default="", help="JSON cache for the opt2 calibration (default next to the binary)")
    ap.add_argument("--parallelism", type=int, default=1,
                    help="divide step-1 server time by this many workers (Coeus scales ~linearly, paper Fig. 9)")
    args = ap.parse_args()

    paths = common.dataset_paths(args.dataset, args.name)
    print(f"loading {paths['corpus']}", flush=True)
    docs = common.load_corpus(paths["corpus"])
    queries = common.load_queries(paths["queries"])
    qrels = common.load_qrels(paths["qrels"])
    if args.max_queries:
        queries = queries[:args.max_queries]
    tokenize, tok_name = make_tokenizer(args.tokenizer)
    print(f"{len(docs)} docs, {len(queries)} queries, tokenizer={tok_name}", flush=True)

    t0 = time.time()
    term_id, ptr, rows, w = build_tfidf(docs, args.features, tokenize, args.tf)
    pre_time = time.time() - t0
    print(f"tf-idf built over {len(term_id)} keywords in {pre_time:.1f}s", flush=True)

    results = {}
    for qid, text in queries:
        results[qid] = [docs[i][0] for i in coeus_topk(text, args.k, len(docs), tokenize, term_id, ptr, rows, w)]

    # Same local cosine rerank every other method gets in main.go
    doc_vecs = np.load(paths["corpus_vec"], mmap_mode="r")
    qvecs = np.load(paths["query_vec"], mmap_mode="r")
    doc_index = {d[0]: i for i, d in enumerate(docs)}
    qvec_by_id = {qid: np.asarray(qvecs[i], dtype=np.float32) for i, (qid, _) in enumerate(common.load_queries(paths["queries"]))
                  if i < len(qvecs)}
    reranked = common.cosine_rerank(results, args.k, doc_index, doc_vecs, qvec_by_id)

    n_q = len(results)
    cost = coeus_cost(len(docs), len(term_id), args.k)   # F = keywords actually in the dictionary
    meta = {
        "Method": "coeus",
        "NumQueries": n_q,
        "MRRPreReRank": f"{common.calc_mrr(results, qrels):.4f}",
        "RecallPreReRank": f"{common.calc_recall(results, qrels):.4f}",
        "MRR": f"{common.calc_mrr(reranked, qrels):.4f}",
        "Recall": f"{common.calc_recall(reranked, qrels):.4f}",
        "ClientStorageMB": cost["client_state"] / 2**20,
        "DBSizeInBytesMB": cost["server_db"] / 2**20,
        "CommCostPerBatchKB": cost["per_query"] / 1024,
        "TotalByteSent": cost["per_query"] * n_q,
        "TotalLANTime": n_q * common.net_time(cost["per_query"], 2, 1000, 0.005),
        "TotalWANTime": n_q * common.net_time(cost["per_query"], 2, 400, 0.05),
        "CoeusFeatures": args.features, "CoeusKeywordsUsed": len(term_id),
        "CoeusR": cost["R"], "CoeusC": cost["C"], "CoeusBuckets": cost["buckets"],
        "CoeusStep1UpBytes": cost["s1_up"], "CoeusStep1DownBytes": cost["s1_down"],
        "CoeusStep2UpBytes": cost["s2_up"], "CoeusStep2DownBytes": cost["s2_down"],
        "CoeusSetupUploadMB": cost["setup_upload"] / 2**20,
        "CoeusTokenizer": tok_name, "CoeusTF": args.tf,
        "CoeusTfidfBuildTime": f"{pre_time:.3f}",
        "CoeusParallelism": args.parallelism,
        "CoeusTimeSource": "none",
    }
    if args.opt2_bin:
        cache = args.opt2_cache or os.path.join(os.path.dirname(os.path.abspath(args.opt2_bin)), "opt2_calibration.json")
        cal = calibrate_opt2(args.opt2_bin, cache)
        a, b, c, d = cal["coef"]
        t_s1 = max(0.0, a * cost["R"] * cost["C"] + b * cost["R"] + c * cost["C"] + d) / args.parallelism
        meta["TotalAnswerTime"] = f"{t_s1 * n_q:.6f}"
        meta["CoeusStep1SecondsPerQuery"] = f"{t_s1:.6f}"
        meta["CoeusTimeSource"] = f"opt2 fit {cal['model']} coef={cal['coef']}"
        meta["CoeusStep2TimeIncluded"] = "false"
    else:
        print("no --opt2-bin: leaving TotalAnswerTime out (plots show NaN for Coeus time)")

    out = os.path.join(args.results_root, f"coeus_{args.name}_k{args.k}_f{args.features}")
    common.write_run(out, meta, results, reranked)
    print(json.dumps(meta, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
