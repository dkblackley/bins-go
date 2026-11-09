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

2. COST, from Coeus' own code. Communication is what the artifact's code sends
   and counts (step 1: seeded query ciphertexts, raw 2-prime response
   ciphertexts; step 2: SealPIR with N=2048, logt=15, d=2, 1.5k cuckoo
   buckets, CIPHER_SIZE-byte ciphertexts). Server time is the artifact's own
   single-machine kernel `matmult/bin/opt2 -r R -c C` (R = ceil(N_docs/8192)
   response ciphertexts, C = ceil(F/8192) query ciphertexts):
     * run DIRECTLY at the real (R, C) when the encoded matrix fits in
       --opt2-direct-max-gb (one 8192x8192 block is 1 GiB of NTT plaintexts),
       e.g. SciFact; otherwise
     * fitted as T(R, C) = a*R*C + b*R + c*C + d on small grids and evaluated
       at the real (R, C). MS MARCO's matrix is ~8.4 TiB, which is why the
       paper spreads it over many machines.
   Without --opt2-bin the time keys are left out (NaN in the plots), never guessed.

Step 3 of Coeus (fetching the documents themselves) is not counted, because
none of the other methods in this repo count document download either.

Outputs <results-root>/coeus_<dataset>_k<K>_f<F>_tf<raw|log>/{metadata.json,results.json,results_reRank.json}

Usage
-----
  python baselines/coeus/run_coeus.py --dataset ../datasets --name scifact --k 10
  python baselines/coeus/run_coeus.py --dataset ../datasets --name msmarco --k 10 \
      --opt2-bin ~/Coeus_artifact/matmult/bin/opt2 --parallelism 16
"""

import argparse
from array import array
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
# Sizes follow what the artifact's own code sends and counts (its client tallies total_upload/total_download).
# Where its prototype re-sends something per query that a deployment would only send once (the Galois keys,
# in both steps), we count it once, as a setup cost: the best case for Coeus that is still accurate.
#
# step 1 (step1/common: N=8192, coeff modulus {60,60,60}, PLAIN_BIT 46). The last 60-bit prime is SEAL's
# special prime, so ciphertexts and NTT plaintexts live at the 2-prime data level.
S1_N = 8192
S1_DATA_PRIMES = 2
# query: step1/client/merge.cpp encrypts with encrypt_symmetric -> Serializable (seeded): only c0 is sent,
# plus a 64 B seed. Saved with SEAL's default compression, which shrinks uniformly random 60-bit
# coefficients stored in 64-bit words to ~60/64 of the raw size.
S1_QUERY_CT_BYTES = S1_N * S1_DATA_PRIMES * 60 // 8 + 64        # 122,944 B
# response: step1/server/worker SEND_TO_CLIENT ships raw uint64 vectors of N*2*2 words per ciphertext, and the
# client counts total_download += N*2*2*8 per ciphertext.
S1_RESP_CT_BYTES = S1_N * 2 * 2 * 8                              # 262,144 B
S1_PT_NTT_BYTES = S1_N * S1_DATA_PRIMES * 8                      # one NTT plaintext in server memory
# Galois keys (Serializable, deflate): steps {0, 1, 2, ..., 4096/2} = 13 keys, each holding one seeded
# key-switching ciphertext (1 poly at all 3 primes) per data prime.
S1_GALOIS_STEPS = 1 + int(math.log2(S1_N // 2))
S1_GALOIS_BYTES = S1_GALOIS_STEPS * S1_DATA_PRIMES * (S1_N * 3 * 60 // 8 + 64)
S1_SK_BYTES = S1_N * 3 * 8                                       # secret key (all 3 primes)
# step 2 (step2/common/globals.h + pir.cpp): SealPIR over SEAL 3.2, N=2048, logt=15, d=2, one 60-bit prime,
# 1.5k cuckoo buckets, every ciphertext serialised to CIPHER_SIZE bytes.
S2_CT_BYTES = 32841                              # CIPHER_SIZE
S2_D = 2
S2_EXPANSION = 2 * math.ceil(60 / 15)            # pir_params.expansion_ratio = 8 -> reply = 8^(d-1) cts per bucket
S2_BUCKET_FACTOR = 1.5                           # FACTOR
S2_GALOIS_BYTES = int(math.log2(2048)) * math.ceil(60 / 6) * S2_CT_BYTES  # 11 keys x (60/dbc=6) parts, one-time

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


def build_tfidf(corpus_path, features, tokenize, tf_mode, tmp_dir):
    """CSC-style term -> (doc rows, weights) for the top-`features` terms by document frequency.

    Streams the corpus once: token ids go to a temporary int32 file, so nothing but the vocabulary and the final
    postings (12 B each) is held in memory. That is what makes msmarco-v2 (138M docs) possible on one node.
    Returns (term_id, ptr, rows, w, n_docs); rows are in doc order within each term."""
    vocab, df, lens = {}, np.zeros(1 << 20, dtype=np.int64), array("I")
    os.makedirs(tmp_dir, exist_ok=True)
    tok_path = os.path.join(tmp_dir, f"coeus_tokens_{os.getpid()}.bin")
    t0 = time.time()
    try:
        with open(tok_path, "wb") as out:
            buf = []
            for n, (_, title, text) in enumerate(common.iter_corpus(corpus_path)):
                ids = np.fromiter((vocab.setdefault(t, len(vocab)) for t in tokenize(f"{title} {text}")), dtype=np.int32)
                if len(vocab) > len(df):
                    df = np.concatenate([df, np.zeros(len(df), dtype=np.int64)])
                if len(ids):
                    df[np.unique(ids)] += 1
                lens.append(len(ids))
                buf.append(ids)
                if len(buf) >= 100_000:
                    out.write(np.concatenate(buf).tobytes())
                    buf = []
                if n % 1_000_000 == 0 and n:
                    print(f"  tokenised {n} docs ({time.time() - t0:.0f}s, {len(vocab)} terms)", flush=True)
            if buf:
                out.write(np.concatenate(buf).tobytes())
        n_docs = len(lens)
        df = df[:len(vocab)]
        keep = np.argsort(-df, kind="stable")[:features]
        nkeep = len(keep)
        remap = np.full(len(vocab), -1, dtype=np.int64)
        remap[keep] = np.arange(nkeep)
        idf = np.log(n_docs / np.maximum(df[keep], 1)).astype(np.float64)

        # Fill the postings in place: term t's docs go to rows[ptr[t]:ptr[t+1]], in doc order
        ptr = np.concatenate([[0], np.cumsum(df[keep])]).astype(np.int64)
        rows, w = np.empty(ptr[-1], dtype=np.int32), np.empty(ptr[-1], dtype=np.float64)
        cursor = ptr[:-1].copy()
        lens = np.frombuffer(lens, dtype=np.uint32).astype(np.int64)
        offs = np.concatenate([[0], np.cumsum(lens)])
        tokens = np.memmap(tok_path, dtype=np.int32, mode="r") if offs[-1] else np.zeros(0, dtype=np.int32)
        step = 200_000
        for a in range(0, n_docs, step):
            b = min(a + step, n_docs)
            tok = remap[tokens[offs[a]:offs[b]]]
            doc = np.repeat(np.arange(b - a, dtype=np.int64), lens[a:b])
            m = tok >= 0
            if not m.any():
                continue
            uk, cnt = np.unique(doc[m] * nkeep + tok[m], return_counts=True)   # sorted by (doc, term)
            d, t = uk // nkeep + a, uk % nkeep
            tf = cnt.astype(np.float32)
            if tf_mode == "log":
                tf = 1 + np.log(tf)
            o = np.argsort(t, kind="stable")                                     # by term, doc order kept
            d, t, wt = d[o], t[o], tf[o] * idf[t[o]]
            per_t = np.bincount(t, minlength=nkeep)
            first = np.cumsum(per_t) - per_t
            pos = cursor[t] + np.arange(len(t)) - first[t]
            rows[pos], w[pos] = d, wt
            cursor += per_t
        del tokens
    finally:
        if os.path.exists(tok_path):
            os.remove(tok_path)
    term_id = {t: remap[i] for t, i in vocab.items() if remap[i] >= 0}
    return term_id, ptr, rows, w, n_docs


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

def run_opt2(binary, r, c):
    out = subprocess.run([binary, "-r", str(r), "-c", str(c)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Total time:\s*\n?\s*(\d+)", out)
    if not m:
        raise SystemExit(f"could not parse opt2 output:\n{out}")
    return int(m.group(1)) / 1e6


def write_json_atomic(path, data):
    """Several slurm tasks share the cache: write a temp file and rename, so nobody reads half a file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def opt2_direct(binary, cache, r, c):
    """Runs opt2 at the real (R, C) once and caches it (keyed by R,C) next to the calibration."""
    data = json.load(open(cache)) if cache and os.path.exists(cache) else {}
    key = f"{r}x{c}"
    if key not in data.get("direct", {}):
        t = run_opt2(binary, r, c)
        print(f"  opt2 -r {r} -c {c} (direct): {t:.2f}s", flush=True)
        if cache:
            data = json.load(open(cache)) if os.path.exists(cache) else {}
            data.setdefault("direct", {})[key] = t
            write_json_atomic(cache, data)
        return t
    return data["direct"][key]


def calibrate_opt2(binary, cache, grid=((1, 1), (1, 2), (2, 1), (2, 2)), reps=1):
    """Runs Coeus' own single-machine benchmark and fits T(R,C) = a*R*C + b*R + c*C + d (seconds)."""
    old = json.load(open(cache)) if cache and os.path.exists(cache) else {}
    if "coef" in old:
        print(f"opt2 calibration from {cache}: {old['coef']}")
        return old
    pts = []
    for r, c in grid:
        for _ in range(reps):
            t = run_opt2(binary, r, c)
            pts.append((r, c, t))
            print(f"  opt2 -r {r} -c {c}: {t:.2f}s", flush=True)
    A = np.array([[r * c, r, c, 1.0] for r, c, _ in pts])
    y = np.array([t for _, _, t in pts])
    coef = np.linalg.lstsq(A, y, rcond=None)[0].tolist()
    cal = {**old, "binary": binary, "points": pts, "coef": coef, "model": "a*R*C + b*R + c*C + d"}
    if cache:
        write_json_atomic(cache, {**(json.load(open(cache)) if os.path.exists(cache) else {}), **cal})
    return cal


def coeus_cost(n_docs, features, k):
    R = math.ceil(n_docs / S1_N)        # response ciphertexts (8192 scores each)
    C = math.ceil(features / S1_N)      # query ciphertexts (8192 keywords each)
    buckets = math.ceil(S2_BUCKET_FACTOR * k)
    s1_up, s1_down = C * S1_QUERY_CT_BYTES, R * S1_RESP_CT_BYTES
    s2_up = buckets * S2_D * S2_CT_BYTES
    s2_down = buckets * (S2_EXPANSION ** (S2_D - 1)) * S2_CT_BYTES
    return {
        "R": R, "C": C, "buckets": buckets,
        "s1_up": s1_up, "s1_down": s1_down, "s2_up": s2_up, "s2_down": s2_down,
        "per_query": s1_up + s1_down + s2_up + s2_down,
        # The artifact re-sends the Galois keys with EVERY query in both steps; we count them once (setup).
        "setup_upload": S1_GALOIS_BYTES + S2_GALOIS_BYTES,
        # Keys can be regenerated from the secret key, so the minimum the client must keep is the secret key
        # (step 2's SealPIR key is generated fresh per query in the artifact).
        "client_state": S1_SK_BYTES,
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
    ap.add_argument("--tmp-dir", default="", help="where the temporary token file goes (default: --results-root); "
                    "about 4 B per token, ~25 GB for msmarco-v2")
    ap.add_argument("--results-root", default="../results")
    ap.add_argument("--opt2-bin", default="", help="Coeus_artifact/matmult/bin/opt2, to measure server time")
    ap.add_argument("--opt2-cache", default="", help="JSON cache for the opt2 calibration (default next to the binary)")
    ap.add_argument("--opt2-direct-max-gb", type=float, default=48,
                    help="run opt2 at the real (R, C) when the encoded matrix is at most this big, instead of fitting")
    ap.add_argument("--parallelism", type=int, default=1,
                    help="divide step-1 server time by this many workers (Coeus scales ~linearly, paper Fig. 9)")
    args = ap.parse_args()

    paths = common.dataset_paths(args.dataset, args.name)
    queries = common.load_queries(paths["queries"])
    qrels = common.load_qrels(paths["qrels"])
    if args.max_queries:
        queries = queries[:args.max_queries]
    tokenize, tok_name = make_tokenizer(args.tokenizer)
    print(f"{len(queries)} queries, tokenizer={tok_name}, streaming {paths['corpus']}", flush=True)

    t0 = time.time()
    term_id, ptr, rows, w, n_docs = build_tfidf(paths["corpus"], args.features, tokenize, args.tf,
                                                args.tmp_dir or args.results_root)
    pre_time = time.time() - t0
    print(f"tf-idf built over {n_docs} docs and {len(term_id)} keywords in {pre_time:.1f}s", flush=True)
    # A dictionary padded past the vocabulary only adds empty query ciphertexts: same results, more bytes.
    # Coeus would never be configured like that, so skip it instead of plotting a needlessly worse point.
    if len(term_id) <= args.features - S1_N:
        print(f"F={args.features} is over the vocabulary ({len(term_id)} terms, i.e. "
              f"F={math.ceil(len(term_id) / S1_N) * S1_N}); this config would duplicate that one with more upload. Skipping.")
        return

    results = {}
    for qid, text in queries:
        results[qid] = coeus_topk(text, args.k, n_docs, tokenize, term_id, ptr, rows, w)   # corpus rows

    # Same local cosine rerank every other method gets in main.go
    doc_vecs = np.load(paths["corpus_vec"], mmap_mode="r")
    qvecs = np.load(paths["query_vec"], mmap_mode="r")
    qvec_by_id = {qid: np.asarray(qvecs[i], dtype=np.float32) for i, (qid, _) in enumerate(common.load_queries(paths["queries"]))
                  if i < len(qvecs)}
    reranked = common.cosine_rerank(results, args.k, common.AllRows(), doc_vecs, qvec_by_id)
    idmap = common.ids_for_rows(paths["corpus"], {r for rs in results.values() for r in rs})
    results, reranked = common.rows_to_ids(results, idmap), common.rows_to_ids(reranked, idmap)

    n_q = len(results)
    cost = coeus_cost(n_docs, len(term_id), args.k)   # F = keywords actually in the dictionary
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
        "CoeusGaloisKeysResentPerQueryInArtifactMB": cost["setup_upload"] / 2**20,
        "CoeusCostNote": "sizes as sent by the artifact; Galois keys counted once not per query; step-2 server time not included",
        "CoeusTokenizer": tok_name, "CoeusTF": args.tf,
        "CoeusTfidfBuildTime": f"{pre_time:.3f}",
        "CoeusParallelism": args.parallelism,
        "CoeusTimeSource": "none",
    }
    if args.opt2_bin:
        cache = args.opt2_cache or os.path.join(os.path.dirname(os.path.abspath(args.opt2_bin)), "opt2_calibration.json")
        if cost["server_db"] / 2**30 <= args.opt2_direct_max_gb:
            t_s1 = opt2_direct(args.opt2_bin, cache, cost["R"], cost["C"]) / args.parallelism
            meta["CoeusTimeSource"] = f"opt2 run directly at R={cost['R']} C={cost['C']}"
        else:
            cal = calibrate_opt2(args.opt2_bin, cache)
            a, b, c, d = cal["coef"]
            t_s1 = max(0.0, a * cost["R"] * cost["C"] + b * cost["R"] + c * cost["C"] + d) / args.parallelism
            meta["CoeusTimeSource"] = f"opt2 fit {cal['model']} coef={cal['coef']}"
        meta["TotalAnswerTime"] = f"{t_s1 * n_q:.6f}"
        meta["CoeusStep1SecondsPerQuery"] = f"{t_s1:.6f}"
        meta["CoeusStep2TimeIncluded"] = "false"
    else:
        print("no --opt2-bin: leaving TotalAnswerTime out (plots show NaN for Coeus time)")

    out = os.path.join(args.results_root, f"coeus_{args.name}_k{args.k}_f{args.features}_tf{args.tf}")
    common.write_run(out, meta, results, reranked)
    print(json.dumps(meta, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
