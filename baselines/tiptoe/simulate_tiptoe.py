#!/usr/bin/env python3
"""
Simulated Tiptoe run: computes the same metadata.json as a real `-t tiptoe`
run, but without SEAL. It works from the corpus that preprocess_tiptoe.py
builds and from Tiptoe's own code paths. Every number is either exact or a
LOWER bound on Tiptoe's cost, so the simulation can never make Tiptoe look
worse than it is.

QUALITY (exact). Tiptoe's crypto decrypts to the true values (SimplePIR
inner products, PIR), so the client always ends up with the same docs. The
script replays search/protocol/client.go runRound: pick the nearest centroid,
score every doc in that cluster with the 5-bit query, then return the docs of
the URL bundle that holds the best doc, sorted by score. It writes
results.json and, after the same cosine rerank main.go applies,
results_reRank.json.

DATABASE SHAPES (exact). The packing is search/packing PackChunks (first-fit
decreasing) and the database code is search/database/build_database.go:
  ranking DB: clusters packed into columns of height <= hint_mb*125 rows
              -> L_emb = tallest column, M_emb = #columns * 192
              (logq=64, p=2^17, so it is invalid if M_emb > 2^21)
  URL DB:     zlib-compressed bundles packed into columns of height
              <= max(longest bundle, hint_mb*250) bytes
              -> L_url = tallest column, M_url = #columns
              (logq=32, p=2^8, so it is invalid if M_url > 2^19)

COMMUNICATION PER QUERY (lower bounds: raw payload only, no serialisation headers)
  round 1 (ranking): up = ceil3(M_emb) * 8 B,  down = L_emb * 8 B
  round 2 (URL):     up = ceil3(M_url) * 4 B,  down = L_url * 4 B
      (simplepir pir/client.go pads the query to a multiple of the squish ratio 3;
       Server.Answer returns L elements)
  offline token (underhood): up   = 2048 seeded RLWE ciphertexts
                                     (n_lwe=2048 secret entries, one ciphertext each;
                                      N=2048, 1 prime, c0 = 2048*8 B plus a 64 B seed)
                              down = (8*ceil(L_emb/2048) + 5*ceil(L_url/2048)) RLWE
                                     ciphertexts of 2*2048*8 B
                                     (8 limbs for the 64-bit hint, 5 for the 32-bit one)

ROUNDS: 2 online (ranking, URL) and 1 offline (token), as in the real run.
LAN/WAN time uses the online bytes only, the same as tiptoe/tiptoe.go. The
token bytes are still counted in TotalByteSent.

CLIENT STORAGE (lower bound): centroids (C x 192 float32), plus 8 B per
cluster and 16 B per bundle for the index maps. With underhood the client
keeps no SimplePIR hint.

REDUCED PRECISION (--precision fp16|fp8|fp4|fp2|fp1, default fp32 = the DB as built).
Every embedding is cast to a lower-precision float after it is loaded, and
everything after that uses the cast values: the doc vectors in the ranking DB
(re-read from the float corpus .npy for each cluster, cast, then Tiptoe's
5-bit quantisation with the DB's own scale), the query, the client's
centroids, and the cosine rerank. The clustering and bundling stay those of
the preprocessed DB (server side, done once), so all precisions share one DB.
  fp16  IEEE half, plain cast
  fp8   OCP E4M3 (max 448), per-vector absmax scale
  fp4   OCP E2M1 (0, .5, 1, 1.5, 2, 3, 4, 6), per-vector absmax scale
  fp2   E1M0 = ternary {-1, 0, +1}, per-vector absmean scale (as in BitNet b1.58;
        absmax would round almost every coordinate to 0)
  fp1   sign only = binary {-1, +1}, per-vector absmean scale (1-bit, as in BitNet)
Rounding is round-to-nearest-even, saturating. Bytes per query do not change:
the ranking DB entries are mod-p plaintexts in 64-bit LWE (and the URL DB has no
embeddings), so the up/down sizes depend on L and M only. Client storage counts
the centroids at the reduced width (+ a 4 B scale per centroid for fp8/4/2/1).
Output folders get a _<precision> suffix: tiptoe_..._k10_fp4.

SERVER TIME (only with --bench-bin): Tiptoe's online answers are exactly
SimplePIR's Server.Answer = MulVecPacked(squished DB, query), so
cmd/simplepir_bench times that at the real (L, M) for both services. The
result is divided by --parallelism (Tiptoe shards across servers). The token's
server-side work (RLWE inner products) is NOT included, which again favours
Tiptoe. Without --bench-bin, TotalAnswerTime is left out rather than guessed.

Usage:
  python baselines/tiptoe/simulate_tiptoe.py --dataset ../datasets --name scifact \
      --tiptoe-dir ../datasets/tiptoe/scifact_dpc500 --k 10 --results-root ../results \
      --bench-bin ./simplepir_bench --parallelism 16
"""

import argparse
import json
import math
import os
import subprocess
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402

SUBCLUSTER_DELIM = "-------------------------"
SQUISH = 3                         # matrix.h COMPRESSION_32 / COMPRESSION_64
EMB_SLOTS = 192
EMB_MAX_M = 1 << 21                # lwe/params.go plaintextModulus64: largest m with p >= 2^17
URL_MAX_M = 1 << 19                # plaintextModulus32: largest m with p >= 256
RLWE_N = 2048                      # underhood rlwe.cpp poly_modulus_degree, one 38-bit prime
LWE_N64 = 2048                     # lwe secretDimension64
RLWE_CT = 2 * RLWE_N * 8           # full ciphertext payload
RLWE_SEEDED_CT = RLWE_N * 8 + 64   # seeded (symmetric, compressed) ciphertext payload
LIMBS64, LIMBS32 = 8, 5            # underhood NumLimbs64 / NumLimbs32

# precision -> (mantissa bits, min exponent, max finite, per-vector scale rule)
MINIFLOATS = {
    "fp8": (3, -6, 448.0, "absmax"),   # OCP E4M3, bias 7
    "fp4": (1, 0, 6.0, "absmax"),      # OCP E2M1, bias 1
    "fp2": (0, 0, 1.0, "absmean"),     # E1M0, bias 1: {0, 1}, i.e. ternary with the sign
}
PRECISION_BITS = {"fp32": 32, "fp16": 16, "fp8": 8, "fp4": 4, "fp2": 2, "fp1": 1}


def minifloat_round(x, mant, emin, maxv):
    """Round to the nearest value of a sign + exponent + mantissa float (subnormals below 2^emin), saturating."""
    a = np.abs(x)
    e = np.maximum(np.floor(np.log2(np.maximum(a, 1e-30))), emin)
    step = np.exp2(e - mant)
    return np.sign(x) * np.minimum(np.round(a / step) * step, maxv)   # np.round = half to even


def reduce_precision(x, prec):
    """x (1-D vector or rows of vectors) cast to `prec` and back to float32 (see REDUCED PRECISION above)."""
    x = np.asarray(x, dtype=np.float32)
    if prec == "fp32":
        return x
    if prec == "fp16":
        return x.astype(np.float16).astype(np.float32)
    if prec == "fp1":   # sign bit only, scaled by the vector's mean |x|
        s = np.abs(x).mean(axis=-1, keepdims=True)
        return (np.where(x >= 0, 1.0, -1.0) * s).astype(np.float32)
    mant, emin, maxv, rule = MINIFLOATS[prec]
    a = np.abs(x)
    s = (a.max(axis=-1, keepdims=True) / maxv) if rule == "absmax" else a.mean(axis=-1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return (minifloat_round(x / s, mant, emin, maxv) * s).astype(np.float32)


class ReducedRows:
    """doc_vecs for cosine_rerank: rows of a (memmapped) array, cast to `prec` on read."""
    def __init__(self, base, prec):
        self.base, self.prec = base, prec

    def __getitem__(self, rows):
        return reduce_precision(self.base[rows], self.prec)


def read_cluster_full(path):
    """[(row, emb, bundle)] in file order (= Tiptoe's within-cluster doc order)."""
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


def cluster_shape(path):
    """(#docs, [compressed bytes of each URL bundle]), as ReadUrlsTxt builds them (URL_DELIM=' ', zlib level 9)."""
    bundles, cur, n = [], [], 0
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.strip() == SUBCLUSTER_DELIM:
                bundles.append(cur)
                cur = []
                continue
            cur.append(line.rsplit(" | ", 1)[1].strip())
            n += 1
    if cur:
        bundles.append(cur)
    return n, [len(zlib.compress(" ".join(b).encode(), 9)) for b in bundles]


def pack_chunks(sizes, cap):
    """search/packing PackChunks: first-fit decreasing, strict '<', capacity raised to the longest chunk."""
    sizes = sorted(sizes, reverse=True)
    cap = max(cap, sizes[0])
    cols = [sizes[0]]
    for s in sizes[1:]:
        for j in range(len(cols)):
            if cols[j] + s < cap:
                cols[j] += s
                break
        else:
            cols.append(s)
    return cols


def ceil_to(x, m):
    return x + (-x) % m


def bench(binary, rows, cols, logq, p, max_gb):
    out = subprocess.run([binary, "-rows", str(rows), "-cols", str(cols), "-logq", str(logq), "-p", str(p),
                          "-max-gb", str(max_gb)], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])["seconds"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../datasets")
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--tiptoe-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--emb-hint-mb", type=int, default=500, help="Tiptoe DEFAULT_EMBEDDINGS_HINT_SZ (text)")
    ap.add_argument("--url-hint-mb", type=int, default=100, help="Tiptoe DEFAULT_URL_HINT_SZ")
    ap.add_argument("--results-root", default="", help="write <root>/tiptoe_<name>_c<C>_m<MA>_b<B>_h<H>_k<K>/metadata.json")
    ap.add_argument("--bench-bin", default="", help="cmd/simplepir_bench binary, to measure online server time")
    ap.add_argument("--bench-max-gb", type=float, default=8)
    ap.add_argument("--parallelism", type=int, default=1, help="divide server time by this (Tiptoe shards freely)")
    ap.add_argument("--precision", choices=list(PRECISION_BITS), default="fp32",
                    help="cast every embedding to this float format after loading (REDUCED PRECISION above)")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.tiptoe_dir, "tiptoe_meta.json")))
    C = meta["num_clusters"]
    prec = args.precision
    cents = reduce_precision(np.load(os.path.join(args.tiptoe_dir, "centroids.npy")), prec)
    cdir = os.path.join(args.tiptoe_dir, "clusters")

    # ---------- database shapes ----------
    npy = meta.get("format", "txt") == "npy"
    if npy:   # compact arrays from preprocess_tiptoe.py --format npy (same content as the txt files)
        cptr = np.load(os.path.join(args.tiptoe_dir, "cluster_ptr.npy"))
        cluster_docs = np.diff(cptr).tolist()
        bundle_bytes = np.load(os.path.join(args.tiptoe_dir, "bundle_bytes.npy")).tolist()
        n_bundles = len(bundle_bytes)
        emb_all, rows_all, bun_all = (np.load(os.path.join(args.tiptoe_dir, f), mmap_mode="r")
                                      for f in ("emb.npy", "rows.npy", "bundle.npy"))
    else:
        cluster_docs, bundle_bytes, n_bundles = [], [], 0
        for c in range(C):
            n, bb = cluster_shape(os.path.join(cdir, f"cluster_{c}.txt"))
            cluster_docs.append(n)
            bundle_bytes += bb
            n_bundles += len(bb)
    emb_cols = pack_chunks(cluster_docs, args.emb_hint_mb * 125)
    L_emb, M_emb = max(emb_cols), len(emb_cols) * EMB_SLOTS
    url_cap = max(max(bundle_bytes), args.url_hint_mb * 250)
    url_cols = pack_chunks(bundle_bytes, url_cap)
    L_url, M_url = max(url_cols), len(url_cols)
    valid = M_emb <= EMB_MAX_M and M_url <= URL_MAX_M
    if not valid:
        print(f"WARNING: Tiptoe would refuse this config (M_emb={M_emb} > {EMB_MAX_M} or M_url={M_url} > {URL_MAX_M}); "
              f"use bigger clusters", file=sys.stderr)

    # ---------- bytes per query ----------
    emb_up, emb_down = ceil_to(M_emb, SQUISH) * 8, L_emb * 8
    url_up, url_down = ceil_to(M_url, SQUISH) * 4, L_url * 4
    tok_up = LWE_N64 * RLWE_SEEDED_CT
    tok_down = (LIMBS64 * math.ceil(L_emb / RLWE_N) + LIMBS32 * math.ceil(L_url / RLWE_N)) * RLWE_CT
    online = emb_up + emb_down + url_up + url_down
    cent_bytes = cents.size * PRECISION_BITS[prec] / 8 + (4 * len(cents) if prec in MINIFLOATS or prec == "fp1" else 0)
    client_bytes = cent_bytes + 8 * C + 16 * n_bundles
    server_bytes = L_emb * math.ceil(M_emb / SQUISH) * 8 + L_url * math.ceil(M_url / SQUISH) * 4

    # ---------- quality ----------
    paths = common.dataset_paths(args.dataset, args.name)
    all_queries = common.load_queries(paths["queries"])
    queries = all_queries[:args.max_queries] if args.max_queries else all_queries
    qvecs = np.load(paths["query_vec"], mmap_mode="r")
    qrels = common.load_qrels(paths["qrels"])
    lo, hi = -(1 << (meta["slot_bits"] - 1)), (1 << (meta["slot_bits"] - 1)) - 1
    doc_vecs = np.load(paths["corpus_vec"], mmap_mode="r")

    def requantise(rows):
        """The DB's 5-bit doc embeddings, rebuilt from the float corpus at `prec` (same rule as preprocess_tiptoe.py)."""
        srt = np.argsort(rows, kind="stable")       # sorted = sequential reads from the memmap
        out = np.empty((len(rows), doc_vecs.shape[1]), dtype=np.int8)
        for s_ in range(0, len(rows), 65536):
            part = srt[s_:s_ + 65536]
            v = reduce_precision(doc_vecs[rows[part]], prec)
            out[part] = np.clip(np.round(v * meta["scale"]), lo, hi).astype(np.int8)
        return out

    cache, results, returned = {}, {}, []
    for i, (qid, _) in enumerate(queries):
        v = reduce_precision(qvecs[i], prec)
        c = int(np.argmin(((cents - v) ** 2).sum(1)))                                         # nearestCluster
        q = np.clip(np.round(v.astype(np.float64) * meta["scale"]), lo, hi).astype(np.int64)  # quantise
        if c not in cache:
            if len(cache) > 256:
                cache.clear()
            if npy:
                s_, e_ = cptr[c], cptr[c + 1]
                cache[c] = (np.asarray(emb_all[s_:e_], dtype=np.int8), np.asarray(rows_all[s_:e_]),
                            np.asarray(bun_all[s_:e_]))
            else:
                cl = read_cluster_full(os.path.join(cdir, f"cluster_{c}.txt"))
                cache[c] = (np.stack([e for _, e, _ in cl]).astype(np.int8), np.array([r for r, _, _ in cl]),
                            np.array([b for _, _, b in cl]))
            if prec != "fp32":
                cache[c] = (requantise(cache[c][1]),) + cache[c][1:]
        embs, crow, cbun = cache[c]
        # exact integer inner products, as decrypted (|score| <= 192*16*16, so int32 is exact); clusters are cached
        # as int8 to keep memory at ~200 B per doc even with 40k-doc clusters
        scores = embs.astype(np.int32) @ q.astype(np.int32)
        order = np.argsort(-scores, kind="stable")
        best_bundle = cbun[order[0]]
        ids, seen = [], set()
        for j in order:
            row, b = int(crow[j]), cbun[j]
            if b != best_bundle or row in seen:   # results hold corpus rows; mapped to doc IDs below
                continue
            seen.add(row)
            ids.append(row)
            if len(ids) == args.k:
                break
        results[qid] = ids
        returned.append(len(ids))

    qvec_by_id = {qid: reduce_precision(qvecs[i], prec) for i, (qid, _) in enumerate(all_queries) if i < len(qvecs)}
    reranked = common.cosine_rerank(results, args.k, common.AllRows(), ReducedRows(doc_vecs, prec), qvec_by_id)
    # Only the returned rows' IDs are looked up, so the corpus is never held in memory (msmarco-v2 is 138M docs)
    idmap = common.ids_for_rows(paths["corpus"], {r for rows in results.values() for r in rows})
    results, reranked = common.rows_to_ids(results, idmap), common.rows_to_ids(reranked, idmap)

    nq = len(results)
    out = {
        "Method": "tiptoe", "TiptoeSimulated": "true", "TiptoeValidConfig": str(valid).lower(),
        "NumQueries": nq,
        "MRRPreReRank": f"{common.calc_mrr(results, qrels):.4f}",
        "RecallPreReRank": f"{common.calc_recall(results, qrels):.4f}",
        "MRR": f"{common.calc_mrr(reranked, qrels):.4f}",
        "Recall": f"{common.calc_recall(reranked, qrels):.4f}",
        "ClientStorageMB": client_bytes / 2**20,
        "DBSizeInBytesMB": server_bytes / 2**20,
        "CommCostPerBatchKB": online / 1024,
        "TotalByteSent": (online + tok_up + tok_down) * nq,
        "TotalLANTime": nq * common.net_time(online, 2, 1000, 0.005),
        "TotalWANTime": nq * common.net_time(online, 2, 400, 0.05),
        "TiptoeOfflineLANTime": nq * common.net_time(tok_up + tok_down, 1, 1000, 0.005),
        "TiptoeOfflineWANTime": nq * common.net_time(tok_up + tok_down, 1, 400, 0.05),
        "TiptoeTokenKBPerQuery": (tok_up + tok_down) / 1024,
        "TiptoeOfflineUpBytes": tok_up * nq, "TiptoeOfflineDownBytes": tok_down * nq,
        "TiptoeOnlineUpBytes": (emb_up + url_up) * nq, "TiptoeOnlineDownBytes": (emb_down + url_down) * nq,
        "TiptoeRounds": 2 * nq,
        "TiptoeNumClusters": C, "TiptoeMaxClusterSize": max(cluster_docs),
        "TiptoeMultiAssign": meta.get("multi_assign"), "TiptoeBundleSize": meta.get("bundle_size"),
        "TiptoeEmbHintMB": args.emb_hint_mb, "TiptoeURLHintMB": args.url_hint_mb,
        "TiptoeNumBundles": n_bundles,
        "TiptoeEmbL": L_emb, "TiptoeEmbM": M_emb, "TiptoeURLL": L_url, "TiptoeURLM": M_url,
        "TiptoeMeanDocsReturned": f"{np.mean(returned):.2f}",
        "TiptoeDir": args.tiptoe_dir,
        "TiptoePrecision": prec,
        "TiptoeCostNote": "lower bounds: payload only, no serialisation headers; token server time not included",
    }
    if args.bench_bin:
        t_emb = bench(args.bench_bin, L_emb, M_emb, 64, 1 << 17, args.bench_max_gb)
        t_url = bench(args.bench_bin, L_url, M_url, 32, 256, args.bench_max_gb)
        t = (t_emb + t_url) / args.parallelism
        out.update({"TotalAnswerTime": f"{t * nq:.6f}", "TiptoeRound1Time": f"{t_emb / args.parallelism * nq:.6f}",
                    "TiptoeRound2Time": f"{t_url / args.parallelism * nq:.6f}", "TiptoeParallelism": args.parallelism,
                    "TiptoeTokenServerTimeIncluded": "false"})

    print(json.dumps(out, indent=2))
    if args.results_root:
        tag = f"c{C}_m{meta.get('multi_assign', 2)}_b{meta.get('bundle_size', 160)}_h{args.emb_hint_mb}"
        d = os.path.join(args.results_root, f"tiptoe_{args.name}_{tag}_k{args.k}" + ("" if prec == "fp32" else f"_{prec}"))
        common.write_run(d, out, results, reranked)
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
