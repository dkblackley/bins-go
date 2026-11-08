#!/usr/bin/env python3
"""
Builds the on-disk corpus that the real Tiptoe server code reads
(github.com/ahenzinger/tiptoe, search/corpus/read_corpus.go), from the SAME
192-dim embeddings PILLAR and PACMANN use, so the three are compared on one
embedding model.

It mirrors Tiptoe's own offline pipeline (cluster/kmeans/*.py, dim-reduce/*.py):

  1. k-means over the corpus embeddings            (compute-centroids-msmarco.py)
  2. assign each doc to its nearest cluster, and to a 2nd cluster when it sits
     on a boundary (gap to the 2nd centroid under the 20th percentile)
                                                     (assign-all-msmarco.py, MULTI_ASSIGN=2)
  3. split each cluster into URL "bundles" of <= 160 docs with a small k-means,
     so one PIR fetch returns the metadata of similar docs (url-cluster-msmarco.py)
  4. quantise every embedding to 5-bit signed ints   (pca.py: run_pca / clip(-16, 15))

Two places differ from Tiptoe, both because our inputs differ:
  * Our embeddings are already 192-dim, so the PCA step is skipped (Tiptoe used
    PCA 768 -> 192). The quantisation scale is fitted to the data instead of
    Tiptoe's constant (x*32 /10), which was tuned for their PCA output.
  * The "URL" stored per doc is its row index in corpus.jsonl; the Go side maps
    it back to the BEIR doc ID. Row indices have no spaces, which Tiptoe uses as
    its URL delimiter.

Output (in --out):
  clusters/cluster_<i>.txt   "<row> | e0,e1,...,e191 | <row>" lines, bundles
                             separated by Tiptoe's SUBCLUSTER_DELIM
  centroids.npy              float32 [C, dim] (client-side state in Tiptoe)
  tiptoe_meta.json           scale, slot bits, cluster stats

Usage:
  python baselines/tiptoe/preprocess_tiptoe.py --dataset ../datasets --name scifact
  python baselines/tiptoe/preprocess_tiptoe.py --dataset ../datasets --name msmarco --docs-per-cluster 2500
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:            # no tqdm: no bars, same output
    def tqdm(it=None, **kw):
        return it


def bar(it, desc, total=None):
    """Progress bar on stderr (run_all_simulated.sh shows stderr in the terminal; stdout only goes to the log)."""
    return tqdm(it, desc=desc, total=total, file=sys.stderr, mininterval=1, dynamic_ncols=True)

SUBCLUSTER_DELIM = "-------------------------"   # search/corpus/utils.go
TIPTOE_MAX_CLUSTERS = 25196                      # search/config/constants.go TOTAL_NUM_CLUSTERS (text)

# Same file layout as GetDatasets() in main.go
DATASET_EMBEDS = {
    "msmarco": "msmarco/msmarco_corpus_embed_f32.npy",
    "scifact": "scifact/scifact_corpus_embed_f32.npy",
    "trec-covid": "trec-covid/trec_corpus_embed_f32.npy",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kmeans(data, k, seed, niter=20, verbose=False):
    """faiss if available (what Tiptoe used), otherwise sklearn MiniBatchKMeans.
    verbose: progress bar with an ETA, one step per k-means iteration."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    try:
        import faiss
        mppc = max(256, math.ceil(len(data) / k))
        if verbose:
            # Same as faiss.Kmeans(...).train(data), but one iteration per train() call so we can show progress:
            # each call continues from the previous centroids, and since mppc*k >= len(data) faiss never subsamples,
            # so the result is bit-identical (checked on scifact and a 200k-doc slice of nq).
            cl = faiss.Clustering(data.shape[1], k)
            cl.niter, cl.seed, cl.max_points_per_centroid = 1, seed, mppc
            index = faiss.IndexFlatL2(data.shape[1])
            for _ in bar(range(niter), f"k-means ({k} clusters, {len(data)} docs)"):
                cl.train(data, index)
                index.reset()
            return faiss.vector_to_array(cl.centroids).reshape(k, -1).astype(np.float32), "faiss"
        km = faiss.Kmeans(data.shape[1], k, niter=niter, seed=seed, verbose=False, max_points_per_centroid=mppc)
        # the small per-cluster bundle k-means make faiss print "WARNING clustering N points to K centroids"
        # to fd 2 (harmless: it just wants >= 39 points per centroid); send it to the log (fd 1), not the bars
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(1, 2)
        try:
            km.train(data)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        return km.centroids.astype(np.float32), "faiss"
    except ImportError:
        pass
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=max(4096, 4 * k),
                             n_init=1, max_iter=niter)
        km.fit(data)
        return km.cluster_centers_.astype(np.float32), "sklearn"
    except ImportError:
        pass
    # Plain Lloyd's as a last resort (fine for small corpora / testing)
    rng = np.random.default_rng(seed)
    cents = data[rng.choice(len(data), k, replace=False)].copy()
    for _ in range(niter):
        a, _ = nearest(data, cents, 1)
        a = a[:, 0]
        sums = np.zeros_like(cents)
        np.add.at(sums, a, data)
        cnt = np.bincount(a, minlength=k)
        nz = cnt > 0
        cents[nz] = sums[nz] / cnt[nz, None]
    return cents, "numpy"


def nearest(data, centroids, n, chunk=0, desc=None):
    """Indices and squared L2 distances of the n nearest centroids, in chunks to bound memory (~800 MB).
    desc: show a progress bar over the chunks."""
    chunk = chunk or max(1000, int(2e8 // len(centroids)))
    c_norm = (centroids ** 2).sum(1)
    idx = np.empty((len(data), n), dtype=np.int64)
    dist = np.empty((len(data), n), dtype=np.float32)
    starts = range(0, len(data), chunk)
    for s in (bar(starts, desc) if desc else starts):
        x = data[s:s + chunk]
        d = (x ** 2).sum(1, keepdims=True) - 2 * x @ centroids.T + c_norm[None, :]
        part = np.argpartition(d, n - 1, axis=1)[:, :n] if n < d.shape[1] else np.tile(np.arange(d.shape[1]), (len(x), 1))
        pd = np.take_along_axis(d, part, axis=1)
        order = np.argsort(pd, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        dist[s:s + chunk] = np.take_along_axis(pd, order, axis=1)
    return idx, dist


def make_bundles(members, data, bundle_size, seed):
    """Tiptoe's url-cluster step: group a cluster's docs into bundles of similar docs, <= bundle_size each."""
    if len(members) <= bundle_size:
        return [members]
    nb = math.ceil(len(members) / bundle_size)
    cents, _ = kmeans(data[members], nb, seed, niter=10)
    assign, _ = nearest(data[members], cents, 1)
    bundles = []
    for b in range(nb):
        m = members[assign[:, 0] == b]
        # Tiptoe recursively re-splits oversize bundles; splitting in order is the same size guarantee
        for s in range(0, len(m), bundle_size):
            bundles.append(m[s:s + bundle_size])
    return [b for b in bundles if len(b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../datasets", help="datasets root (same as -dataset in main.go)")
    ap.add_argument("--name", default="scifact", help="msmarco | scifact | trec-covid | any generic-layout set, e.g. nq, msmarco-v2 (or use --corpus-embed)")
    ap.add_argument("--corpus-embed", default="", help="explicit path to the corpus .npy (overrides --name)")
    ap.add_argument("--out", default="", help="output dir, default <dataset>/tiptoe/<name>_c<C>")
    ap.add_argument("--num-clusters", type=int, default=0, help="number of clusters (overrides --docs-per-cluster)")
    ap.add_argument("--docs-per-cluster", type=int, default=2500,
                    help="target docs per cluster; Tiptoe used ~2500 on MS MARCO (1280 clusters, 3.2M docs)")
    ap.add_argument("--multi-assign", type=int, default=2, help="clusters a boundary doc may join (Tiptoe: 2)")
    ap.add_argument("--boundary-percentile", type=float, default=20, help="Tiptoe: 20")
    ap.add_argument("--bundle-size", type=int, default=160, help="docs per URL bundle (Tiptoe URLS_PER_BUNDLE=160)")
    ap.add_argument("--slot-bits", type=int, default=5, help="embedding precision (Tiptoe SLOT_BITS=5)")
    ap.add_argument("--scale-percentile", type=float, default=99.9,
                    help="|value| at this percentile is mapped to the largest 5-bit value")
    ap.add_argument("--train-sample", type=int, default=2_000_000, help="max docs used to train k-means")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmeans-cache", default="",
                    help="dir to cache the main k-means centroids in. They only depend on the embeddings, C, seed and "
                         "train sample, not on --multi-assign/--bundle-size, so those sweeps reuse one k-means")
    args = ap.parse_args()

    path = args.corpus_embed or os.path.join(
        args.dataset, DATASET_EMBEDS.get(args.name, f"{args.name}/{args.name}_corpus_embed_f32.npy"))
    log(f"loading {path}")
    data = np.load(path, mmap_mode="r")
    N, dim = data.shape
    data = np.asarray(data, dtype=np.float32)

    C = args.num_clusters or max(1, round(N / args.docs_per_cluster))
    if C > TIPTOE_MAX_CLUSTERS:
        log(f"WARNING: {C} clusters > Tiptoe's TOTAL_NUM_CLUSTERS={TIPTOE_MAX_CLUSTERS}; capping")
        C = TIPTOE_MAX_CLUSTERS
    out = args.out or os.path.join(args.dataset, "tiptoe", f"{args.name}_c{C}")
    os.makedirs(os.path.join(out, "clusters"), exist_ok=True)
    log(f"N={N} dim={dim} clusters={C} multi_assign={args.multi_assign} -> {out}")

    rng = np.random.default_rng(args.seed)
    train = data if N <= args.train_sample else data[rng.choice(N, args.train_sample, replace=False)]
    log(f"k-means: {len(train)} points -> {C} clusters, 20 iterations")
    cache = None
    if args.kmeans_cache:
        os.makedirs(args.kmeans_cache, exist_ok=True)
        tag = os.path.splitext(os.path.basename(path))[0]
        cache = os.path.join(args.kmeans_cache, f"{tag}_N{N}_C{C}_seed{args.seed}_ts{args.train_sample}_it20.npz")
    lock = None
    if cache:
        # Concurrent runs with the same C (slurm array tasks for other ma/bundle values) wait here for the first one's
        # k-means instead of all computing it. The lock is released when the file closes or the process dies.
        import fcntl
        lock = open(cache + ".lock", "w")
        log(f"waiting for k-means lock {cache}.lock")
        fcntl.flock(lock, fcntl.LOCK_EX)
    if cache and os.path.exists(cache):
        z = np.load(cache)
        centroids, backend = z["centroids"], str(z["backend"])
        log(f"k-means centroids loaded from cache {cache}")
    else:
        centroids, backend = kmeans(train, C, args.seed, verbose=True)
        if cache:
            tmp = cache[:-4] + f".tmp{os.getpid()}.npz"
            np.savez(tmp, centroids=centroids, backend=backend)
            os.replace(tmp, cache)
            log(f"k-means centroids cached in {cache}")
    if lock:
        lock.close()
    log(f"k-means done ({backend})")

    ma = min(args.multi_assign, C)
    assign, dist = nearest(data, centroids, ma, desc="assign docs to clusters")
    doc_parts, clu_parts = [np.arange(N)], [assign[:, 0]]
    over = 0
    for k in range(1, ma):
        gap = dist[:, k] - dist[:, 0]
        extra = np.nonzero(gap < np.percentile(gap, args.boundary_percentile))[0]
        doc_parts.append(extra)
        clu_parts.append(assign[extra, k])
        over += len(extra)
    docs_all, clus_all = np.concatenate(doc_parts), np.concatenate(clu_parts)
    order = np.argsort(clus_all, kind="stable")
    counts = np.bincount(clus_all, minlength=C)
    members = np.split(docs_all[order], np.cumsum(counts)[:-1])
    log(f"assigned, {over} extra boundary copies")

    # Tiptoe requires consecutive, non-empty clusters; drop empties and renumber.
    keep = [c for c in range(C) if len(members[c])]
    centroids = centroids[keep]
    members = [members[c] for c in keep]
    C = len(keep)

    # 5-bit quantisation (Tiptoe clips to [-16, 15])
    lo, hi = -(1 << (args.slot_bits - 1)), (1 << (args.slot_bits - 1)) - 1
    scale = hi / float(np.percentile(np.abs(train), args.scale_percentile))
    q = np.clip(np.round(data * scale), lo, hi).astype(np.int8)

    sizes, n_bundles = [], 0
    for c in bar(range(C), "bundle + write clusters"):
        bundles = make_bundles(members[c], data, args.bundle_size, args.seed + c)
        n_bundles += len(bundles)
        sizes.append(len(members[c]))
        lines = []
        for b, bundle in enumerate(bundles):
            if b > 0:
                lines.append(SUBCLUSTER_DELIM)
            for i in bundle:
                lines.append(f"{i} | {','.join(map(str, q[i].tolist()))} | {i}")
        with open(os.path.join(out, "clusters", f"cluster_{c}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        if c % 500 == 0:
            log(f"wrote cluster {c}/{C}")

    np.save(os.path.join(out, "centroids.npy"), centroids.astype(np.float32))
    meta = {
        "source": path, "num_docs": int(N), "dim": int(dim), "num_clusters": int(C),
        "slot_bits": args.slot_bits, "scale": scale, "multi_assign": ma,
        "boundary_copies": int(over), "bundle_size": args.bundle_size, "num_bundles": int(n_bundles),
        "max_cluster_size": int(max(sizes)), "mean_cluster_size": float(np.mean(sizes)),
        "kmeans_backend": backend, "seed": args.seed,
    }
    with open(os.path.join(out, "tiptoe_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"done: {json.dumps(meta)}")
    print(out)


if __name__ == "__main__":
    sys.exit(main())
