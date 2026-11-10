#!/usr/bin/env python3
"""
Fast, dataset-generic builder for the tree method's precomputed artifacts.

Produces exactly the same files as build_small_test_artifacts.py (which ports the collaborator's prepare_data.py,
prepare_data_stage2.py, gen_layout.py, create_docmap.py and opt3.py query analysis), but:
  * postings are fetched ONCE per term and every (B, r, s) in the grid is built from them in the same pass,
  * terms are split over --workers processes (each with its own Lucene reader), and
  * the per-node work is vectorised and sparse (no num_leaves-sized array per term), so it scales to 100M+ docs.

Usage (index must already exist, built single-threaded so Lucene docids follow corpus order; datasets/prep_tree.sh
does that):
  python tree/build_tree_artifacts.py --dataset-dir ../datasets/nq --out data/nq_tree --name nq \
      --B 16 32 64 --r 8 32 128 --s 4 8 16 --workers 32

Output (what run_tree_pir_old.slurm reads):
  <out>/stage1_vocab.json                       term -> id (shared by every combo)
  <out>/stage1/B<B>_r<r>/{data,idmap}.bin       stage-1 DB (shared by every s)
  <out>/B<B>_r<r>_s<s>/layout.json, stage2/{data,idmap}.bin,
                       stage1_data.bin, stage1_idmap.bin, stage1_vocab.json   (the last three are symlinks)
  <out>/index_<name>/docmap.bin, <out>/stage3_doc.npy(.ids), <out>/queries.qrels.{tsv,analyzed.json}, <out>/qrels.trec

Scores use min_dl = avgdl for every node (int(avgdl) for stage 1, round(avgdl) for stage 2), like the existing
scripts do when no min_dl_map is given.
"""
import argparse
import json
import math
import os
import shutil
import struct
import sys
import time

import numpy as np

IDMAP_DT = np.dtype([("term", "<u4"), ("node", "<u4"), ("row", "<u8")])


# ----------------------------------------------------------------------------------------------- reader helpers

def open_reader(index):
    from pyserini.index.lucene import LuceneIndexReader
    return LuceneIndexReader(index)


def index_stats(rd):
    st = rd.stats()
    N = int(st["documents"])
    return N, st["total_terms"] / N


class PostingsStream:
    """A term's postings as two numpy arrays (Lucene docids ascending, term frequencies), read straight from Lucene's
    PostingsEnum (docs + freqs, no positions). Same values, in the same order, as pyserini's
    get_postings_list(term, analyzer=None) (field "contents", the raw term), which builds the whole list in Java with
    positions and then one Python object per posting: ~10 GB per worker for a very common term of a 100M+ doc corpus,
    which is why msmarco-v2 needed 1 TB. Here a posting costs 8 bytes while it is read (two int32 arrays).
    Same number of Java calls per posting (nextDoc + freq instead of getDocid + getTF + getPositions)."""

    def __init__(self, rd):
        from pyserini.pyclass import autoclass
        self.BytesRef = autoclass("org.apache.lucene.util.BytesRef")
        PostingsEnum = autoclass("org.apache.lucene.index.PostingsEnum")
        self.FREQS = PostingsEnum.FREQS
        self.NO_MORE = autoclass("org.apache.lucene.search.DocIdSetIterator").NO_MORE_DOCS
        terms = autoclass("org.apache.lucene.index.MultiTerms").getTerms(rd.reader, "contents")
        self.te = terms.iterator()

    def get(self, term):
        """-> (docs int64, tfs int64) or None if the term is not in the index."""
        # pyjnius won't pass a Python str as Java CharSequence, so hand BytesRef the UTF-8 bytes (its byte[]
        # constructor): the same bytes BytesRef(CharSequence) would make, which is how Lucene stores the term.
        if not self.te.seekExact(self.BytesRef(term.encode("utf-8"))):
            return None
        from array import array
        pe = self.te.postings(None, self.FREQS)
        docs, tfs = array("i"), array("i")
        nxt, freq, end = pe.nextDoc, pe.freq, self.NO_MORE
        d = nxt()
        while d != end:
            docs.append(d)
            tfs.append(freq())
            d = nxt()
        if not docs:
            return None
        return np.frombuffer(docs, dtype=np.int32).astype(np.int64), np.frombuffer(tfs, dtype=np.int32).astype(np.int64)


def postings_via_pyserini(rd, term):
    """The original reader (pyserini get_postings_list): used to check PostingsStream, and by --pyserini-postings."""
    pl = rd.get_postings_list(term, analyzer=None)
    if not pl:
        return None
    docs = np.fromiter((p.docid for p in pl), dtype=np.int64, count=len(pl))
    tfs = np.fromiter((p.tf for p in pl), dtype=np.int64, count=len(pl))
    order = np.argsort(docs, kind="stable")
    return docs[order], tfs[order]


def rss_anon_mb():
    """This process's resident anonymous memory (heap, not mmap'd index pages), in MB; 0 if unknown."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def tree_height(num_leaves, r):
    # same expression as the existing scripts (float log, so keep it identical)
    return math.ceil(math.log(num_leaves, r)) if num_leaves > 1 else 0


# ----------------------------------------------------------------------------------------------- per-term math

def quantize(tf, dl, avgdl, idf, k1, b, scale):
    """BM25 upper bound for (possibly zero) tf array -> uint8 like the existing scripts."""
    tf = tf.astype(np.float64)
    sc = np.where(tf > 0, (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl / max(1.0, avgdl))), 0.0) * idf
    return np.clip(np.round(sc * scale), 0, 255).astype(np.uint8)


def group_max(keys, vals):
    """keys sorted ascending -> (unique keys, max of vals per key)."""
    if keys.size == 0:
        return keys, vals
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return keys[starts], np.maximum.reduceat(vals, starts)


def stage1_rows(leaf_ids, leaf_max, r, height, idf, cfg):
    """Yield (node_ids, rows uint8 [n, r]) per level, bottom-up, nodes ascending (matches prepare_data.py order)."""
    child, val = leaf_ids, leaf_max
    for level in range(height - 1, -1, -1):
        parent = child // r
        pos = child % r
        up, inv = np.unique(parent, return_inverse=True)
        tf = np.zeros((up.size, r), dtype=np.int64)
        tf[inv, pos] = val
        rows = quantize(tf, cfg["ia"], cfg["avgdl"], idf, cfg["k1"], cfg["b"], cfg["scale"])
        yield (r ** level + up).astype(np.uint32), rows
        child, val = up, tf.max(axis=1)


def stage2_rows(docs, tfs, B, s, leaf_start, idf, cfg):
    """(leaf node ids, rows uint8 [n, B//s]) with leaves ascending (matches prepare_data_stage2.py order)."""
    nsub = B // s
    leaf = docs // B
    sub = (docs % B) // s
    ul, inv = np.unique(leaf, return_inverse=True)
    tf = np.zeros((ul.size, nsub), dtype=np.int64)
    np.maximum.at(tf, (inv, sub), tfs)
    rows = quantize(tf, cfg["iar"], cfg["avgdl"], idf, cfg["k1"], cfg["b"], cfg["scale"])
    return (leaf_start + ul).astype(np.uint32), rows


# ----------------------------------------------------------------------------------------------- worker

def part_paths(out, w):
    return f"{out}/_parts/w{w:03d}"


VERIFY_TERMS = 20          # per worker: check PostingsStream against pyserini on its first terms with df <= VERIFY_MAX_DF
VERIFY_MAX_DF = 200_000


def worker(args_tuple):
    out, index, terms_file, w, W, plan, cfg = args_tuple   # plan = {B: {r: [s, ...]}}
    rd = open_reader(index)
    stream = None if cfg.get("pyserini_postings") else PostingsStream(rd)
    verified, peak_mb, peak_df = 0, rss_anon_mb(), 0
    N = cfg["N"]
    pdir = part_paths(out, w)
    os.makedirs(pdir, exist_ok=True)

    files = {}

    def f(key):
        if key not in files:
            files[key] = (open(f"{pdir}/{key}.data", "wb"), open(f"{pdir}/{key}.idmap", "wb"), [0])
        return files[key]

    geo = {}
    for B, by_r in plan.items():
        nl = math.ceil(N / B)
        for r in by_r:
            geo[(B, r)] = tree_height(nl, r)

    t0, n_terms = time.time(), 0
    with open(terms_file, encoding="utf-8") as tf_in:
        for tid, line in enumerate(tf_in):
            if tid % W != w:
                continue
            term = json.loads(line)
            got = postings_via_pyserini(rd, term) if stream is None else stream.get(term)
            if got is None:
                continue
            docs, tfs = got
            if stream is not None and verified < VERIFY_TERMS and docs.size <= VERIFY_MAX_DF:
                ref = postings_via_pyserini(rd, term)
                if ref is None or not (np.array_equal(ref[0], docs) and np.array_equal(ref[1], tfs)):
                    raise RuntimeError(f"worker {w}: postings of {term!r} differ from pyserini's get_postings_list "
                                       f"({docs.size} vs {0 if ref is None else ref[0].size} postings); "
                                       f"rerun with --pyserini-postings")
                verified += 1
            if stream is not None and docs.size > 1 and np.any(docs[1:] <= docs[:-1]):
                raise RuntimeError(f"worker {w}: postings of {term!r} are not in ascending docid order")
            idf = math.log(1 + (N - len(docs) + 0.5) / (len(docs) + 0.5))

            for B, by_r in plan.items():
                leaf_ids, leaf_max = group_max(docs // B, tfs)
                for r, s_list in by_r.items():
                    h = geo[(B, r)]
                    # ---- stage 1 (depends on B, r)
                    d, m, cnt = f(f"s1_B{B}_r{r}")
                    for nodes, rows in stage1_rows(leaf_ids, leaf_max, r, h, idf, cfg):
                        rec = np.empty(nodes.size, dtype=IDMAP_DT)
                        rec["term"], rec["node"] = tid, nodes
                        rec["row"] = np.arange(cnt[0], cnt[0] + nodes.size, dtype=np.uint64)
                        d.write(rows.tobytes()); m.write(rec.tobytes()); cnt[0] += nodes.size
                    # ---- stage 2 (depends on B, r via leaf numbering, and s)
                    for s in s_list:
                        d, m, cnt = f(f"s2_B{B}_r{r}_s{s}")
                        nodes, rows = stage2_rows(docs, tfs, B, s, r ** h, idf, cfg)
                        rec = np.empty(nodes.size, dtype=IDMAP_DT)
                        rec["term"], rec["node"] = tid, nodes
                        rec["row"] = np.arange(cnt[0], cnt[0] + nodes.size, dtype=np.uint64)
                        d.write(rows.tobytes()); m.write(rec.tobytes()); cnt[0] += nodes.size
            n_terms += 1
            if docs.size > peak_df or n_terms % 5000 == 0:   # the heap peaks on the longest postings lists
                m = rss_anon_mb()
                if m > peak_mb:
                    peak_mb = m
                peak_df = max(peak_df, docs.size)
            if n_terms % 200000 == 0:
                print(f"  worker {w}: {n_terms:,} terms, {time.time() - t0:.0f}s, heap peak {peak_mb / 1024:.1f} GB "
                      f"(longest postings list {peak_df:,})", flush=True)

    for d, m, _ in files.values():
        d.close(); m.close()
    print(f"  worker {w} done: {n_terms:,} terms in {time.time() - t0:.0f}s, heap peak {peak_mb / 1024:.1f} GB, "
          f"longest postings list {peak_df:,}" + (f", {verified} terms checked against pyserini" if stream else ""),
          flush=True)
    res = {k: v[2][0] for k, v in files.items()}
    res["__peak_anon_mb"], res["__max_df"] = peak_mb, peak_df
    return res


# ----------------------------------------------------------------------------------------------- merge

def merge(out, W, key, dst_data, dst_idmap, counts):
    os.makedirs(os.path.dirname(dst_data), exist_ok=True)
    offset = 0
    with open(dst_data, "wb") as fd, open(dst_idmap, "wb") as fm:
        for w in range(W):
            p = f"{part_paths(out, w)}/{key}"
            n = counts[w].get(key, 0)
            if n == 0:
                continue
            with open(p + ".data", "rb") as src:
                shutil.copyfileobj(src, fd, 64 << 20)
            rec = np.memmap(p + ".idmap", dtype=IDMAP_DT, mode="r")
            assert rec.size == n, f"{p}: {rec.size} idmap rows, expected {n}"
            step = 4_000_000
            for a in range(0, n, step):
                chunk = np.array(rec[a:a + step])
                chunk["row"] += offset
                fm.write(chunk.tobytes())
            offset += n
    return offset


def symlink(target, link):
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.relpath(target, os.path.dirname(link)), link)


def write_stats(T, stats, t_start):
    """<out>/build_stats.json: time and heap peaks of this build, for sizing the next one (datasets/run_all.sh)."""
    stats["total_s"] = round(time.time() - t_start)
    stats["main_peak_anon_gb"] = round(max(stats.get("main_peak_anon_gb", 0), rss_anon_mb() / 1024), 2)
    hist = []
    p = f"{T}/build_stats.json"
    if os.path.exists(p):
        try:
            prev = json.load(open(p))
            hist = prev if isinstance(prev, list) else [prev]
        except Exception:
            pass
    json.dump(hist + [stats], open(p, "w"), indent=1)
    print(f"build stats: {json.dumps(stats)}", flush=True)


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help="../datasets/<name> (corpus/queries/qrels/embeddings)")
    ap.add_argument("--out", required=True, help="Tree artifact folder, e.g. data/<name>_tree")
    ap.add_argument("--name", required=True)
    ap.add_argument("--index", default=None, help="Default: <out>/index_<name>")
    ap.add_argument("--B", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--r", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--s", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--combos", nargs="+", default=None, metavar="B,r,s",
                    help="Explicit (B, r, s) triples instead of the --B x --r x --s product, e.g. 32,32,8 4,2,1 "
                         "(what tree_sweep_grid.sh --shapes lists)")
    ap.add_argument("--db-only", action="store_true",
                    help="Only write the stage 1/2 DBs, vocab and layouts; leave docmap/stage3/queries alone "
                         "(for msmarco/scifact, whose index and stage-3 files already exist)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--skip-db", action="store_true", help="Only (re)write docmap/stage3/queries")
    ap.add_argument("--force", action="store_true", help="Rebuild combos whose files already exist")
    ap.add_argument("--pyserini-postings", action="store_true",
                    help="Read postings with pyserini's get_postings_list (the original, ~10x the memory) instead of "
                         "streaming them from Lucene")
    args = ap.parse_args()
    stats = {"workers": args.workers, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t_start = time.time()

    D, T, name = args.dataset_dir, args.out, args.name
    idx = args.index or f"{T}/index_{name}"
    os.makedirs(T, exist_ok=True)

    rd = open_reader(idx)
    N, avgdl = index_stats(rd)
    cfg = {"N": N, "avgdl": avgdl, "ia": int(avgdl), "iar": int(round(avgdl)),
           "k1": args.k1, "b": args.b, "scale": args.scale, "pyserini_postings": args.pyserini_postings}
    stats["N"] = N
    print(f"N={N:,} avgdl={avgdl:.2f} workers={args.workers}")

    if args.combos:
        combos = sorted({tuple(int(x) for x in c.split(",")) for c in args.combos})
        bad = [c for c in combos if len(c) != 3 or c[2] <= 0 or c[0] % c[2] or c[1] < 2]
        if bad:
            ap.error(f"--combos needs B,r,s with s dividing B and r >= 2, got {bad}")
    else:
        combos = [(B, r, s) for B in args.B for r in args.r for s in args.s if B % s == 0]
    todo = [c for c in combos if args.force or
            not os.path.exists(f"{T}/B{c[0]}_r{c[1]}_s{c[2]}/stage2/idmap.bin")]
    if args.skip_db:
        todo = []
    print(f"{len(combos)} (B, r, s) combos, {len(todo)} to build: {todo}")

    # ---------------- vocab (term id = Lucene term enumeration order, like build_small_test_artifacts.py)
    vocab_path = f"{T}/stage1_vocab.json"
    terms_file = f"{T}/_terms.jsonl"
    if todo or not os.path.exists(vocab_path):
        n_terms = 0
        with open(terms_file, "w", encoding="utf-8") as ft, open(vocab_path, "w", encoding="utf-8") as fv:
            fv.write("{")
            for i, t in enumerate(rd.terms()):
                ft.write(json.dumps(t.term) + "\n")
                fv.write(("," if i else "") + json.dumps(t.term) + ":" + json.dumps(str(i)))
                n_terms += 1
            fv.write("}")
        print(f"vocab: {n_terms:,} terms")

    # ---------------- stage 1 / stage 2 DBs for every combo still missing
    if todo:
        # only the (B, r, s) still missing, not the product of their values
        plan = {}
        for B, r, s in todo:
            plan.setdefault(B, {}).setdefault(r, []).append(s)
        W = max(1, args.workers)
        jobs = [(T, idx, terms_file, w, W, plan, cfg) for w in range(W)]
        t0 = time.time()
        if W == 1:
            counts = [worker(jobs[0])]
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(W) as pool:
                counts = pool.map(worker, jobs)
        peaks = [c.pop("__peak_anon_mb", 0) for c in counts]
        max_df = max(c.pop("__max_df", 0) for c in counts)
        print(f"postings pass done in {time.time() - t0:.0f}s, merging. Heap peaks: workers max "
              f"{max(peaks) / 1024:.1f} GB, all {W} together at most {sum(peaks) / 1024:.1f} GB (each worker's own "
              f"peak, not simultaneous); main {rss_anon_mb() / 1024:.1f} GB; longest postings list {max_df:,} of {N:,} docs",
              flush=True)
        stats["postings_pass_s"] = round(time.time() - t0)
        stats["worker_peak_anon_gb"] = [round(p / 1024, 2) for p in peaks]
        stats["max_df"] = max_df

        for B, by_r in plan.items():
            for r, s_list in by_r.items():
                s1 = f"{T}/stage1/B{B}_r{r}"
                n1 = merge(T, W, f"s1_B{B}_r{r}", f"{s1}/data.bin", f"{s1}/idmap.bin", counts)
                for s in s_list:
                    out = f"{T}/B{B}_r{r}_s{s}"
                    n2 = merge(T, W, f"s2_B{B}_r{r}_s{s}", f"{out}/stage2/data.bin", f"{out}/stage2/idmap.bin", counts)
                    json.dump({"N": N, "B": B, "r": r, "min_dl_map": {}}, open(f"{out}/layout.json", "w"))
                    symlink(f"{s1}/data.bin", f"{out}/stage1_data.bin")
                    symlink(f"{s1}/idmap.bin", f"{out}/stage1_idmap.bin")
                    symlink(vocab_path, f"{out}/stage1_vocab.json")
                    print(f"B={B} r={r} s={s}: height={tree_height(math.ceil(N / B), r)} "
                          f"stage1 rows={n1:,} stage2 rows={n2:,}")
        shutil.rmtree(f"{T}/_parts", ignore_errors=True)
    if os.path.exists(terms_file):
        os.remove(terms_file)
    if args.db_only:
        write_stats(T, stats, t_start)
        print(f"--db-only: skipped docmap/stage3/queries. Done: {T}")
        return

    # ---------------- docmap.bin (tools/create_docmap.py): internal docid -> external (integer) id
    ext = [rd.convert_internal_docid_to_collection_docid(i) for i in range(N)]
    with open(f"{idx}/docmap.bin", "wb") as f:
        f.write(np.asarray([int(e) for e in ext], dtype="<u8").tobytes())

    # ---------------- stage3 doc embeddings in index order (+ ids)
    emb = f"{D}/{name}_corpus_embed_f32.npy"
    if not os.path.exists(f"{T}/stage3_doc.npy.ids") or args.force:
        if os.path.exists(emb + ".ids"):
            with open(emb + ".ids") as fi:
                ids = [l.rstrip("\n") for l in fi]
        else:   # older embeddings (msmarco) have no .ids: their rows follow corpus.jsonl, as bins assumes too
            print(f"no {emb}.ids: taking the row order from {D}/corpus.jsonl")
            with open(f"{D}/corpus.jsonl") as fi:
                ids = [json.loads(l)["_id"] for l in fi]
        if ids == ext:
            # single-threaded index => already in corpus order, no need to copy ~100 GB
            symlink(emb, f"{T}/stage3_doc.npy")
            print("stage3: index order == embedding order, symlinked")
        else:
            from numpy.lib.format import open_memmap
            pos = {x: i for i, x in enumerate(ids)}
            rows = np.fromiter((pos[e] for e in ext), dtype=np.int64, count=N)
            del pos
            Y = np.load(emb, mmap_mode="r")
            out = open_memmap(f"{T}/stage3_doc.npy", mode="w+", dtype=np.float32, shape=(N, Y.shape[1]))
            for a in range(0, N, 1_000_000):
                out[a:a + 1_000_000] = Y[rows[a:a + 1_000_000]]
            del out
            print("stage3: reordered embeddings to index order")
        with open(f"{T}/stage3_doc.npy.ids", "w") as f:
            f.write("\n".join(ext) + "\n")
    stats["main_peak_anon_gb"] = round(rss_anon_mb() / 1024, 2)   # docmap + stage3 ids: the main process's peak
    del ext

    # ---------------- queries.tsv + analyzed json (opt3.py) + qrels.trec, only queries with qrels
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    # BEIR layout (qrels/test.tsv with a header, queries.jsonl), or msmarco's (qrels/qrels.dev.tsv in TREC form
    # "qid 0 pid rel", queries.dev.small.jsonl, the same files bins uses in main.go). Pick the qrels that go with the
    # queries file: ../datasets/msmarco also has a qrels/test.tsv (TREC-DL), whose query ids are not in
    # queries.dev.small.jsonl, and choosing it by itself wrote an empty queries.qrels.tsv (2026-11-10).
    if os.path.exists(f"{D}/queries.jsonl"):
        qfile, qp = f"{D}/queries.jsonl", f"{D}/qrels/test.tsv"
    else:
        qfile, qp = f"{D}/queries.dev.small.jsonl", f"{D}/qrels/qrels.dev.tsv"
    qrels = []
    for l in open(qp).read().splitlines():
        f = l.split()
        if len(f) >= 4:
            f = [f[0], f[2], f[3]]
        if len(f) == 3 and f[2].lstrip("-").isdigit():   # skips the BEIR header
            qrels.append(f)
    want = {q[0] for q in qrels}
    an = Analyzer(get_lucene_analyzer())
    qs = [json.loads(l) for l in open(qfile)]
    for q in qs:   # BEIR queries have "_id", msmarco's queries.dev.small.jsonl has "id"
        if "_id" not in q:
            q["_id"] = str(q["id"])
    qs = [q for q in qs if q["_id"] in want]
    if not qs:   # every tree run would answer 0 queries and fail after its whole preprocessing
        sys.exit(f"no query in {qfile} has qrels in {qp}: not writing the query files")
    with open(f"{T}/queries.qrels.tsv", "w") as f:
        for q in qs:
            f.write(f"{q['_id']}\t{q['text']}\n")
    json.dump([{"id": q["_id"], "text": q["text"], "terms": an.analyze(q["text"])} for q in qs],
              open(f"{T}/queries.qrels.analyzed.json", "w"))
    with open(f"{T}/qrels.trec", "w") as f:
        for q in qrels:
            f.write(f"{q[0]} 0 {q[1]} {q[2]}\n")
    print(f"queries: {len(qs):,}. Done: {T}")
    write_stats(T, stats, t_start)


if __name__ == "__main__":
    main()
