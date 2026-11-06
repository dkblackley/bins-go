#!/usr/bin/env python3
"""
Build tree PIR artifacts (Stage-1 + Stage-2 DBs, vocab, layout) for one tree
arity r, one block size B and any number of sub-block sizes s, in a single pass
over the Lucene postings.

Output layout (what run_tree_pir_old.slurm expects):
    <out-root>/B{B}_r{r}_s{s}/layout.json
    <out-root>/B{B}_r{r}_s{s}/stage1_vocab.json
    <out-root>/B{B}_r{r}_s{s}/stage1_data.bin       (shared across s via symlink)
    <out-root>/B{B}_r{r}_s{s}/stage1_idmap.bin      (shared across s via symlink)
    <out-root>/B{B}_r{r}_s{s}/stage2/data.bin
    <out-root>/B{B}_r{r}_s{s}/stage2/idmap.bin

The maths is the same as prepare_data.py / prepare_data_stage2.py (no min_dl
map, avgdl-based BM25 upper bounds, uint8 quantisation with --scale) and as
build_small_test_artifacts.py, but done sparsely with numpy so that small B / r
(deep trees, millions of leaves) stay cheap on MS MARCO.

Usage:
    python tree/build_tree_artifacts.py --index data/msmarco_reordered \
        --out-root data/msmarco/pir --B 16 --r 4 --s 2,4,8 \
        [--vocab data/msmarco/pir/B32_r32_s8/stage1_vocab.json] [--force]
"""
import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
from pyserini.index.lucene import LuceneIndexReader

IDMAP_DTYPE = np.dtype([("term", "<u4"), ("node", "<u4"), ("row", "<u8")])   # == struct '<IIQ'


def go_height(num_leaves, r):
    """Must match LayoutWithStats.Height() in tree/internal/bm25/scoring.go bit for bit."""
    if r <= 1 or num_leaves <= 1:
        return 0
    h = int(math.ceil(math.log(num_leaves) / math.log(r)))
    exact = 0
    while r ** exact < num_leaves:
        exact += 1
    if exact != h:
        print(f"WARNING: float height {h} != integer height {exact} for num_leaves={num_leaves} r={r}; "
              f"using {h} to match the Go loader", file=sys.stderr)
    return h


def quantise(scores, scale):
    return np.clip(np.round(scores * scale), 0, 255).astype(np.uint8)


def group_max(keys, values):
    """keys sorted ascending -> (unique keys, max value per key)."""
    if keys.size == 0:
        return keys, values
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return keys[starts], np.maximum.reduceat(values, starts)


class Writer:
    def __init__(self, data_path, idmap_path):
        self.data = open(data_path, "wb")
        self.idmap = open(idmap_path, "wb")
        self.rows = 0

    def write(self, term_id, nodes, rows):
        n = rows.shape[0]
        rec = np.empty(n, dtype=IDMAP_DTYPE)
        rec["term"] = term_id
        rec["node"] = nodes
        rec["row"] = np.arange(self.rows, self.rows + n, dtype=np.uint64)
        self.data.write(np.ascontiguousarray(rows).tobytes())
        self.idmap.write(rec.tobytes())
        self.rows += n

    def close(self):
        self.data.close()
        self.idmap.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="Pyserini/Lucene index (internal docids = tree order)")
    ap.add_argument("--out-root", required=True, help="Directory that holds the B*_r*_s* artifact dirs")
    ap.add_argument("--B", type=int, required=True, help="Leaf block size (docs per leaf)")
    ap.add_argument("--r", type=int, required=True, help="Tree arity for Round 1")
    ap.add_argument("--s", required=True, help="Comma separated sub-block sizes for Round 2, each must divide B")
    ap.add_argument("--vocab", help="Existing stage1_vocab.json (term -> id) to keep term IDs identical across "
                                    "artifacts; default enumerates the index terms")
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--force", action="store_true", help="Rebuild even if the artifact dir is complete")
    args = ap.parse_args()

    B, r = args.B, args.r
    s_list = sorted({int(x) for x in args.s.split(",") if x})
    if r < 2:
        ap.error("--r must be >= 2")
    for s in s_list:
        if s <= 0 or B % s:
            ap.error(f"s={s} must be positive and divide B={B}")

    def art_dir(s):
        return os.path.join(args.out_root, f"B{B}_r{r}_s{s}")

    def complete(s):
        d = art_dir(s)
        need = ["layout.json", "stage1_vocab.json", "stage1_data.bin", "stage1_idmap.bin",
                "stage2/data.bin", "stage2/idmap.bin"]
        return all(os.path.exists(os.path.join(d, f)) for f in need)

    todo = [s for s in s_list if args.force or not complete(s)]
    if not todo:
        print(f"B={B} r={r} s={s_list}: all artifacts already exist, nothing to do (use --force to rebuild)")
        return
    print(f"Building B={B} r={r} s={todo} (skipping complete: {sorted(set(s_list) - set(todo))})")

    rd = LuceneIndexReader(args.index)
    st = rd.stats()
    N = int(st["documents"])
    avgdl = st["total_terms"] / N
    ia, iar = int(avgdl), int(round(avgdl))   # stage1 / stage2 use these, as in the original scripts
    num_leaves = math.ceil(N / B)
    H = go_height(num_leaves, r)
    leaf_start = r ** H
    if leaf_start + num_leaves > 0xFFFFFFFF:
        sys.exit(f"node IDs up to {leaf_start + num_leaves} do not fit the uint32 Cuckoo key; pick a larger B or r")
    print(f"N={N} avgdl={avgdl:.2f} num_leaves={num_leaves} height={H} leaf_start={leaf_start}")

    def ub(tf, dl):
        tf = tf.astype(np.float64)
        return (args.k1 + 1) * tf / (tf + args.k1 * (1 - args.b + args.b * dl / max(1.0, avgdl)))

    def idf(df):
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    # ---- vocab ----
    if args.vocab:
        with open(args.vocab, encoding="utf-8") as f:
            vocab = {t: int(i) for t, i in json.load(f).items()}
    else:
        vocab = {t.term: i for i, t in enumerate(rd.terms())}

    # ---- outputs: stage1 once per (B, r), symlinked into every s dir ----
    shared = os.path.join(args.out_root, f"_stage1_B{B}_r{r}")
    os.makedirs(shared, exist_ok=True)
    s1 = Writer(os.path.join(shared, "stage1_data.bin"), os.path.join(shared, "stage1_idmap.bin"))
    s2 = {}
    for s in todo:
        d = art_dir(s)
        os.makedirs(os.path.join(d, "stage2"), exist_ok=True)
        s2[s] = Writer(os.path.join(d, "stage2", "data.bin"), os.path.join(d, "stage2", "idmap.bin"))

    t0 = time.time()
    for n_done, (term, tid) in enumerate(vocab.items()):
        if n_done % 200000 == 0:
            print(f"  {n_done}/{len(vocab)} terms, {time.time() - t0:.0f}s, stage1 rows={s1.rows}", flush=True)
        pl = rd.get_postings_list(term, analyzer=None)
        if not pl:
            continue
        docs = np.fromiter((p.docid for p in pl), dtype=np.int64, count=len(pl))
        tfs = np.fromiter((p.tf for p in pl), dtype=np.int64, count=len(pl))
        order = np.argsort(docs, kind="stable")
        docs, tfs = docs[order], tfs[order]
        tidf = idf(len(docs))

        # Stage 1: walk up from the leaves; each (term, parent) row holds its r child scores.
        idx, val = group_max(docs // B, tfs)
        for level in range(H - 1, -1, -1):
            parent = idx // r
            p_uniq, p_max = group_max(parent, val)
            rows = np.zeros((p_uniq.size, r), dtype=np.uint8)
            rows[np.searchsorted(p_uniq, parent), idx % r] = quantise(ub(val, ia) * tidf, args.scale)
            s1.write(tid, (r ** level + p_uniq).astype(np.uint32), rows)
            idx, val = p_uniq, p_max

        # Stage 2: one row per (term, leaf) with the B/s sub-block scores.
        for s, w in s2.items():
            nsub = B // s
            key, kmax = group_max((docs // B) * nsub + (docs % B) // s, tfs)
            leaf = key // nsub
            l_uniq = np.unique(leaf)
            rows = np.zeros((l_uniq.size, nsub), dtype=np.uint8)
            rows[np.searchsorted(l_uniq, leaf), key % nsub] = quantise(ub(kmax, iar) * tidf, args.scale)
            w.write(tid, (leaf_start + l_uniq).astype(np.uint32), rows)

    s1.close()
    for w in s2.values():
        w.close()

    vocab_json = {t: str(i) for t, i in vocab.items()}
    for s in todo:
        d = art_dir(s)
        for f in ("stage1_data.bin", "stage1_idmap.bin"):
            dst = os.path.join(d, f)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.relpath(os.path.join(shared, f), d), dst)
        with open(os.path.join(d, "stage1_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_json, f)
        with open(os.path.join(d, "layout.json"), "w") as f:
            json.dump({"N": N, "B": B, "r": r, "min_dl_map": {}}, f)
        print(f"wrote {d}: stage2 rows={s2[s].rows}")
    print(f"stage1 rows={s1.rows}, done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
