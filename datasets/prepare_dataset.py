#!/usr/bin/env python3
"""
Convert a downloaded retrieval dataset into the layout main.go expects (see GetDatasets):

    <out>/corpus.jsonl      {"_id", "title", "text"} one doc per line (no other fields, LoadCorpus rejects them)
    <out>/queries.jsonl     {"_id", "text"}          only queries with >= 1 relevant doc in the corpus
    <out>/qrels/test.tsv    query-id \t corpus-id \t score   (with header, like BEIR)
    <out>/num_docs.txt      the -n value to pass to pir_app (the slurm scripts read it)
    <out>/docid_map.tsv     new_id \t original_id   (only with --remap-ids, the default)
    <out>/stats.json

By default doc IDs are renumbered 0..N-1 in output order. That keeps the PIR ID entries short and fixed width
(StringsToUint64Grid pads every ID to the longest one) and the tree's docmap.bin needs integer IDs anyway.

Formats
  --format beir        <src>/corpus.jsonl[.gz], queries.jsonl[.gz], qrels/<split>.tsv   (NQ, HotpotQA, FEVER, ...)
  --format msmarco-v2  <src>/msmarco_v2_passage/msmarco_passage_*.gz, passv2_<split>_queries.tsv, passv2_<split>_qrels.tsv
  --format miracl      <src>/miracl-corpus-v1.0-en/*.jsonl.gz, miracl-v1.0-en/{topics,qrels}/*-<split>.tsv

Subsampling (e.g. MS MARCO v2 down to 100M, to match PACMANN's SIFT100M)
  --max-docs N keeps every doc that is relevant to some kept query, then fills up to N with a uniform random sample
  (seeded). If the source is an already-prepared dataset that has embeddings, pass --src-embed / --src-query-embed
  and the matching rows are copied too, so the subsample doesn't need re-encoding.
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np


def zopen(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    sys.exit(f"None of these exist: {paths}")


# ---------------------------------------------------------------- readers: each yields (orig_id, title, text)

def beir_corpus(src):
    path = first_existing(f"{src}/corpus.jsonl", f"{src}/corpus.jsonl.gz")
    with zopen(path) as f:
        for line in f:
            d = json.loads(line)
            yield str(d["_id"]), d.get("title", "") or "", d.get("text", "") or ""


def beir_queries(src):
    path = first_existing(f"{src}/queries.jsonl", f"{src}/queries.jsonl.gz")
    with zopen(path) as f:
        for line in f:
            d = json.loads(line)
            yield str(d["_id"]), d.get("text", "")


def beir_qrels(src, split):
    return read_qrels(first_existing(f"{src}/qrels/{split}.tsv"))


def msmarco_v2_corpus(src):
    files = sorted(glob.glob(f"{src}/msmarco_v2_passage/msmarco_passage_*.gz")) or \
        sorted(glob.glob(f"{src}/msmarco_passage_*.gz"))
    if not files:
        sys.exit(f"No msmarco_passage_*.gz under {src} (did you untar msmarco_v2_passage.tar?)")
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                yield d["pid"], "", d["passage"]


def msmarco_v2_queries(src, split):
    with open(first_existing(f"{src}/passv2_{split}_queries.tsv")) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            yield qid, text


def msmarco_v2_qrels(src, split):
    return read_qrels(first_existing(f"{src}/passv2_{split}_qrels.tsv"))


def miracl_corpus(src):
    files = sorted(glob.glob(f"{src}/miracl-corpus-v1.0-en/*.jsonl.gz")) or \
        sorted(glob.glob(f"{src}/miracl-corpus-v1.0-en/*.jsonl"))
    if not files:
        sys.exit(f"No miracl-corpus-v1.0-en/*.jsonl(.gz) under {src}")
    for fp in files:
        with zopen(fp) as f:
            for line in f:
                d = json.loads(line)
                yield d["docid"], d.get("title", "") or "", d.get("text", "") or ""


def miracl_queries(src, split):
    with open(first_existing(f"{src}/miracl-v1.0-en/topics/topics.miracl-v1.0-en-{split}.tsv")) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            yield qid, text


def miracl_qrels(src, split):
    return read_qrels(first_existing(f"{src}/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-{split}.tsv"))


def query_length_stats(texts):
    """Whitespace word counts. The lexical side sees fewer terms (stopwords removed), so this is an upper bound."""
    if not texts:
        return {}
    n = np.array([len(t.split()) for t in texts])
    return {"mean": round(float(n.mean()), 2), "median": float(np.median(n)), "p90": float(np.percentile(n, 90)),
            "max": int(n.max()), "frac_le_10": round(float((n <= 10).mean()), 3),
            "frac_le_15": round(float((n <= 15).mean()), 3), "frac_le_20": round(float((n <= 20).mean()), 3)}


def read_qrels(path):
    """qid -> {docid: score} for score > 0. Handles BEIR (qid docid score, with header) and TREC (qid 0 docid score)."""
    rels = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            if len(p) > 3:
                p = [p[0], p[2], p[3]]
            try:
                score = int(p[2])
            except ValueError:  # header
                continue
            if score > 0:
                rels.setdefault(p[0], {})[p[1]] = score
    return rels


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["beir", "msmarco-v2", "miracl"], required=True)
    ap.add_argument("--src", required=True, help="Downloaded dataset folder")
    ap.add_argument("--out", required=True, help="Output folder, e.g. ../datasets/nq")
    ap.add_argument("--split", default=None, help="qrels split (beir default: test, msmarco-v2 default: dev)")
    ap.add_argument("--max-docs", type=int, default=0, help="Subsample the corpus to this many docs (0 = keep all)")
    ap.add_argument("--max-queries", type=int, default=0, help="Subsample queries (0 = keep all with qrels)")
    ap.add_argument("--max-query-words", type=int, default=0,
                    help="Drop queries longer than this many words (0 = keep all). Changes the benchmark, so say so.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-remap-ids", dest="remap_ids", action="store_false", help="Keep the original doc IDs")
    ap.add_argument("--src-embed", default=None, help="Corpus embeddings (.npy) row-aligned with <src>/corpus.jsonl")
    ap.add_argument("--src-query-embed", default=None, help="Query embeddings (.npy) row-aligned with <src>/queries.jsonl")
    ap.add_argument("--name", default=None, help="Prefix for copied embedding files (default: basename of --out)")
    args = ap.parse_args()

    split = args.split or ("test" if args.format == "beir" else "dev")
    name = args.name or os.path.basename(os.path.normpath(args.out))
    rng = np.random.default_rng(args.seed)
    os.makedirs(f"{args.out}/qrels", exist_ok=True)

    if args.format == "beir":
        corpus = lambda: beir_corpus(args.src)
        queries = list(beir_queries(args.src))
        qrels = beir_qrels(args.src, split)
    elif args.format == "msmarco-v2":
        corpus = lambda: msmarco_v2_corpus(args.src)
        queries = list(msmarco_v2_queries(args.src, split))
        qrels = msmarco_v2_qrels(args.src, split)
    else:
        corpus = lambda: miracl_corpus(args.src)
        queries = list(miracl_queries(args.src, split))
        qrels = miracl_qrels(args.src, split)

    # ---- queries: keep the ones that have qrels (and optionally subsample them)
    q_order = [i for i, (qid, _) in enumerate(queries) if qid in qrels]
    print("query words (all with qrels):", query_length_stats([queries[i][1] for i in q_order]))
    if args.max_query_words:
        q_order = [i for i in q_order if len(queries[i][1].split()) <= args.max_query_words]
        print(f"--max-query-words {args.max_query_words}: {len(q_order):,} queries left")
    if args.max_queries and len(q_order) > args.max_queries:
        q_order = sorted(rng.choice(q_order, size=args.max_queries, replace=False).tolist())
    kept_qids = {queries[i][0] for i in q_order}
    relevant = {d for q in kept_qids for d in qrels[q]}
    print(f"{len(queries):,} queries, {len(kept_qids):,} with qrels kept, {len(relevant):,} relevant docs")

    # ---- pass 1 (only when subsampling): count docs, find positions of the relevant ones
    keep_mask = None
    if args.max_docs:
        total, rel_pos = 0, []
        for i, (oid, _, _) in enumerate(corpus()):
            if oid in relevant:
                rel_pos.append(i)
            total = i + 1
            if total % 10_000_000 == 0:
                print(f"  pass 1: {total:,} docs")
        print(f"Source corpus: {total:,} docs, {len(rel_pos):,} relevant docs found")
        if args.max_docs < total:
            keep_mask = np.zeros(total, dtype=bool)
            keep_mask[rel_pos] = True
            fill = args.max_docs - len(rel_pos)
            if fill < 0:
                sys.exit(f"--max-docs {args.max_docs} is smaller than the {len(rel_pos)} relevant docs")
            others = np.flatnonzero(~keep_mask)
            keep_mask[rng.choice(others, size=fill, replace=False)] = True
            del others

    # ---- pass 2: write corpus
    old2new = {}  # only for relevant docs (needed for qrels)
    src_rows = [] if args.src_embed else None
    n_out = 0
    with open(f"{args.out}/corpus.jsonl", "w", encoding="utf-8") as fc, \
            (open(f"{args.out}/docid_map.tsv", "w") if args.remap_ids else open(os.devnull, "w")) as fm:
        for i, (oid, title, text) in enumerate(corpus()):
            if keep_mask is not None and (i >= len(keep_mask) or not keep_mask[i]):
                continue
            nid = str(n_out) if args.remap_ids else oid
            fc.write(json.dumps({"_id": nid, "title": title, "text": text}, ensure_ascii=False) + "\n")
            if args.remap_ids:
                fm.write(f"{nid}\t{oid}\n")
            if oid in relevant:
                old2new[oid] = nid
            if src_rows is not None:
                src_rows.append(i)
            n_out += 1
            if n_out % 10_000_000 == 0:
                print(f"  wrote {n_out:,} docs")

    # ---- queries + qrels (drop judgements whose doc isn't in the corpus, then queries left with none)
    n_rels, final_q = 0, []
    with open(f"{args.out}/qrels/test.tsv", "w") as fr:
        fr.write("query-id\tcorpus-id\tscore\n")
        for i in q_order:
            qid = queries[i][0]
            rels = [(old2new[d], s) for d, s in qrels[qid].items() if d in old2new]
            if not rels:
                continue
            final_q.append(i)
            for d, s in rels:
                fr.write(f"{qid}\t{d}\t{s}\n")
                n_rels += 1
    with open(f"{args.out}/queries.jsonl", "w", encoding="utf-8") as fq:
        for i in final_q:
            fq.write(json.dumps({"_id": queries[i][0], "text": queries[i][1]}, ensure_ascii=False) + "\n")

    # ---- optional: copy embedding rows so subsamples don't need re-encoding
    if args.src_embed:
        from numpy.lib.format import open_memmap
        src = np.load(args.src_embed, mmap_mode="r")
        assert src.shape[0] == len(keep_mask if keep_mask is not None else src_rows), \
            f"{args.src_embed} has {src.shape[0]} rows, corpus has a different count"
        out_path = f"{args.out}/{name}_corpus_embed_f32.npy"
        dst = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_out, src.shape[1]))
        rows = np.asarray(src_rows, dtype=np.int64)
        step = 1_000_000
        for a in range(0, n_out, step):
            dst[a:a + step] = src[rows[a:a + step]]
        del dst
        with open(out_path + ".ids", "w") as f, open(f"{args.out}/corpus.jsonl") as fc:
            for line in fc:
                f.write(json.loads(line)["_id"] + "\n")
        print(f"Copied {n_out:,} corpus embedding rows -> {out_path}")
    if args.src_query_embed:
        srcq = np.load(args.src_query_embed, mmap_mode="r")
        assert srcq.shape[0] == len(queries), f"{args.src_query_embed}: {srcq.shape[0]} rows vs {len(queries)} queries"
        out_path = f"{args.out}/{name}_query_embed_f32.npy"
        np.save(out_path, np.asarray(srcq[final_q], dtype=np.float32))
        print(f"Copied {len(final_q):,} query embedding rows -> {out_path}")

    with open(f"{args.out}/num_docs.txt", "w") as f:
        f.write(f"{n_out}\n")
    stats = {"name": name, "format": args.format, "src": os.path.abspath(args.src), "split": split,
             "num_docs": n_out, "num_queries": len(final_q), "num_qrels": n_rels,
             "max_docs": args.max_docs, "max_queries": args.max_queries, "max_query_words": args.max_query_words,
             "seed": args.seed, "query_words": query_length_stats([queries[i][1] for i in final_q]),
             "remapped_ids": args.remap_ids}
    with open(f"{args.out}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
