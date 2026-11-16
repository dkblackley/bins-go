"""
Helpers shared by the Python baselines, written to match main.go exactly so
their metadata.json is read by plotting/load_results.py like any Go run:

  * dataset paths       = GetDatasets() in main.go
  * loaders             = bins/dataloader.go (LoadCorpus / LoadQueries / LoadQrels)
  * calc_mrr/calc_recall= calcMRR / calcRecall in main.go
  * cosine_rerank       = CosineReRank in main.go (rerank by cosine with the
                          dataset's query/doc embeddings, keep top-k)
  * net_time            = pianopir.netTime (LAN 1000 Mbps / 5 ms, WAN 400 Mbps / 50 ms)
"""

import json
import os
import re

import numpy as np

DATASETS = {
    # name: (corpus, queries, qrels, corpus_vec, query_vec)
    "msmarco": ("msmarco/corpus.jsonl", "msmarco/queries.dev.small.jsonl", "msmarco/qrels/qrels.dev.tsv",
                "msmarco/msmarco_corpus_embed_f32.npy", "msmarco/msmarco_query_embed_f32.npy"),
    "scifact": ("scifact/corpus.jsonl", "scifact/queries.jsonl", "scifact/qrels/test.tsv",
                "scifact/scifact_corpus_embed_f32.npy", "scifact/scifact_query_embed_f32.npy"),
    "trec-covid": ("trec-covid/corpus.jsonl", "trec-covid/queries.jsonl", "trec-covid/qrels/test.tsv",
                   "trec-covid/trec_corpus_embed_f32.npy", "trec-covid/trec_query_embed_f32.npy"),
}


def generic_layout(name):
    """The layout datasets/prepare_dataset.py + embed.slurm write (nq, fever, msmarco-v2, ...), = GetDatasets' fallback."""
    return (f"{name}/corpus.jsonl", f"{name}/queries.jsonl", f"{name}/qrels/test.tsv",
            f"{name}/{name}_corpus_embed_f32.npy", f"{name}/{name}_query_embed_f32.npy")


def dataset_paths(root, name):
    files = DATASETS.get(name) or generic_layout(name)
    paths = {k: os.path.join(root, v) for k, v in zip(("corpus", "queries", "qrels", "corpus_vec", "query_vec"), files)}
    missing = [v for v in paths.values() if not os.path.exists(v)]
    if missing:
        raise SystemExit(f"dataset {name!r}: missing {', '.join(missing)} (run datasets/prepare_dataset.py + embed first)")
    return paths


def num_docs(root, name):
    """Corpus size without reading the corpus: num_docs.txt if prepare_dataset.py wrote one, else the embedding rows."""
    f = os.path.join(root, name, "num_docs.txt")
    if os.path.exists(f):
        return int(open(f).read().split()[0])
    return int(np.load(dataset_paths(root, name)["corpus_vec"], mmap_mode="r").shape[0])


def load_corpus(path):
    """[(doc_id, title, text)] in file order (row i == embedding row i)."""
    docs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            body = d.get("text") or d.get("abstract") or ""
            docs.append((str(d.get("_id", "")), d.get("title", "") or "", body))
    return docs


def iter_corpus(path):
    """Like load_corpus, one doc at a time (for corpora too big to hold, e.g. msmarco-v2's 138M docs)."""
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            yield str(d.get("_id", "")), d.get("title", "") or "", d.get("text") or d.get("abstract") or ""


_ID_RE = re.compile(r'"_id"\s*:\s*"([^"\\]*)"')


def ids_for_rows(path, rows):
    """{row: doc_id} for just the corpus rows a run returned, in one streaming pass (no corpus in memory)."""
    want, out = set(int(r) for r in rows), {}
    if not want:
        return out
    last = max(want)
    with open(path) as f:
        r = -1
        for line in f:
            if not line.strip():
                continue
            r += 1
            if r in want:
                m = _ID_RE.search(line)
                out[r] = m.group(1) if m else str(json.loads(line).get("_id", ""))
                if r == last:
                    break
    return out


class AllRows:
    """doc_index for cosine_rerank when results already hold corpus row numbers."""
    def __contains__(self, r):
        return True

    def __getitem__(self, r):
        return r


def rows_to_ids(results, idmap):
    return {qid: [idmap[r] for r in rows] for qid, rows in results.items()}


def load_queries(path):
    """[(qid, text)] in file order, same ID fallback as LoadQueries."""
    qs = []
    with open(path) as f:
        for line in f:
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = q.get("_id") or q.get("id") or q.get("query_id") or ""
            qs.append((str(qid), q.get("text", "")))
    return qs


def load_qrels(path):
    """qid -> {docid: score}, positive scores only (LoadQrels)."""
    rel = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            qid, docid, score = (p[0], p[2], p[3]) if len(p) > 3 else (p[0], p[1], p[2])
            try:
                v = int(score)
            except ValueError:
                continue
            if v <= 0:
                continue
            rel.setdefault(qid, {})[docid] = v
    return rel


def calc_mrr(results, qrels):
    total, n = 0.0, 0
    for qid, ranked in results.items():
        rel = qrels.get(qid)
        if not rel:
            continue
        for i, d in enumerate(ranked):
            if d in rel:
                total += 1.0 / (i + 1)
                break
        n += 1
    return total / n if n else 0.0


def calc_recall(results, qrels):
    total, n = 0.0, 0
    for qid, ranked in results.items():
        rel = qrels.get(qid)
        if not rel:
            continue
        total += len(set(ranked) & set(rel)) / len(rel)
        n += 1
    return total / n if n else 0.0


def cosine_rerank(results, k, doc_index, doc_vecs, query_vecs_by_qid):
    """CosineReRank: rescore each returned list by cosine similarity and keep the top-k."""
    out = {}
    for qid, ids in results.items():
        q = query_vecs_by_qid.get(qid)
        if q is None:
            out[qid] = ids[:k]
            continue
        seen, rows, kept = set(), [], []
        for d in ids:
            if d in seen or d not in doc_index:
                continue
            seen.add(d)
            rows.append(doc_index[d])
            kept.append(d)
        if not rows:
            out[qid] = []
            continue
        m = np.asarray(doc_vecs[rows], dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        mn = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        order = np.argsort(-(mn @ qn), kind="stable")[:k]
        out[qid] = [kept[i] for i in order]
    return out


def net_time(nbytes, rounds, mbps, rtt):
    return nbytes * 8 / (mbps * 1e6) + rounds * rtt


def write_run(out_dir, meta, results=None, reranked=None):
    """metadata.json (+ results.json / results_reRank.json) with every value as a string, like writeAnswers."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({k: (v if isinstance(v, str) else repr(v) if isinstance(v, float) else str(v))
                   for k, v in meta.items()}, f, indent=2)
    if results is not None:
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    if reranked is not None:
        with open(os.path.join(out_dir, "results_reRank.json"), "w") as f:
            json.dump(reranked, f, indent=2)
