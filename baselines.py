"""
Non-private retrieval baselines: what each scoring function gets with no PIR
at all, so the paper can show how close the private methods come.

    python baselines.py                                   # msmarco + scifact, k=10
    python baselines.py --datasets scifact --k 10 100
    python baselines.py --dataset-root /scratch/dblackle/datasets --out baselines

For every dataset and k it writes <out>/<dataset>_k<k>.json holding MRR@k and
Recall@k for:

    bm25            BM25 over the whole corpus (Lucene index via pyserini,
                    falls back to rank_bm25 if pyserini/Java is missing)
    dense           exact cosine similarity over the whole corpus, using the
                    same 192-d embeddings the PIR methods use
    hybrid          true score fusion: per query, take the union of the BM25
                    and dense top-<depth>, min-max normalise each score over
                    that union and rank by alpha * bm25 + (1 - alpha) * dense.
                    A doc missing from the BM25 list gets that list's lowest
                    BM25 score (same convention as pyserini's HybridSearcher);
                    dense scores are exact for every candidate.
    hybrid_best     same, alpha swept over 0.0..1.0 and the best one by MRR.
                    Tuned on the test queries, so it is an upper bound.
    hybrid_rrf      reciprocal rank fusion of the two lists (rank-based, no
                    score normalisation, k_rrf = 60)
    bm25_rerank_N   BM25 top-N, then reranked by cosine similarity and cut to
                    k. This is the plaintext version of what PILLAR does
                    (keyword candidates, local dense rerank).

Metrics match main.go: averaged over every query in the query file that has
at least one relevant doc in qrels; a query with no results scores 0.
"""

import argparse
import json
import logging
import math
import os
import re
import time
from collections import defaultdict

import numpy as np

log = logging.getLogger('baselines')

# Mirrors GetDatasets() in main.go. {root} is --dataset-root.
DATASETS = {
    'msmarco': dict(
        index='{root}/index_marco',
        corpus='{root}/msmarco/corpus.jsonl',
        queries='{root}/msmarco/queries.dev.small.jsonl',
        qrels='{root}/msmarco/qrels/qrels.dev.tsv',
        corpus_vec='{root}/msmarco/msmarco_corpus_embed_f32.npy',
        query_vec='{root}/msmarco/msmarco_query_embed_f32.npy'),
    'scifact': dict(
        index='{root}/index_scifact',
        corpus='{root}/scifact/corpus.jsonl',
        queries='{root}/scifact/queries.jsonl',
        qrels='{root}/scifact/qrels/test.tsv',
        corpus_vec='{root}/scifact/scifact_corpus_embed_f32.npy',
        query_vec='{root}/scifact/scifact_query_embed_f32.npy'),
    'trec-covid': dict(
        index='{root}/index_trec_covid',
        corpus='{root}/trec-covid/corpus.jsonl',
        queries='{root}/trec-covid/queries.jsonl',
        qrels='{root}/trec-covid/qrels/test.tsv',
        corpus_vec='{root}/trec-covid/trec_corpus_embed_f32.npy',
        query_vec='{root}/trec-covid/trec_query_embed_f32.npy'),
}

ALPHAS = [round(a * 0.1, 1) for a in range(11)]
RRF_K = 60


# ==========================================
# LOADING
# ==========================================

def dataset_paths(root, name):
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r}, add it to DATASETS (known: {', '.join(DATASETS)})")
    return {key: path.format(root=root) for key, path in DATASETS[name].items()}


def load_queries(path):
    """[(qid, text)] in file order, same ID fallbacks as bins.LoadQueries."""
    queries = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            qid = str(q.get('_id') or q.get('id') or q.get('query_id') or '')
            if qid:
                queries.append((qid, q.get('text', '')))
    return queries


def load_qrels(path):
    """qid -> set of relevant docids (3- or 4-column, header skipped, rel > 0)."""
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
            elif len(parts) == 3:
                qid, docid, rel = parts
            else:
                continue
            try:
                if int(float(rel)) > 0:
                    qrels[qid].add(docid)
            except ValueError:   # header row
                continue
    return qrels


def load_corpus_ids(corpus_path, corpus_vec_path):
    """Doc ID of each embedding row: <npy>.ids if it exists, else corpus.jsonl order."""
    ids_path = corpus_vec_path + '.ids'
    if os.path.exists(ids_path):
        with open(ids_path) as f:
            return [line.strip() for line in f if line.strip()]
    ids = []
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ids.append(str(json.loads(line)['_id']))
    return ids


def load_corpus_texts(corpus_path):
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                yield str(d['_id']), f"{d.get('title', '')} {d.get('text') or d.get('abstract', '')}"


def normalise_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


# ==========================================
# RETRIEVERS: each returns {qid: [(docid, score), ...]} best first
# ==========================================

def bm25_pyserini(index_path, queries, depth, threads):
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(index_path)
    qids = [q for q, _ in queries]
    texts = [t for _, t in queries]
    hits = searcher.batch_search(texts, qids, k=depth, threads=threads)
    return {qid: [(h.docid, float(h.score)) for h in hits.get(qid, [])] for qid in qids}


TOKEN = re.compile(r'\w+')


def bm25_rank_bm25(corpus_path, queries, depth):
    """In-memory BM25, fine for SciFact-sized corpora, far too slow for MS MARCO."""
    from rank_bm25 import BM25Okapi
    ids, docs = [], []
    for docid, text in load_corpus_texts(corpus_path):
        ids.append(docid)
        docs.append(TOKEN.findall(text.lower()))
    if len(ids) > 1_000_000:
        log.warning("rank_bm25 on %d docs will be very slow, build a Lucene index instead", len(ids))
    bm25 = BM25Okapi(docs)
    out = {}
    for qid, text in queries:
        scores = bm25.get_scores(TOKEN.findall(text.lower()))
        top = np.argsort(-scores)[:depth]
        out[qid] = [(ids[i], float(scores[i])) for i in top]
    return out


def run_bm25(paths, queries, depth, backend, threads):
    if backend in ('auto', 'pyserini') and os.path.isdir(paths['index']):
        try:
            return bm25_pyserini(paths['index'], queries, depth, threads), 'pyserini'
        except Exception as err:   # no Java, no pyserini, ...
            if backend == 'pyserini':
                raise
            log.warning("pyserini failed (%s), falling back to rank_bm25", err)
    elif backend == 'pyserini':
        raise SystemExit(f"no Lucene index at {paths['index']}")
    return bm25_rank_bm25(paths['corpus'], queries, depth), 'rank_bm25'


def dense_search(doc_vecs, query_vecs, depth, doc_chunk=500_000, query_batch=1024):
    """
    Exact top-<depth> cosine search, chunked so MS MARCO (8.8M x 192) never needs
    more than one doc chunk and one query batch in memory at a time. Returns
    (indices, scores), both (num_queries, depth), best first.
    """
    n_docs = doc_vecs.shape[0]
    depth = min(depth, n_docs)
    queries = normalise_rows(query_vecs)
    best_idx = np.zeros((len(queries), depth), dtype=np.int64)
    best_score = np.full((len(queries), depth), -np.inf, dtype=np.float32)

    for start in range(0, n_docs, doc_chunk):
        chunk = normalise_rows(doc_vecs[start:start + doc_chunk])
        for q0 in range(0, len(queries), query_batch):
            q1 = min(q0 + query_batch, len(queries))
            scores = queries[q0:q1] @ chunk.T
            d = min(depth, scores.shape[1])
            top = np.argpartition(-scores, d - 1, axis=1)[:, :d]
            merged_idx = np.concatenate([best_idx[q0:q1], top + start], axis=1)
            merged_score = np.concatenate([best_score[q0:q1], np.take_along_axis(scores, top, axis=1)], axis=1)
            keep = np.argpartition(-merged_score, depth - 1, axis=1)[:, :depth]
            best_idx[q0:q1] = np.take_along_axis(merged_idx, keep, axis=1)
            best_score[q0:q1] = np.take_along_axis(merged_score, keep, axis=1)
        log.info("dense: %d / %d docs", min(start + doc_chunk, n_docs), n_docs)

    order = np.argsort(-best_score, axis=1)
    return np.take_along_axis(best_idx, order, axis=1), np.take_along_axis(best_score, order, axis=1)


# ==========================================
# FUSION
# ==========================================

def minmax(scores):
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def fusion_candidates(bm25_list, dense_list, cosines):
    """
    Union of both lists with a BM25 and a dense score for every doc. cosines(docids)
    gives the exact dense scores; BM25 falls back to the list's lowest score.
    """
    bm25_scores = dict(bm25_list)
    floor = min(bm25_scores.values()) if bm25_scores else 0.0
    docs = list(dict.fromkeys([d for d, _ in bm25_list] + [d for d, _ in dense_list]))
    bm = minmax([bm25_scores.get(d, floor) for d in docs]) if docs else []
    de = minmax(list(cosines(docs))) if docs else []
    return docs, bm, de


def fuse(docs, bm, de, alpha, k):
    order = sorted(range(len(docs)), key=lambda i: -(alpha * bm[i] + (1 - alpha) * de[i]))
    return [docs[i] for i in order[:k]]


def rrf(bm25_list, dense_list, k):
    score = defaultdict(float)
    for ranked in (bm25_list, dense_list):
        for rank, (docid, _) in enumerate(ranked, start=1):
            score[docid] += 1.0 / (RRF_K + rank)
    return sorted(score, key=lambda d: -score[d])[:k]


# ==========================================
# METRICS (same as calcMRR / calcRecall in main.go)
# ==========================================

def evaluate(results, qrels, qids, k):
    mrr = recall = 0.0
    n = 0
    for qid in qids:
        relevant = qrels.get(qid)
        if not relevant:
            continue
        n += 1
        ranked = list(dict.fromkeys(results.get(qid, [])[:k]))
        for rank, docid in enumerate(ranked, start=1):
            if docid in relevant:
                mrr += 1.0 / rank
                break
        recall += len(relevant.intersection(ranked)) / len(relevant)
    if n == 0:
        return {'MRR': math.nan, 'Recall': math.nan, 'NumQueries': 0}
    return {'MRR': round(mrr / n, 4), 'Recall': round(recall / n, 4), 'NumQueries': n}


# ==========================================
# MAIN
# ==========================================

def run_dataset(args, name):
    paths = dataset_paths(args.dataset_root, name)
    queries = load_queries(paths['queries'])
    qrels = load_qrels(paths['qrels'])
    query_rows = {qid: i for i, (qid, _) in enumerate(queries)}   # row in the query npy

    if args.qids_file:
        with open(args.qids_file) as f:
            keep = {line.strip() for line in f if line.strip()}
        queries = [q for q in queries if q[0] in keep]
    if args.max_queries:
        queries = queries[:args.max_queries]
    queries = [q for q in queries if qrels.get(q[0])]   # only queries that can be scored
    qids = [q for q, _ in queries]
    log.info("%s: %d queries with qrels", name, len(qids))

    depth = max([args.depth, max(args.k)] + args.rerank_depth)

    t = time.time()
    bm25, backend = run_bm25(paths, queries, depth, args.bm25, args.threads)
    log.info("%s: BM25 (%s) took %.1fs", name, backend, time.time() - t)

    t = time.time()
    doc_vecs = np.load(paths['corpus_vec'], mmap_mode='r')
    query_vecs = np.load(paths['query_vec'], mmap_mode='r')
    doc_ids = load_corpus_ids(paths['corpus'], paths['corpus_vec'])
    if len(doc_ids) != doc_vecs.shape[0]:
        raise SystemExit(f"{name}: {len(doc_ids)} doc IDs but {doc_vecs.shape[0]} embedding rows")
    doc_row = {d: i for i, d in enumerate(doc_ids)}
    q_vecs = normalise_rows(query_vecs[[query_rows[q] for q in qids]])
    idx, scores = dense_search(doc_vecs, q_vecs, depth)
    dense = {qid: [(doc_ids[j], float(s)) for j, s in zip(idx[i], scores[i])] for i, qid in enumerate(qids)}
    log.info("%s: dense search took %.1fs", name, time.time() - t)

    def cosine_fn(qi):
        """docids -> exact cosine with query qi (-1 for a doc with no embedding)."""
        qv = q_vecs[qi]
        def cosines(docids):
            rows = [doc_row.get(d, -1) for d in docids]
            found = [r for r in rows if r >= 0]
            sims = normalise_rows(doc_vecs[np.sort(found)]) @ qv if found else np.zeros(0)
            by_row = dict(zip(np.sort(found).tolist(), sims.tolist()))
            return [by_row[r] if r >= 0 else -1.0 for r in rows]
        return cosines

    # Fusion candidates are the same for every alpha and k, so build them once.
    bm25_at = {qid: bm25[qid][:args.depth] for qid in qids}
    dense_at = {qid: dense[qid][:args.depth] for qid in qids}
    candidates = {qid: fusion_candidates(bm25_at[qid], dense_at[qid], cosine_fn(i))
                  for i, qid in enumerate(qids)}

    outputs = []
    for k in args.k:
        out = {
            'dataset': name, 'k': k, 'num_queries': len(qids),
            'bm25_backend': backend, 'fusion_depth': args.depth,
            'bm25': evaluate({q: [d for d, _ in bm25[q]] for q in qids}, qrels, qids, k),
            'dense': evaluate({q: [d for d, _ in dense[q]] for q in qids}, qrels, qids, k),
        }

        sweep = {}
        for alpha in ALPHAS:
            fused = {q: fuse(*candidates[q], alpha, k) for q in qids}
            sweep[alpha] = evaluate(fused, qrels, qids, k)
        out['hybrid'] = dict(sweep[args.alpha] if args.alpha in sweep else
                             evaluate({q: fuse(*candidates[q], args.alpha, k) for q in qids}, qrels, qids, k),
                             alpha=args.alpha)
        best_alpha = max(sweep, key=lambda a: (sweep[a]['MRR'], sweep[a]['Recall']))
        out['hybrid_best'] = dict(sweep[best_alpha], alpha=best_alpha)
        out['hybrid_sweep'] = {str(a): {'MRR': m['MRR'], 'Recall': m['Recall']} for a, m in sweep.items()}
        out['hybrid_rrf'] = evaluate({q: rrf(bm25_at[q], dense_at[q], k) for q in qids}, qrels, qids, k)

        for n in args.rerank_depth:
            reranked = {}
            for i, q in enumerate(qids):
                pool = [d for d, _ in bm25[q][:n]]
                sims = cosine_fn(i)(pool)
                reranked[q] = [pool[j] for j in np.argsort(-np.asarray(sims), kind='stable')]
            out[f'bm25_rerank_{n}'] = evaluate(reranked, qrels, qids, k)

        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f'{name}_k{k}.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        log.info("wrote %s", path)
        outputs.append(out)
    return outputs


def print_summary(outputs):
    rows = [key for key in outputs[0] if isinstance(outputs[0][key], dict) and 'MRR' in outputs[0][key]]
    print(f"\n{'dataset':<12}{'k':>4}  {'baseline':<18}{'MRR':>8}{'Recall':>8}")
    for out in outputs:
        for key in rows:
            m = out[key]
            extra = f"  alpha={m['alpha']}" if 'alpha' in m else ''
            print(f"{out['dataset']:<12}{out['k']:>4}  {key:<18}{m['MRR']:>8.4f}{m['Recall']:>8.4f}{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', default='../datasets', help='same as -dataset in main.go')
    ap.add_argument('--datasets', nargs='+', default=['msmarco', 'scifact'])
    ap.add_argument('--k', nargs='+', type=int, default=[10], help='cut-off(s) for MRR@k / Recall@k')
    ap.add_argument('--depth', type=int, default=1000, help='per-list depth used for hybrid fusion')
    ap.add_argument('--alpha', type=float, default=0.5, help='BM25 weight in the reported hybrid')
    ap.add_argument('--rerank-depth', nargs='*', type=int, default=[100, 1000],
                    help='BM25 pool sizes for the bm25_rerank_N baselines')
    ap.add_argument('--bm25', choices=['auto', 'pyserini', 'rank_bm25'], default='auto')
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--qids-file', help='only score these query IDs (one per line), e.g. the tree subset')
    ap.add_argument('--max-queries', type=int, help='only the first N queries of the file')
    ap.add_argument('--out', default='baselines', help='output folder for the JSONs')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    outputs = []
    for name in args.datasets:
        outputs.extend(run_dataset(args, name))
    if outputs:
        print_summary(outputs)


if __name__ == '__main__':
    main()
