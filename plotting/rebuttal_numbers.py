"""
Every number the rebuttal [FILL]s need that existing results can give (items A-K
of the agent's list), in one report: figures/rebuttal_numbers.md (+ .json).
Nothing here runs an experiment; items that would need new runs are listed as
missing with what would be needed.

    python rebuttal_numbers.py                 # everything
    python rebuttal_numbers.py --only A G I    # some items

Inputs:
  * metadata.json of every run (load_results: real run if there is one, else the
    simulated run; simulated numbers are used as they are)
  * per-query results (results.json / results_reRank.json) of the msmarco runs,
    for D and J; ragas_output_*.csv for F. Missing files are listed at the end
    with an rsync command that fetches just those.
  * g.DATASETS_ROOT/msmarco: corpus.jsonl, qrels/qrels.dev.tsv,
    queries.dev.small.analyzed.json (or g.TREE_DATA_DIR/msmarco_tree/
    queries.qrels.analyzed.json); baselines.py JSONs in g.BASELINES_DIRS (C).

Item by item:
  A  frozen values: frozen_values.py (one MRR@10 / Recall@10 target and one WAN budget
     shared by every dataset; Bin, Tree, PACMANN, and the PACMANN+Bin / PACMANN+Tree pairs)
  B  passage fetch (msmarco): PianoPIR cost of fetching the 10 final passages,
     modelled with the formulas pianopir/batch-pir.go uses (same batching as the
     tree's Stage 3), from the real passage lengths in corpus.jsonl
  C  non-private references: baselines.py output (BM25, dense, hybrid)
  D  candidate recall before the dense rerank: Bin's best config (its
     results.json keeps every candidate). The tree only saves its final 10.
  E  query terms: tree Recall@10 vs q* (other params fixed), % of queries with
     more than 8 analysed terms
  F  RAGAS: answer relevancy per method; Tree - PACMANN with a paired bootstrap
     95% CI where both have per-query CSVs
  G  scale: msmarco (8.8M) -> msmarco-v2 25M / 50M / 138M
  H  PIR hint refresh: preprocessing rounds -> queries between refreshes, and the
     preprocessing time (real runs only measure it)
  I  storage-matched points at ~2.1 GB client storage (msmarco)
  J  per-query overlap buckets (msmarco, Tree vs PACMANN)
  K  block clustering: tree runs on the collaborator's reordered blocks (older
     real tree_grid_msmarco_*_q*_k10 runs, no _c in the name) vs the same
     B/r/s/L/q on corpus-order blocks (data/msmarco_tree, the _c runs)
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

import globals as g
import load_results
import plot_utils as pu
import frozen_values

log = logging.getLogger('rebuttal_numbers')

K = g.K_MAIN
MS = 'msmarco'
SCALE = [('msmarco', 8_841_823), ('msmarco-v2-25m', 25_000_000), ('msmarco-v2-50m', 50_000_000),
         ('msmarco-v2', 138_364_198)]
NEEDED = defaultdict(set)   # results dir -> folders whose per-query files are missing locally


def L(m):
    return g.METHOD_LABELS.get(m, m)


def runs(nd, method, dataset):
    return [r for r in nd.get(method, {}).get(dataset, []) if r['k'] == K]


def best(rs, key='recall'):
    rs = [r for r in rs if pu.is_number(r.get(key))]
    return max(rs, key=lambda r: (r[key], -r.get('wan_time', 0))) if rs else None


def f(v, fmt='.4f'):
    return 'n/a' if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)


def desc(r):
    if r is None:
        return 'none'
    return (f"{r['config']}: Recall@10 {f(r.get('recall'))}, MRR@10 {f(r.get('mrr'))}, WAN {f(r.get('wan_time'), '.3f')} s, "
            f"LAN {f(r.get('lan_time'), '.3f')} s, comm {f(r.get('comm_kb'), '.1f')} KB, "
            f"client {f(r.get('client_storage_mb'), '.1f')} MB [{r.get('source', 'real')}]")


# ------------------------------------------------------------------------------------------- per-query inputs

def run_file(run, name):
    p = os.path.join(run['results_dir'], run['folder'], name)
    if os.path.exists(p):
        return p
    NEEDED[run['results_dir']].add(run['folder'])
    return None


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


_QRELS = {}


def qrels(dataset):
    """qid -> set of relevant doc ids (relevance > 0)."""
    if dataset in _QRELS:
        return _QRELS[dataset]
    root = os.path.join(g.DATASETS_ROOT, dataset, 'qrels')
    path = next((p for p in (os.path.join(root, 'qrels.dev.tsv'), os.path.join(root, 'test.tsv')) if os.path.exists(p)), None)
    out = defaultdict(set)
    if path is None:
        log.warning("no qrels for %s under %s", dataset, root)
    else:
        for line in open(path):
            p = line.split()
            if len(p) >= 4:
                p = [p[0], p[2], p[3]]
            if len(p) == 3 and p[2].lstrip('-').isdigit() and int(p[2]) > 0:
                out[p[0]].add(p[1])
    _QRELS[dataset] = out
    return out


def analyzed_queries(dataset=MS):
    """[{'id', 'text', 'terms'}] from the tree's analysed query file (Lucene analyser: stopwords out, stemmed)."""
    for p in (os.path.join(g.TREE_DATA_DIR, f'{dataset}_tree', 'queries.qrels.analyzed.json'),
              os.path.join(g.DATASETS_ROOT, dataset, 'queries.dev.small.analyzed.json')):
        if os.path.exists(p) and os.path.getsize(p) > 10:
            return load_json(p), p
    return None, None


def rr_at_k(ranked, rel, k=K):
    for i, d in enumerate(list(dict.fromkeys(ranked))[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


# ------------------------------------------------------------------------------------------------------ items

def item_A(nd):
    _, text = frozen_values.frozen_values(nd, write=True, quiet=True)
    return text.replace('\n#', '\n##').replace('# A. ', '## A. ', 1)


def item_B(nd):
    out = ['## B. Passage fetch cost (MS MARCO): PIR-fetching the 10 final passages']
    corpus = os.path.join(g.DATASETS_ROOT, MS, 'corpus.jsonl')
    cache = os.path.join(g.FIGURE_DIR, '_msmarco_passage_bytes.json')
    lengths = None
    if os.path.exists(cache):
        lengths = load_json(cache)
    elif os.path.exists(corpus):
        lengths = []
        with open(corpus) as fh:
            for line in fh:
                d = json.loads(line)
                lengths.append(len((d.get('title', '') + ' ' + d.get('text', '')).strip().encode()))
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        json.dump(lengths, open(cache, 'w'))
    if not lengths:
        return '\n'.join(out + [f'MISSING: {corpus} (passage lengths)'])
    n = len(lengths)
    batches = Counter(int(float(r.get('stage3_batch', 0) or 0)) for r in runs(nd, 'tree', MS) if r.get('stage3_batch'))
    B = batches.most_common(1)[0][0] if batches else 32
    parts = B // 2                       # RealQueryPerPartition = 2 (pianopir/batch-pir.go)
    P = math.ceil(n / parts)
    chunk = 1
    while chunk < int(2 * math.sqrt(P)):
        chunk *= 2
    setsize = math.ceil(P / chunk)
    setsize = (setsize + 3) // 4 * 4
    # a reply is the XOR of one entry per chunk, cut to the longest of them (in 8-byte words) + a 4-byte length
    rnd = random.Random(0)
    mx = [max(lengths[rnd.randrange(n)] for _ in range(setsize)) for _ in range(300)]
    exp_max = statistics.mean(math.ceil(x / 8) * 8 for x in mx)
    per_lookup = setsize * 4 + exp_max + 4
    per_batch = B * per_lookup           # every partition answers QueryPerPartition (2) lookups, real or dummy
    wan = per_batch * 8 / 400e6 + 0.05   # netTime(bytes, 1 round, 400 Mb/s, 50 ms)
    lan = per_batch * 8 / 1000e6 + 0.005
    c = 2.8 * math.sqrt(P)
    hint_entries = 15 * c + 6 * math.sqrt(P) * math.log(P)   # pirhint/README.md
    hint_mb = parts * hint_entries * statistics.mean(lengths) / 2 ** 20
    out += [f'{n:,} passages, {sum(lengths) / 2 ** 20:,.0f} MB of UTF-8 text (mean {statistics.mean(lengths):.0f} B, '
            f'median {statistics.median(lengths):.0f} B, max {max(lengths):,} B)',
            f'model: one extra PIR round, batch of {B} lookups (the tree\'s Stage 3 batch size) over {parts} partitions '
            f'of {P:,} rows; set size {setsize}, reply = XOR of {setsize} passages cut to the longest '
            f'(expected {exp_max:.0f} B)',
            f'  per lookup: {per_lookup / 1024:.1f} KB (upload {setsize * 4 / 1024:.1f} KB of offsets, download '
            f'{exp_max / 1024:.1f} KB); per batch (fetches the 10 passages): {per_batch / 1024:,.0f} KB',
            f'  extra latency: WAN +{wan * 1000:.0f} ms (400 Mb/s, 50 ms RTT), LAN +{lan * 1000:.0f} ms (1 Gb/s, 5 ms)',
            f'  extra client storage for this DB\'s hints: roughly {hint_mb / 1024:.1f} GB (15C + 6 sqrt(P) ln P entries '
            f'per partition, mean passage size)',
            'Estimate from the cost formulas (no run); text uncompressed (bins -text compresses it).']
    return '\n'.join(out)


def baselines_json(dataset):
    for d in g.BASELINES_DIRS:
        p = os.path.join(d, f'{dataset}_k{K}.json')
        if os.path.exists(p):
            return load_json(p), p
    return None, None


def item_C(nd, datasets=('msmarco', 'scifact', 'nq', 'fever', 'msmarco-v2')):
    out = ['## C. Non-private references (Recall@10 / MRR@10)']
    for d in datasets:
        js, p = baselines_json(d)
        if not js:
            out.append(f'{g.DATASET_LABELS.get(d, d)}: MISSING (no {d}_k{K}.json from baselines.py in {", ".join(g.BASELINES_DIRS)})')
            continue
        rows = [f'{key} R {v["Recall"]:.4f} MRR {v["MRR"]:.4f}' + (f' (alpha {v["alpha"]})' if 'alpha' in v else '')
                for key, v in js.items() if isinstance(v, dict) and 'MRR' in v]
        out.append(f'{g.DATASET_LABELS.get(d, d)} ({p}): ' + '; '.join(rows))
    return '\n'.join(out)


def item_D(nd):
    out = ['## D. Lexical candidate recall (MS MARCO): queries whose relevant passage is among the candidates, before the dense rerank']
    rel = qrels(MS)
    for label, run in (('PILLAR-Bin best Recall@10', best(runs(nd, 'bins', MS))),
                       ('PILLAR-Bin best Recall@10, two-DB (vec0)', best([r for r in runs(nd, 'bins', MS) if r.get('vec') == 0]))):
        if run is None:
            out.append(f'{label}: no runs')
            continue
        p = run_file(run, 'results.json')
        if not p:
            out.append(f'{label} {run["config"]}: MISSING results.json (see the end)')
            continue
        res = load_json(p)
        qs = [q for q in res if q in rel]
        hit = sum(1 for q in qs if rel[q] & set(res[q]))
        sizes = [len(res[q]) for q in qs]
        out.append(f'{label} {run["config"]}: {hit / len(qs):.4f} of {len(qs)} queries '
                   f'(mean {statistics.mean(sizes):.1f} candidates/query; Recall@10 after rerank {run["recall"]:.4f})')
    t = best(runs(nd, 'tree', MS))
    out.append(f'PILLAR-Tree (best Recall@10: {t["config"] if t else "-"}): MISSING. Its results.json holds only the final '
               f'10, not its {(t.get("k_candidates") or "") if t else ""} Stage-2 candidates, so this needs a run that saves them.')
    return '\n'.join(out)


def item_E(nd):
    out = ['## E. Query-term ablation (MS MARCO)']
    qs, p = analyzed_queries()
    if qs:
        n = [len(q['terms']) for q in qs]
        out.append(f'{sum(1 for x in n if x > 8) / len(n):.1%} of {len(n)} queries have more than 8 terms after the '
                   f'Lucene analyser (stopwords removed); mean {statistics.mean(n):.2f}, max {max(n)} ({p})')
        for q in (2, 4, 8, 16):
            out.append(f'  > {q:>2} terms: {sum(1 for x in n if x > q) / len(n):.1%}  (q*={q} drops terms from these)')
    else:
        out.append('MISSING: the analysed query file (data/msmarco_tree/queries.qrels.analyzed.json)')
    fam = defaultdict(dict)
    for r in runs(nd, 'tree', MS):
        if r.get('q') is not None:
            fam[(r['b'], r['r'], r['s'], r['L'], r.get('k_candidates'), r.get('source'))][r['q']] = r
    rich = sorted(((k, v) for k, v in fam.items() if len(v) >= 3),
                  key=lambda kv: -max(x['recall'] for x in kv[1].values()))
    out.append('PILLAR-Tree Recall@10 by q* (everything else fixed; the 5 strongest families with >= 3 q* values):')
    for (b, r, s, Lc, c, src), byq in rich[:5]:
        out.append(f'  B{b} r{r} s{s} L{Lc}' + (f' c{c}' if c else '') + f' [{src}]: ' +
                   ', '.join(f'q{q} {byq[q]["recall"]:.4f}' for q in sorted(byq)))
    if not rich:
        out.append('  none: no tree msmarco family has 3+ q* values')
    qmax = max((q for byq in fam.values() for q in byq), default=None)
    out.append(f'Largest q* run: {qmax}. PILLAR-Bin has no q* setting (it uses every query term). '
               'All datasets: query_terms_table.py.')
    out.append('Highest-IDF vs first q* terms: MISSING, the tree keeps the first q* terms (normalizeQueryTerms); '
               'needs a run with IDF ordering.')
    return '\n'.join(out)


def bootstrap(diffs, n=10000, seed=0):
    rnd = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return statistics.mean(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def ragas_csvs(nd, method, dataset):
    out = []
    for r in runs(nd, method, dataset):
        for p in glob.glob(os.path.join(r['results_dir'], r['folder'], 'ragas_output*.csv')):
            out.append((r, p))
    return out


def item_F(nd, datasets=('msmarco', 'scifact', 'nq', 'fever', 'msmarco-v2')):
    out = ['## F. RAGAS answer relevancy']
    for d in datasets:
        line = []
        for m in ('tree', 'pacmann', 'bins'):
            vals = [r for r in runs(nd, m, d) if pu.is_number(r.get('answer_relevancy'))]
            if vals:
                line.append(f'{L(m)} {len(vals)} runs, mean relevancy {statistics.mean(r["answer_relevancy"] for r in vals):.4f} '
                            f'(best {max(r["answer_relevancy"] for r in vals):.4f})')
        out.append(f'{g.DATASET_LABELS.get(d, d)}: ' + ('; '.join(line) if line else 'no runs with RAGAS scores'))
        tcs, pcs = ragas_csvs(nd, 'tree', d), ragas_csvs(nd, 'pacmann', d)
        if tcs and pcs:
            def per_q(p):
                return {row['qid']: float(row['answer_relevancy']) for row in csv.DictReader(open(p))
                        if row.get('answer_relevancy') not in (None, '', 'nan')}
            tr = max(tcs, key=lambda x: x[0]['recall']); pr = max(pcs, key=lambda x: x[0]['recall'])
            a, b = per_q(tr[1]), per_q(pr[1])
            common = sorted(set(a) & set(b))
            if common:
                mean, lo, hi = bootstrap([a[q] - b[q] for q in common])
                out.append(f'  Tree {tr[0]["config"]} - PACMANN {pr[0]["config"]}: {mean:+.4f} (95% CI {lo:+.4f} .. {hi:+.4f}), '
                           f'N = {len(common)} paired queries')
        elif tcs:
            out.append(f'  per-query CSVs: {len(tcs)} Tree, 0 PACMANN -> no paired CI (MISSING: PACMANN RAGAS run, '
                       f'rag-eval/run_eval_pacmann.slurm); Tree N = '
                       f'{len(list(csv.DictReader(open(tcs[0][1]))))} queries per CSV')
    return '\n'.join(out)


def item_G(nd):
    out = ['## G. Scale (MS MARCO 8.8M and MS MARCO v2 subsets)',
           'subset sizes: ' + ', '.join(f'{g.DATASET_LABELS.get(d, d)} {n / 1e6:.1f}M' for d, n in SCALE) +
           ' (v2-25m and v2-50m are nested subsets of v2)']
    for m in ('tree', 'bins', 'pacmann', 'tiptoe', 'coeus'):
        lines = []
        for d, n in SCALE:
            b = best(runs(nd, m, d))
            if b:
                lines.append(f'  {g.DATASET_LABELS.get(d, d):<20} best Recall@10 {desc(b)}')
        if lines:
            out.append(f'{L(m)}:')
            out += lines
            have = [d for d, _ in SCALE if runs(nd, m, d)]
            if len(have) >= 2:
                lo, hi = have[0], have[-1]
                a = {r['config']: r for r in runs(nd, m, lo)}
                pairs = [(a[r['config']], r) for r in runs(nd, m, hi) if r['config'] in a]
                for key in ('wan_time', 'client_storage_mb'):
                    rat = [y[key] / x[key] for x, y in pairs if pu.is_number(x.get(key)) and pu.is_number(y.get(key)) and x[key] > 0]
                    if rat:
                        out.append(f'  {key} growth {lo} -> {hi} (same config, {len(rat)} configs): median x{statistics.median(rat):.2f} '
                                   f'(range x{min(rat):.2f} .. x{max(rat):.2f}); docs x{dict(SCALE)[hi] / dict(SCALE)[lo]:.1f}')
    v2 = 'msmarco-v2'
    bt = best(runs(nd, 'tree', v2))
    if bt:
        others = {m: best(runs(nd, m, v2)) for m in ('bins', 'pacmann', 'tiptoe', 'coeus')}
        others = {m: r for m, r in others.items() if r}
        if others:
            m2, r2 = max(others.items(), key=lambda kv: kv[1]['recall'])
            out.append(f'MS MARCO v2: Tree margin over the best private baseline ({L(m2)}): {bt["recall"] - r2["recall"]:+.4f} Recall@10')
        js, _ = baselines_json(v2)
        if js and 'hybrid' in js:
            out.append(f'MS MARCO v2: Tree gap to fused hybrid: {bt["recall"] - js["hybrid"]["Recall"]:+.4f} Recall@10')
        else:
            out.append('MS MARCO v2: Tree gap to fused hybrid: MISSING (item C)')
    else:
        out.append('MS MARCO v2: no tree runs yet (the tree-db build was still queued on 2026-11-10)')
    return '\n'.join(out)


def item_H(nd):
    out = ['## H. PIR hint refresh (offline phase re-runs)']
    for d in (MS,):
        rs = runs(nd, 'tree', d)
        sel = [('best Recall@10', best(rs))]
        for label, r in sel:
            if r is None:
                continue
            out.append(f'PILLAR-Tree {g.DATASET_LABELS.get(d, d)} {label} {r["config"]} ({r["num_queries"]} queries, [{r.get("source")}]):')
            for s in ('s1', 's2', 's3'):
                rounds = r.get(f'preproc_rounds_{s}')
                if not pu.is_number(rounds):
                    continue
                every = f'every {r["num_queries"] / (rounds - 1):.0f} queries' if rounds > 1 else f'none in {r["num_queries"]} queries'
                out.append(f'  Stage {s[1]}: {rounds:.0f} preprocessings -> refresh {every}; first preprocessing '
                           f'{f(r.get(f"preprocessing_time_{s}"), ".1f")} s, all {f(r.get(f"maintenance_time_{s}"), ".1f")} s')
        timed = [r for r in rs if pu.is_number(r.get('preprocessing_time'))]
        if timed:
            pts = [r['preprocessing_time'] for r in timed]
            mts = [r['maintenance_time'] for r in timed if pu.is_number(r.get('maintenance_time'))]
            out.append(f'  measured preprocessing time (real runs, {len(timed)} configs): median {statistics.median(pts):.1f} s '
                       f'(range {min(pts):.1f} .. {max(pts):.1f} s)' +
                       (f'; all refreshes of a run: median {statistics.median(mts):.1f} s' if mts else ''))
        else:
            out.append('  preprocessing time: MISSING (simulated runs do not time it; no real run of these configs)')
    out.append('PILLAR-Bin / PACMANN: their metadata.json has no preprocessing count (only the job log prints MaxQueryNum).')
    return '\n'.join(out)


def item_I(nd, target_gb=2.1, tol=0.15):
    out = [f'## I. Storage-matched points (MS MARCO, client storage {target_gb} GB +/- {tol:.0%})']
    lo, hi = target_gb * (1 - tol) * 1024, target_gb * (1 + tol) * 1024
    t = [r for r in runs(nd, 'tree', MS) if pu.is_number(r.get('client_storage_mb')) and lo <= r['client_storage_mb'] <= hi]
    out.append(f'PILLAR-Tree best Recall@10 in range ({len(t)} configs): {desc(best(t))}')
    allb = runs(nd, 'bins', MS) + [r for r in load_results.DROPPED_BINS.get(MS, []) if r['k'] == K]
    v0 = [r for r in allb if r.get('vec') == 0]
    b0 = best(v0)
    out.append(f'PILLAR-Bin two-DB (vec0) best Recall@10 ({len(v0)} configs, including any the client storage filter '
               f'drops): {desc(b0)}' + (f' = {b0["client_storage_mb"] / 1024:.2f} GB' if b0 else ''))
    near = [r for r in v0 if pu.is_number(r.get('client_storage_mb')) and lo <= r['client_storage_mb'] <= hi]
    out.append(f'PILLAR-Bin two-DB best Recall@10 within the storage range ({len(near)} configs): {desc(best(near))}')
    return '\n'.join(out)


_STOP = set('a an and are as at be but by for if in into is it no not of on or such that the their then there these '
            'they this to was will with'.split())   # Lucene's English stopword list


def make_analyzer():
    try:
        from pyserini.analysis import Analyzer, get_lucene_analyzer
        an = Analyzer(get_lucene_analyzer())
        return an.analyze, 'Lucene (pyserini), same as the query terms'
    except Exception:
        try:
            from nltk.stem import PorterStemmer
            st = PorterStemmer().stem
            note = 'fallback: lowercase + Lucene stopwords + NLTK Porter (close to, not exactly, Lucene\'s)'
        except Exception:
            st = lambda w: w
            note = 'fallback: lowercase + Lucene stopwords, NO stemming (install pyserini or nltk for a closer match)'
        return (lambda text: [st(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]), note


def item_J(nd, buckets=(0.0, 0.25, 0.5, 0.75, 1.0001)):
    out = ['## J. Per-query overlap buckets (MS MARCO, PILLAR-Tree vs PACMANN, RR@10 per query)']
    t, p = best(runs(nd, 'tree', MS)), best(runs(nd, 'pacmann', MS))
    if not (t and p):
        return '\n'.join(out + ['no tree or pacmann msmarco runs'])
    tf, pf = run_file(t, 'results_reRank.json'), run_file(p, 'results_reRank.json')
    qs, qpath = analyzed_queries()
    if not (tf and pf and qs):
        return '\n'.join(out + [f'MISSING: {"" if tf else "tree results_reRank.json "}{"" if pf else "pacmann results_reRank.json "}'
                                f'{"" if qs else "analysed queries"} (see the end)'])
    rel = qrels(MS)
    tres, pres = load_json(tf), load_json(pf)
    terms = {q['id']: set(q['terms']) for q in qs}
    qids = [q for q in terms if q in rel and q in tres and q in pres]
    need = set().union(*(rel[q] for q in qids))
    corpus = os.path.join(g.DATASETS_ROOT, MS, 'corpus.jsonl')
    if not os.path.exists(corpus):
        return '\n'.join(out + [f'MISSING: {corpus}'])
    analyze, note = make_analyzer()
    ptext = {}
    with open(corpus) as fh:
        for line in fh:
            d = json.loads(line)
            if str(d['_id']) in need:
                ptext[str(d['_id'])] = set(analyze((d.get('title', '') + ' ' + d.get('text', '')).strip()))
    stats = defaultdict(lambda: Counter())
    for q in qids:
        qt = terms[q]
        if not qt:
            continue
        ov = max((len(qt & ptext.get(dd, set())) / len(qt) for dd in rel[q]), default=0.0)
        bi = next(i for i in range(len(buckets) - 1) if ov < buckets[i + 1])
        a, b = rr_at_k(tres[q], rel[q]), rr_at_k(pres[q], rel[q])
        res = 'tree' if a > b else 'pacmann' if b > a else 'tie'
        stats[bi][res] += 1
        stats['all'][res] += 1
    tot = sum(stats['all'].values())
    out.append(f'Tree {t["config"]} vs PACMANN {p["config"]} (each method\'s best Recall@10), {tot} queries; overlap = share '
               f'of the query\'s analysed terms that appear in its best-matching relevant passage ({note})')
    out.append(f'  {"overlap":<12}{"% queries":>10}{"Tree wins":>11}{"PACMANN wins":>14}{"ties":>8}')
    for i in range(len(buckets) - 1):
        c = stats[i]; n = sum(c.values())
        if n:
            out.append(f'  {buckets[i]:.2f}-{min(buckets[i + 1], 1):.2f}   {n / tot:>9.1%}{c["tree"] / n:>11.1%}'
                       f'{c["pacmann"] / n:>14.1%}{c["tie"] / n:>8.1%}')
    c = stats['all']
    out.append(f'  overall: Tree wins {c["tree"] / tot:.1%}, loses {c["pacmann"] / tot:.1%}, ties {c["tie"] / tot:.1%} '
               f'(ties are mostly queries both miss)')
    return '\n'.join(out)


def item_K(nd):
    out = ['## K. Block clustering (MS MARCO tree): reordered blocks vs corpus-order blocks']
    rs = runs(nd, 'tree', MS)
    reord = [r for r in rs if r.get('k_candidates') is None and r.get('q') is not None]
    plain = defaultdict(list)
    for r in rs:
        if r.get('k_candidates') is not None:
            plain[(r['b'], r['r'], r['s'], r['L'], r['q'])].append(r)
    pairs = []
    for r in reord:
        cand = plain.get((r['b'], r['r'], r['s'], r['L'], r['q']))
        if cand:
            pairs.append((r, max(cand, key=lambda x: (x['k_candidates'] == 128, x['k_candidates']))))
    if not pairs:
        out.append(f'no matching pairs ({len(reord)} reordered-layout runs, {sum(map(len, plain.values()))} corpus-order runs)')
    else:
        d = [b['recall'] - a['recall'] for a, b in pairs]
        out.append(f'{len(pairs)} configs run on both: Recall@10 change from reordered to corpus order: mean {statistics.mean(d):+.4f}, '
                   f'median {statistics.median(d):+.4f} (range {min(d):+.4f} .. {max(d):+.4f})')
        for a, b in sorted(pairs, key=lambda ab: -ab[0]['recall'])[:5]:
            out.append(f'  {a["config"]}: reordered {a["recall"]:.4f} [{a.get("source")}] -> corpus order {b["recall"]:.4f} '
                       f'({b["config"]}, [{b.get("source")}])')
    out.append('Caveats: corpus order is not random (MS MARCO passages of one web page are adjacent), so this is a lower '
               'bound on the drop with random blocks; the two sets were run months apart (code changes in between). '
               'Name of the method that built data/msmarco_reordered: not in the collaborator\'s files on hopper; ask them.')
    return '\n'.join(out)


ITEMS = {'A': item_A, 'B': item_B, 'C': item_C, 'D': item_D, 'E': item_E, 'F': item_F, 'G': item_G,
         'H': item_H, 'I': item_I, 'J': item_J, 'K': item_K}


def rebuttal_numbers(nd, only=None, write=True):
    parts = []
    for key, fn in ITEMS.items():
        if only and key not in only:
            continue
        try:
            parts.append(fn(nd))
        except Exception as err:   # one broken item must not lose the others
            log.exception("item %s failed", key)
            parts.append(f'## {key}. FAILED: {err!r}')
    if NEEDED:
        lines = ['## Per-query files to download (then run again)']
        for rd, folders in NEEDED.items():
            server = '/scratch/dblackle/results_simulated/' if 'simulated' in rd else '/scratch/dblackle/results/'
            inc = ' '.join(f"--include='{fo}/' --include='{fo}/results*.json'" for fo in sorted(folders))
            lines.append(f"rsync -am {inc} --exclude='*' hopper:{server} {rd}/")
        parts.append('\n'.join(lines))
    parts.append('## L. Non-numerical\nAnonymised code link: not something these scripts can produce.')
    text = '# Rebuttal numbers (k=10; real run when there is one, else simulated)\n\n' + '\n\n'.join(parts) + '\n'
    print(text)
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        path = os.path.join(g.FIGURE_DIR, 'rebuttal_numbers.md')
        open(path, 'w').write(text)
        print(f'saved {path}')
    return text


def _add_stage3_batch(nd):
    """Stage3BatchSize isn't a loaded key; read it from the metadata of the msmarco tree runs (item B)."""
    for r in runs(nd, 'tree', MS)[:50]:
        try:
            meta = load_json(os.path.join(r['results_dir'], r['folder'], 'metadata.json'))
            r['stage3_batch'] = meta.get('Stage3BatchSize')
        except Exception:
            pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=list(ITEMS), help='only these items')
    args = ap.parse_args()
    g.setup_logging()
    data = load_results.load_results()
    _add_stage3_batch(data)
    rebuttal_numbers(data, args.only)
