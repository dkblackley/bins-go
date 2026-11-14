"""
How the number of query terms a method uses affects MRR@k / Recall@k, as Markdown tables
(figures/query_terms_k<K>.md, + one CSV per table). No new runs: everything comes from load_results().

    python query_terms_table.py                 # g.ALL_DATASETS, k = g.K_MAIN
    python query_terms_table.py -d msmarco nq

PILLAR-Tree has a real query-term knob, q* (-q, tree/cmd/tree_main.go normalizeQueryTerms): the query is
padded / truncated to exactly q* analysed terms, keeping the FIRST q*. Table 1 is MRR / Recall vs q* for the
family (B, r, s, L, c) swept over the most q* values on each dataset, and the share of queries that q* cuts.
(rebuttal_numbers.py item E printed the MS MARCO row of this.)

PILLAR-Bin has no such knob: every analysed query term becomes one ID-DB lookup (bins/PIR_bins.go MakeIndices,
bins_old.go for one-DB). The ID-DB batch (-idBatch, 24 by default; 14 for one-DB) is NOT a term cap:
SimpleBatchPianoPIR.Query makes as many Piano queries per partition as the busiest partition needs, so extra terms
cost more sub-queries and more frequent hint refreshes, never a dropped term. Table 4 checks this on the runs:
same config, different -idBatch -> same MRR.
What IS cut in the two-DB (vec0) layout is the candidate list sent to the embedding DB: vecBatch * vecRounds
embeddings per query, the rest dropped. With -vecOrder rr (the default, and every _vb.._vr.._ib.. sweep run) the
list is rank 1 of every term's bin, then rank 2, ..., so a smaller budget makes each term lose its lowest-ranked
docs (fewer docs per term), not the last terms. Table 3 is MRR vs vecRounds at fixed bs / dpb / vecBatch.
Table 2: each dataset's best one-round (one DB: vec1 / text) and two-round (vec0 with one embedding round) configs.

PILLAR-Bin's own q* ablation (table 5): pir_app -maxTerms N keeps the first N analysed query terms, like the tree's
-q. run_bins_*.slurm add_maxterms_sweep runs three vec0 configs per dataset (embedding batch 100 x 1 / 10 / 30 rounds)
at N = 0 (every term) 1 2 3 4 6 8; results folders bins_vec0_..._ib<I>_mt<N>. load_results skips them unless
INCLUDE_MAXTERMS=1, which this script sets for itself (and keeps them out of tables 2-4).
"""

import argparse
import csv
import json
import logging
import os
import statistics
from collections import defaultdict

import globals as g
import plot_utils as pu

log = logging.getLogger('query_terms')

TREE_FAMILY = ('b', 'r', 's', 'L', 'k_candidates')
BIN_FAMILY = ('vec', 'bs', 'dpb', 'vec_batch')


def lab(d):
    return g.DATASET_LABELS.get(d, d)


def num(v, fmt='.4f'):
    return format(v, fmt) if pu.is_number(v) else 'n/a'


def md_table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)] + \
           ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]


# ------------------------------------------------------------------------------------------ query term counts

def term_counts(dataset):
    """[number of analysed terms per evaluated query], and where they came from. The tree's analysed query file
    (Lucene analyser) when there is one; else queries.jsonl restricted to the qrels' queries, through
    rebuttal_numbers.make_analyzer (pyserini if installed, else lowercase + Lucene stopwords)."""
    for p in (os.path.join(g.TREE_DATA_DIR, f'{dataset}_tree', 'queries.qrels.analyzed.json'),
              os.path.join(g.DATASETS_ROOT, dataset, 'queries.dev.small.analyzed.json')):
        if os.path.exists(p) and os.path.getsize(p) > 10:
            with open(p) as fh:
                return [len(q['terms']) for q in json.load(fh)], 'Lucene (tree query file)'
    root = os.path.join(g.DATASETS_ROOT, dataset)
    qfile = os.path.join(root, 'queries.jsonl')
    qrels = next((p for p in (os.path.join(root, 'qrels', 'test.tsv'), os.path.join(root, 'qrels', 'dev.tsv'))
                  if os.path.exists(p)), None)
    if not (os.path.exists(qfile) and qrels):
        return None, 'no local query file'
    keep = set()
    for line in open(qrels):
        parts = line.split()
        if parts and parts[0] != 'query-id':
            keep.add(parts[0])
    import rebuttal_numbers as rn
    analyze, note = rn.make_analyzer()
    counts = []
    with open(qfile) as fh:
        for line in fh:
            q = json.loads(line)
            if str(q['_id']) in keep:
                counts.append(len(analyze(q['text'])))
    return counts, 'Lucene (pyserini)' if note.startswith('Lucene') else \
        'approx.: lowercase + Lucene stopwords' + (', Porter' if 'Porter' in note else ', no stemming')


# ------------------------------------------------------------------------------------------------- tables

def tree_table(nd, datasets, k):
    """Table 1: MRR / Recall vs q* per dataset."""
    rows, out = [], []
    for d in datasets:
        fam = defaultdict(dict)
        for r in nd.get('tree', {}).get(d, []):
            if r['k'] == k and r.get('q') is not None:
                fam[tuple(r.get(x) for x in TREE_FAMILY)][r['q']] = r
        fam = {key: byq for key, byq in fam.items() if len(byq) >= 3}
        if not fam:
            continue
        key, byq = max(fam.items(), key=lambda kv: (len(kv[1]), max(x['mrr'] for x in kv[1].values())))
        counts, src = term_counts(d)
        for q in sorted(byq):
            r = byq[q]
            cut = f'{sum(1 for c in counts if c > q) / len(counts):.1%}' if counts else 'n/a'
            rows.append({'dataset': d, 'family': 'B{} r{} s{} L{} c{}'.format(*key), 'q': q, 'mrr': r['mrr'],
                         'recall': r['recall'], 'queries_truncated': cut, 'wan_time': r.get('wan_time'),
                         'comm_kb': r.get('comm_kb'), 'client_storage_mb': r.get('client_storage_mb'),
                         'source': r.get('source'), 'folder': r['folder']})
    if not rows:
        return rows, ['No PILLAR-Tree runs with 3+ q* values.']

    # wide: one row per dataset, one column per q*
    qs = sorted({r['q'] for r in rows})
    wide = []
    for d in dict.fromkeys(r['dataset'] for r in rows):
        dr = {r['q']: r for r in rows if r['dataset'] == d}
        counts, src = term_counts(d)
        stats = (f'{statistics.mean(counts):.2f}', f'{sum(1 for c in counts if c > 8) / len(counts):.1%}', src) \
            if counts else ('n/a', 'n/a', src)
        cells = [f"{dr[q]['mrr']:.3f} / {dr[q]['recall']:.3f}" if q in dr else '-' for q in qs]
        wide.append([lab(d), next(iter(dr.values()))['family']] + cells + list(stats))
    out += [f'## 1. PILLAR-Tree: MRR@{k} / Recall@{k} vs q* (number of query terms kept)', '',
            'Everything but q* fixed (the family swept over the most q* values on each dataset). The tree keeps the '
            'first q* analysed terms and pads shorter queries with dummy terms, so every query costs the same.', '']
    out += md_table(['Dataset', 'Family'] + [f'q*={q}' for q in qs] + ['mean terms', '> 8 terms', 'term counts'], wide)
    out += ['', f'Share of queries with more than q* terms (the ones q* truncates):', '']
    trunc = []
    for d in dict.fromkeys(r['dataset'] for r in rows):
        dr = {r['q']: r for r in rows if r['dataset'] == d}
        trunc.append([lab(d)] + [dr[q]['queries_truncated'] if q in dr else '-' for q in qs])
    out += md_table(['Dataset'] + [f'q*={q}' for q in qs], trunc)
    out += ['', 'Cost of q* on the same family (WAN s/query, comm KB/query):', '']
    cost = []
    for d in dict.fromkeys(r['dataset'] for r in rows):
        dr = {r['q']: r for r in rows if r['dataset'] == d}
        cost.append([lab(d)] + [f"{num(dr[q]['wan_time'], '.2f')} / {num(dr[q]['comm_kb'], '.0f')}" if q in dr else '-'
                                for q in qs])
    out += md_table(['Dataset'] + [f'q*={q}' for q in qs], cost)
    return rows, out


def _pick(rs, frac=0.9):
    """(best-MRR run, cheapest-WAN run with MRR >= frac * best)"""
    rs = [r for r in rs if pu.is_number(r.get('mrr'))]
    if not rs:
        return None, None
    best = max(rs, key=lambda r: r['mrr'])
    ok = [r for r in rs if r['mrr'] >= frac * best['mrr'] and pu.is_number(r.get('wan_time'))]
    return best, min(ok, key=lambda r: r['wan_time']) if ok else best


def bins_rounds_table(nd, datasets, k):
    """Table 2: best one-round and two-round PILLAR-Bin configs."""
    rows = []
    for d in datasets:
        rs = [r for r in nd.get('bins', {}).get(d, []) if r['k'] == k]
        groups = [('1 (one DB: vec1)', [r for r in rs if r['vec'] == 1]),
                  ('1 (one DB: text)', [r for r in rs if r['vec'] == 2]),
                  ('2 (vec0, 1 embedding round)', [r for r in rs if r['vec'] == 0 and r.get('vec_rounds') == 1]),
                  ('2 + more embedding rounds (any vec0)', [r for r in rs if r['vec'] == 0])]
        for label, sel in groups:
            best, cheap = _pick(sel)
            for pick, r in (('best MRR', best), ('cheapest >= 90% of best MRR', cheap)):
                if r is None or (pick != 'best MRR' and r is best):
                    continue
                rows.append({'dataset': d, 'pir_rounds': label, 'pick': pick, 'config': r['config'], 'mrr': r['mrr'],
                             'recall': r['recall'], 'vec_rounds': r.get('vec_rounds'), 'wan_time': r.get('wan_time'),
                             'client_storage_mb': r.get('client_storage_mb'), 'comm_kb': r.get('comm_kb'),
                             'source': r.get('source'), 'folder': r['folder']})
    out = [f'## 2. PILLAR-Bin: best configs with one PIR round (one DB) and two (ID DB + one embedding round)', '',
           '"cheapest >= 90%": the lowest-WAN config of that group within 90% of its best MRR (shown when it is a '
           'different config). Every config queries every analysed term.', '']
    out += md_table(['Dataset', 'PIR rounds', 'Pick', 'Config', f'MRR@{k}', f'Recall@{k}', 'WAN (s/query)',
                     'Client (MB)', 'Comm (KB/query)'],
                    [[lab(r['dataset']), r['pir_rounds'], r['pick'], f"`{r['config']}`", num(r['mrr']), num(r['recall']),
                      num(r['wan_time'], '.3f'), num(r['client_storage_mb'], '.0f'), num(r['comm_kb'], '.0f')]
                     for r in rows])
    return rows, out


def bins_vecrounds_table(nd, datasets, k):
    """Table 3: MRR vs embedding-DB rounds (candidate budget) at fixed bs / dpb / vecBatch."""
    rows = []
    for d in datasets:
        fam = defaultdict(dict)
        for r in nd.get('bins', {}).get(d, []):
            if r['k'] == k and r['vec'] == 0 and r.get('vec_sweep') and r.get('id_batch') == 24:
                fam[tuple(r.get(x) for x in BIN_FAMILY)][int(r['vec_rounds'])] = r
        fam = {key: v for key, v in fam.items() if len(v) >= 3}
        if not fam:
            continue
        key, byv = max(fam.items(), key=lambda kv: (len(kv[1]), max(x['mrr'] for x in kv[1].values())))
        for vr in sorted(byv):
            r = byv[vr]
            fetched = r['vec_candidates'] - r['vec_dropped'] if pu.is_number(r.get('vec_candidates')) else None
            rows.append({'dataset': d, 'family': f'bs{key[1]} dpb{key[2]} vb{int(key[3])}', 'vec_rounds': vr,
                         'pir_rounds': 1 + vr, 'mrr': r['mrr'], 'recall': r['recall'],
                         'candidates': r.get('vec_candidates'), 'fetched': fetched, 'wan_time': r.get('wan_time'),
                         'client_storage_mb': r.get('client_storage_mb'), 'source': r.get('source'), 'folder': r['folder']})
    out = [f'## 3. PILLAR-Bin (vec0): MRR@{k} vs embedding-DB rounds (how many candidates reach the dense rerank)', '',
           'Fixed bs / dpb / vecBatch / idBatch 24, the family with the most vecRounds values and the best MRR. Each round '
           'fetches vecBatch embeddings; candidates beyond vecBatch x rounds are dropped, round robin over the query\'s '
           'terms (each term keeps its top-ranked docs). PIR rounds = 1 (ID DB) + embedding rounds.', '']
    out += md_table(['Dataset', 'Family', 'Embedding rounds', 'PIR rounds', f'MRR@{k}', f'Recall@{k}',
                     'Candidates/query', 'Fetched/query', 'WAN (s/query)', 'Client (MB)'],
                    [[lab(r['dataset']), r['family'], r['vec_rounds'], r['pir_rounds'], num(r['mrr']), num(r['recall']),
                      num(r['candidates'], '.0f'), num(r['fetched'], '.0f'), num(r['wan_time'], '.3f'),
                      num(r['client_storage_mb'], '.0f')] for r in rows])
    return rows, out


def bins_idbatch_check(nd, datasets, k):
    """Table 4: does -idBatch change MRR? (it shouldn't: no term is dropped)"""
    lines = []
    for d in datasets:
        # candidates/query depends only on the bins, never on -idBatch: a different value means the bins were built
        # by another code version (e.g. before / after the stemming-index fix), so those runs are not compared
        fam, mixed = defaultdict(dict), defaultdict(set)
        for r in nd.get('bins', {}).get(d, []):
            if r['k'] == k and r['vec'] == 0 and r.get('vec_sweep') and pu.is_number(r.get('id_batch')):
                fam[(r['bs'], r['dpb'], r['vec_batch'], r['vec_rounds'], r.get('vec_candidates'))][int(r['id_batch'])] = r
                mixed[(r['bs'], r['dpb'], r['vec_batch'], r['vec_rounds'])].add(r.get('vec_candidates'))
        fam = {key: v for key, v in fam.items() if len(v) >= 2}
        if not fam:
            continue
        diffs = [max(x['mrr'] for x in v.values()) - min(x['mrr'] for x in v.values()) for v in fam.values()]
        ibs = sorted({ib for v in fam.values() for ib in v})
        lines.append([lab(d), len(fam), ', '.join(map(str, ibs)), f'{max(diffs):.4f}', f'{statistics.mean(diffs):.4f}',
                      sum(1 for c in mixed.values() if len(c) > 1)])
    if not lines:
        return ['## 4. PILLAR-Bin -idBatch check', '', 'No dataset has the same config run with 2+ -idBatch values.']
    return ['## 4. PILLAR-Bin: -idBatch does not truncate query terms', '',
            'Configs run with several ID-DB batch sizes, everything else equal (same bins: same candidates/query). A '
            'batch smaller than the query\'s term count only adds Piano sub-queries in the busiest partition; MRR does '
            'not move. "Mixed bins": configs whose idBatch runs have different candidates/query, i.e. were built by '
            'different code versions; left out of the spread.', ''] + \
        md_table(['Dataset', 'Configs', 'idBatch values', 'max MRR spread', 'mean MRR spread', 'mixed bins'], lines)


def bins_maxterms_table(mt_runs, datasets, k):
    """Table 5: MRR / Recall vs -maxTerms (the bin method's q*)."""
    rows = []
    for d in datasets:
        fam = defaultdict(dict)
        for r in mt_runs.get(d, []):
            if r['k'] == k:
                fam[(r['bs'], r['dpb'], int(r['vec_batch']), int(r['vec_rounds']))][r['max_terms']] = r
        for key in sorted(fam, key=lambda f: f[3]):
            for mt, r in sorted(fam[key].items(), key=lambda kv: kv[0] or 99):
                rows.append({'dataset': d, 'family': f'bs{key[0]} dpb{key[1]} vb{key[2]}', 'vec_rounds': key[3],
                             'pir_rounds': 1 + key[3], 'max_terms': mt, 'mrr': r['mrr'], 'recall': r['recall'],
                             'queries_truncated': r.get('queries_truncated'), 'query_terms_mean': r.get('query_terms_mean'),
                             'candidates': r.get('vec_candidates'), 'wan_time': r.get('wan_time'),
                             'comm_kb': r.get('comm_kb'), 'client_storage_mb': r.get('client_storage_mb'),
                             'source': r.get('source'), 'folder': r['folder']})
    head = [f'## 5. PILLAR-Bin: MRR@{k} / Recall@{k} vs -maxTerms (number of query terms kept)', '',
            'The bin method\'s q*: only the first N analysed query terms are looked up in the ID DB (one lookup each). '
            '"all" = every term (-maxTerms 0, the normal PILLAR-Bin), run again with the same code. Three configs per '
            'dataset (embedding batch 100 x 1 / 10 / 30 rounds), everything else fixed.', '']
    if not rows:
        return rows, head + ['No -maxTerms runs found (bins_vec0_..._mt<N>): run_bins_*.slurm add_maxterms_sweep.']
    mts = sorted({r['max_terms'] for r in rows}, key=lambda m: m or 99)
    name = lambda m: 'all' if m == 0 else f'N={m}'
    groups = list(dict.fromkeys((r['dataset'], r['family'], r['vec_rounds']) for r in rows))
    by = {(r['dataset'], r['family'], r['vec_rounds'], r['max_terms']): r for r in rows}
    wide, trunc = [], []
    for d, f, vr in groups:
        cell = lambda m: (f"{by[d, f, vr, m]['mrr']:.3f} / {by[d, f, vr, m]['recall']:.3f}" if (d, f, vr, m) in by else '-')
        wide.append([lab(d), f, f'{1 + vr} ({vr} emb.)'] + [cell(m) for m in mts])
        tr = lambda m: (f"{by[d, f, vr, m]['queries_truncated']:.1%}" if (d, f, vr, m) in by and
                        pu.is_number(by[d, f, vr, m]['queries_truncated']) else '-')
        if vr == min(v for dd, ff, v in groups if dd == d):
            mean = next((x['query_terms_mean'] for x in rows if x['dataset'] == d and pu.is_number(x['query_terms_mean'])), None)
            trunc.append([lab(d), num(mean, '.2f')] + [tr(m) for m in mts if m])
    out = head + md_table(['Dataset', 'Family', 'PIR rounds'] + [name(m) for m in mts], wide)
    out += ['', 'Share of queries -maxTerms cuts (pir_app counts them with the bins\' own analyser):', '']
    out += md_table(['Dataset', 'mean terms'] + [name(m) for m in mts if m], trunc)
    return rows, out


def write_csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def make_tables(nd, datasets=None, k=g.K_MAIN, write=True):
    datasets = datasets or g.ALL_DATASETS
    # the -maxTerms runs (only loaded with INCLUDE_MAXTERMS) are table 5's; tables 2-4 are about normal PILLAR-Bin runs
    mt_runs = {d: [r for r in rs if r.get('max_terms') is not None] for d, rs in nd.get('bins', {}).items()}
    nd = dict(nd, bins={d: [r for r in rs if r.get('max_terms') is None] for d, rs in nd.get('bins', {}).items()})
    tree_rows, t1 = tree_table(nd, datasets, k)
    rr, t2 = bins_rounds_table(nd, datasets, k)
    vr, t3 = bins_vecrounds_table(nd, datasets, k)
    t4 = bins_idbatch_check(nd, datasets, k)
    mt, t5 = bins_maxterms_table(mt_runs, datasets, k)
    text = '\n'.join([f'# Query terms vs effectiveness (k={k})', '', __doc__.strip().split('\n\n', 2)[2], ''] +
                     t1 + [''] + t5 + [''] + t2 + [''] + t3 + [''] + t4) + '\n'
    print(text)
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'query_terms_k{k}')
        open(base + '.md', 'w').write(text)
        write_csv(base + '_tree.csv', tree_rows)
        write_csv(base + '_bins_rounds.csv', rr)
        write_csv(base + '_bins_vecrounds.csv', vr)
        write_csv(base + '_bins_maxterms.csv', mt)
        print(f'saved {base}.md (+ _tree / _bins_maxterms / _bins_rounds / _bins_vecrounds .csv)')
    return text


if __name__ == '__main__':
    import load_results

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--datasets', nargs='+', default=g.ALL_DATASETS)
    ap.add_argument('-k', type=int, default=g.K_MAIN)
    args = ap.parse_args()
    g.setup_logging()
    g.INCLUDE_MAXTERMS = True   # load the bins -maxTerms runs (table 5)
    make_tables(load_results.load_results(), args.datasets, args.k)
