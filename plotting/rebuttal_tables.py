"""
The five representative configs of every method on every dataset, picked with
select_configs.py's procedure (knee, 90% / 95% of the best effectiveness, best
MRR, best Recall) once per cost axis: client storage, WAN latency, LAN latency.
Written as one Markdown file (to hand to whoever fills in the rebuttal) and one
CSV with the same rows.

    python rebuttal_tables.py                      # every method, g.ALL_DATASETS, k = g.K_MAIN
    python rebuttal_tables.py -d msmarco nq -n 3
    python rebuttal_tables.py --costs wan_time

Writes figures/rebuttal_best<N>_k<K>.md and .csv. Runs come from load_results(),
so a config's real run is used when it exists and its simulated run otherwise
(column "source"); simulated runs have no server computation time ("n/a").
"""

import argparse
import csv
import logging
import math
import os

import globals as g
import plot_utils as pu
import select_configs as sc

log = logging.getLogger('rebuttal')

COST_KEYS = ['client_storage_mb', 'wan_time', 'lan_time']   # one knee / selection per cost axis
N_BEST = 5
METHODS = ['bins', 'tree', 'pacmann', 'tiptoe', 'coeus']

COST_TITLES = {'client_storage_mb': 'client storage', 'wan_time': 'WAN latency', 'lan_time': 'LAN latency'}

# (header, run key, format); everything is per query except storage / DB size
COLUMNS = [
    ('MRR@{k}',              'mrr',               '.4f'),
    ('Recall@{k}',           'recall',            '.4f'),
    ('WAN (s/query)',        'wan_time',          '.3f'),
    ('LAN (s/query)',        'lan_time',          '.3f'),
    ('Client storage (MB)',  'client_storage_mb', '.1f'),
    ('Comm (KB/query)',      'comm_kb',           '.1f'),
    ('Server DB (MB)',       'db_size_mb',        '.1f'),
    ('Server compute (s/query)', 'total_time',    '.3f'),
    ('PIR rounds',           'pir_rounds',        '.0f'),
]

PICK_NAMES = {'knee': 'knee', 'E>=90%': '90% of best E', 'E>=95%': '95% of best E',
              'best mrr': 'best MRR', 'best recall': 'best Recall', 'envelope': 'envelope', '(all)': 'all runs'}


def _cell(run, key, fmt):
    value = run.get(key, math.nan)
    if not pu.is_number(value):
        return 'n/a'
    return format(value, fmt)


def select_rows(nested_data, cost_key, datasets, methods, k, n):
    """[{dataset, method, pick, config, <column values>, source, folder}, ...] for one cost axis."""
    rows = []
    for dataset in datasets:
        for method in methods:
            runs = nested_data.get(method, {}).get(dataset, [])
            runs = [r for r in runs if r['k'] == k]
            if not runs:
                continue
            picks = sc.select_with_reasons(runs, n=n, cost_key=cost_key, name=f'{method}/{dataset}/{cost_key}')
            for i, (reason, run, e) in enumerate(picks, 1):
                row = {'cost_axis': cost_key, 'dataset': dataset, 'method': method, 'rank': i,
                       'pick': PICK_NAMES.get(reason, reason), 'effectiveness': round(e, 3),
                       'config': run['config'], 'source': run.get('source', 'real'), 'folder': run['folder']}
                for header, key, fmt in COLUMNS:
                    row[key] = _cell(run, key, fmt)
                rows.append(row)
    return rows


def best_scores(nested_data, datasets, methods, k):
    """Markdown: each method's best Recall@k and best MRR@k per dataset, then the non-private baselines
    (baselines.py JSONs, see rebuttal_numbers.baselines_json) when they exist."""
    import rebuttal_numbers as rn
    out = ['## Best scores per method, and non-private baselines', '',
           '| Dataset | Method | Best Recall@{k} | config (WAN s/query) | Best MRR@{k} | config (WAN s/query) |'.format(k=k),
           '|---|---|---|---|---|---|']
    missing = []
    for d in datasets:
        for m in methods:
            rs = [r for r in nested_data.get(m, {}).get(d, []) if r['k'] == k]
            br = max((r for r in rs if pu.is_number(r.get('recall'))), key=lambda r: r['recall'], default=None)
            bm = max((r for r in rs if pu.is_number(r.get('mrr'))), key=lambda r: r['mrr'], default=None)
            if br is None and bm is None:
                continue
            cell = lambda r, key: ('n/a', 'n/a') if r is None else (f"{r[key]:.4f}", f"`{r['config']}` ({_cell(r, 'wan_time', '.3f')})")
            (rv, rc), (mv, mc) = cell(br, 'recall'), cell(bm, 'mrr')
            out.append(f"| {g.DATASET_LABELS.get(d, d)} | {g.METHOD_LABELS.get(m, m)} | {rv} | {rc} | {mv} | {mc} |")
        js, _ = rn.baselines_json(d)
        if js:
            for key in ('bm25', 'dense', 'hybrid', 'hybrid_best', 'hybrid_rrf') + tuple(x for x in js if x.startswith('bm25_rerank_')):
                v = js.get(key)
                if isinstance(v, dict) and 'MRR' in v:
                    out.append(f"| {g.DATASET_LABELS.get(d, d)} | non-private: {key} | {v['Recall']:.4f} | - | {v['MRR']:.4f} | - |")
        else:
            missing.append(d)
    if missing:
        out += ['', 'Non-private baselines (BM25 / dense / hybrid) not found for: ' + ', '.join(missing) +
                f' (baselines.py writes <dataset>_k{k}.json; searched {", ".join(g.BASELINES_DIRS)}).']
    return '\n'.join(out) + '\n'


def _markdown(all_rows, k, n, datasets):
    headers = ['Method', 'Pick', 'Config'] + [h.format(k=k) for h, _, _ in COLUMNS] + ['Source', 'Folder']
    out = [f'# Best {n} configs per method (k={k})', '',
           'For each cost axis (client storage, WAN latency, LAN latency), each method\'s sweep is reduced to '
           f'{n} configs with the paper\'s selection procedure (plotting/select_configs.py): effectiveness '
           'E = mean of min-max normalised MRR and Recall; Pareto frontier on (MRR, Recall, cost); knee of the '
           '(log cost, E) envelope; then the cheapest frontier config at 90% and 95% of the best E, the best MRR '
           'and the best Recall (topped up from the envelope).', '',
           '- Latency, communication and server compute are per query; storage and DB size are totals.',
           '- Source "simulated": PIR is simulated (same answers, bytes, LAN/WAN, client storage as a real run), '
           'so server compute is not measured ("n/a"). "real": a full run.',
           '- Tiptoe and Coeus report MRR before the local rerank (the list they return).',
           '- Folder: the results folder, for `python load_results.py <folder>`.', '']
    for cost_key in COST_KEYS:
        rows = [r for r in all_rows if r['cost_axis'] == cost_key]
        if not rows:
            continue
        out += [f'## Selected on {COST_TITLES.get(cost_key, cost_key)}', '']
        for dataset in datasets:
            drows = [r for r in rows if r['dataset'] == dataset]
            if not drows:
                continue
            out += [f'### {g.DATASET_LABELS.get(dataset, dataset)}', '',
                    '| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
            for r in drows:
                cells = [g.METHOD_LABELS.get(r['method'], r['method']), r['pick'], f"`{r['config']}`"] + \
                        [r[key] for _, key, _ in COLUMNS] + [r['source'], f"`{r['folder']}`"]
                out.append('| ' + ' | '.join(cells) + ' |')
            out.append('')
    return '\n'.join(out) + '\n'


def make_tables(nested_data, datasets=None, methods=None, k=g.K_MAIN, n=N_BEST, cost_keys=None, write=True):
    datasets = datasets or g.ALL_DATASETS
    methods = methods or METHODS
    all_rows = []
    for cost_key in cost_keys or COST_KEYS:
        all_rows += select_rows(nested_data, cost_key, datasets, methods, k, n)
    if not all_rows:
        log.warning("no runs to select from (k=%s, datasets %s)", k, datasets)
        return []
    have = sorted({(r['dataset'], r['method']) for r in all_rows})
    missing = [(d, m) for d in datasets for m in methods if (d, m) not in have]
    if missing:
        log.info("no runs at k=%s for: %s", k, ', '.join(f'{m}/{d}' for d, m in missing))
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'rebuttal_best{n}_k{k}')
        with open(base + '.md', 'w') as f:
            f.write(_markdown(all_rows, k, n, datasets) + '\n' + best_scores(nested_data, datasets, methods, k))
        with open(base + '.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"rebuttal tables: {len(all_rows)} rows ({len(have)} method/dataset pairs) -> {base}.md / .csv")
    return all_rows


if __name__ == '__main__':
    import load_results

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--datasets', nargs='+', default=g.ALL_DATASETS)
    ap.add_argument('-m', '--methods', nargs='+', default=METHODS)
    ap.add_argument('-k', type=int, default=g.K_MAIN)
    ap.add_argument('-n', type=int, default=N_BEST)
    ap.add_argument('--costs', nargs='+', default=COST_KEYS)
    args = ap.parse_args()
    g.setup_logging()
    make_tables(load_results.load_results(), args.datasets, args.methods, args.k, args.n, args.costs)
