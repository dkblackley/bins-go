"""
The BEST_N configs per method and dataset, picked by select_configs.py exactly as the
plots pick them, with their stats side by side in the terminal:

    python best_configs.py                    # g.DATASETS (DATASETS=new / all / nq,fever ...)
    DATASETS=new python best_configs.py
    python best_configs.py -d nq -d fever -n 8 --cost comm_kb

and the same rows written to figures/best_configs.csv (CSV_NAME) for a spreadsheet or a
paper table. The '#' column is the pick order (C1 = the knee) and 'why' the reason it was
picked (see select_configs.py). A '-' is a value the run doesn't have (simulated runs have
no server timing, only the RAG-evaluated runs have faithfulness / relevancy).
"""

import csv
import math
import os

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

BEST_N = sc.N_BEST
K = g.K_MAIN
CSV_NAME = 'best_configs.csv'   # in g.FIGURE_DIR; None for no file

# (run key, column header, formatter): the columns, left to right after '#', 'why' and 'config'
BYTE_UNITS = ['KB', 'MB', 'GB', 'TB']


def fmt_bytes_kb(kb):
    if not pu.is_number(kb):
        return '-'
    for unit in BYTE_UNITS:
        if abs(kb) < 1024 or unit == BYTE_UNITS[-1]:
            return f'{kb:.3g} {unit}'
        kb /= 1024


def fmt_mb(mb):
    return fmt_bytes_kb(mb * 1024) if pu.is_number(mb) else '-'


def fmt_seconds(s):
    if not pu.is_number(s):
        return '-'
    if s >= 3600:
        return f'{s / 3600:.3g} h'
    if s >= 60:
        return f'{s / 60:.3g} min'
    if s >= 1:
        return f'{s:.3g} s'
    return f'{s * 1000:.3g} ms'


def fmt_score(v):
    return f'{v:.4f}' if pu.is_number(v) else '-'


def fmt_int(v):
    return f'{v:g}' if pu.is_number(v) else '-'


COLUMNS = [
    ('mrr',               'MRR@10',      fmt_score),
    ('recall',            'Recall@10',   fmt_score),
    ('client_storage_mb', 'client',      fmt_mb),        # client hint / storage
    ('comm_kb',           'sent/q',      fmt_bytes_kb),  # bytes up + down per query
    ('wan_time',          'WAN/q',       fmt_seconds),
    ('lan_time',          'LAN/q',       fmt_seconds),
    ('total_time',        'compute/q',   fmt_seconds),
    ('db_size_mb',        'server DB',   fmt_mb),
    ('maintenance_time',  'preproc',     fmt_seconds),
    ('pir_rounds',        'rounds',      fmt_int),
    ('answer_relevancy',  'relev.',      fmt_score),
    ('faithfulness',      'faith.',      fmt_score),
]
DROP_EMPTY_COLUMNS = True   # leave out a column no picked run of the dataset has (e.g. compute/q on simulated runs)


def picks_for(nested_data, dataset, method, n=BEST_N, cost_key=sc.COST_KEY):
    runs = pu.get_runs(nested_data, method, dataset, k=K)
    return sc.select_with_reasons(runs, n=n, cost_key=cost_key, name=f'{method}/{dataset}')


def print_dataset(dataset, per_method, columns):
    label = g.DATASET_LABELS.get(dataset, dataset)
    head = ['#', 'why', 'config'] + [c[1] for c in columns] + ['src']
    rows = []
    for method, picks in per_method.items():
        if not picks:
            continue
        rows.append(None)   # method header line
        rows.append(method)
        for i, (reason, run, _e) in enumerate(picks, 1):
            rows.append([str(i), reason, run['config']] + [f(run.get(k, math.nan)) for k, _, f in columns]
                        + [run.get('source', '')[:3]])
    widths = [len(h) for h in head]
    for row in rows:
        if isinstance(row, list):
            widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    line = '  '.join(h.rjust(w) if i > 2 else h.ljust(w) for i, (h, w) in enumerate(zip(head, widths)))
    print(f"\n{'=' * len(line)}\n{label}  (best {BEST_N} per method, k={K}, picked on {sc.COST_KEY})\n{'=' * len(line)}")
    print(line)
    for row in rows:
        if row is None:
            continue
        if isinstance(row, str):
            print(f"-- {g.METHOD_LABELS.get(row, row)} " + '-' * (len(line) - len(g.METHOD_LABELS.get(row, row)) - 4))
            continue
        print('  '.join(c.rjust(w) if i > 2 else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))))


def best_configs(nested_data, datasets=None, n=BEST_N, cost_key=sc.COST_KEY, csv_name=CSV_NAME):
    """Prints one table per dataset and returns {dataset: {method: [(reason, run, E), ...]}}."""
    datasets = datasets or g.DATASETS
    out, csv_rows = {}, []
    for dataset in datasets:
        per_method = {m: picks_for(nested_data, dataset, m, n, cost_key) for m in g.METHOD_ORDER}
        out[dataset] = per_method
        if not any(per_method.values()):
            print(f"\n{g.DATASET_LABELS.get(dataset, dataset)}: no runs")
            continue
        picked = [run for picks in per_method.values() for _, run, _ in picks]
        columns = [c for c in COLUMNS
                   if not DROP_EMPTY_COLUMNS or any(pu.is_number(r.get(c[0])) for r in picked)]
        print_dataset(dataset, per_method, columns)
        for method, picks in per_method.items():
            for i, (reason, run, e) in enumerate(picks, 1):
                csv_rows.append(dict(dataset=dataset, method=method, pick=i, reason=reason, config=run['config'],
                                     effectiveness=round(e, 4), source=run.get('source', ''), folder=run['folder'],
                                     **{k: run.get(k, '') for k, _, _ in COLUMNS}))
    if csv_name and csv_rows:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        path = os.path.join(g.FIGURE_DIR, csv_name)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n(best configs also in {path})")
    return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dataset', action='append', help='dataset or group (repeatable; default g.DATASETS)')
    parser.add_argument('-n', type=int, default=BEST_N, help=f'configs per method (default {BEST_N})')
    parser.add_argument('--cost', default=sc.COST_KEY, help=f'cost key the picks use (default {sc.COST_KEY})')
    args = parser.parse_args()
    g.setup_logging()
    import logging
    logging.getLogger('select').setLevel(logging.WARNING)   # the table says it all
    data = load_results.load_results()
    BEST_N = args.n
    best_configs(data, g._parse_datasets(','.join(args.dataset)) if args.dataset else None, args.n, args.cost)
