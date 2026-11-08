"""
Fixed-recall comparison: pick one recall level that every method reaches and
compare what each method pays to get there.

    python fixed_recall.py                  # k = g.K_MAIN, cost = wan_time
    python fixed_recall.py --k 100 --cost comm_kb
    python fixed_recall.py --dp 2           # only try 2 decimal places
    python fixed_recall.py --methods bins tree pacmann coeus --datasets msmarco nq

Writes figures/fixed_recall_k<K>.json and prints a table per dataset.

THREE LEVELS PER DATASET
------------------------
Recall is rounded to DP decimal places (3, falling back to 2 if no value is
shared at 3dp). For each method the "levels" are the rounded recalls of its
configs.

  lowest_common    the lowest rounded recall that every method has a config
                   at exactly. This is the literal "same recall" comparison.
  highest_common   the highest such value, i.e. the fairest exact match.
  reachable        floor(min over methods of max recall) at DP: the highest
                   recall every method can reach. Each method then uses its
                   cheapest config with recall >= that level, so methods may
                   overshoot it slightly. This is the usual "cost at fixed
                   quality" comparison and always exists.

If several configs of a method qualify, the cheapest on COST_KEY is used, and
its folder is printed so `python load_results.py <folder>` shows the full run.
"""

import argparse
import json
import logging
import math
import os

import globals as g
import load_results
import plot_utils as pu

log = logging.getLogger('fixed_recall')

RECALL_KEY = 'recall'
COST_KEY = 'wan_time'
DECIMALS = [3, 2]   # tried in order, the first with a common level wins
METHODS = ['bins', 'tree', 'pacmann']   # only these are compared, other methods would lower the shared level
REPORT_KEYS = ['recall', 'mrr', 'wan_time', 'lan_time', 'total_time', 'comm_kb',
               'client_storage_mb', 'db_size_mb', 'pir_rounds', 'maintenance_time']


def _level(value, dp):
    """Recall as an integer number of 10^-dp steps, so set comparisons are exact."""
    return int(round(value * 10 ** dp))


def _usable(runs, cost_key):
    return [r for r in runs if pu.is_number(r.get(RECALL_KEY)) and pu.is_number(r.get(cost_key))]


def _cheapest(runs, cost_key):
    return min(runs, key=lambda r: (r[cost_key], -r[RECALL_KEY])) if runs else None


def _row(run):
    row = {'folder': run['folder'], 'config': run['config']}
    for key in REPORT_KEYS:
        value = run.get(key, math.nan)
        row[key] = value if pu.is_number(value) else None
    return row


def _pick_exact(runs_by_method, level, dp, cost_key):
    return {m: _row(_cheapest([r for r in runs if _level(r[RECALL_KEY], dp) == level], cost_key))
            for m, runs in runs_by_method.items()}


def _pick_at_least(runs_by_method, target, cost_key):
    return {m: _row(_cheapest([r for r in runs if r[RECALL_KEY] >= target - 1e-9], cost_key))
            for m, runs in runs_by_method.items()}


def fixed_recall_dataset(nested_data, dataset, k, cost_key=COST_KEY, decimals=DECIMALS, methods=None):
    runs_by_method = {}
    for method in methods or METHODS:
        runs = _usable(pu.get_runs(nested_data, method, dataset, k=k), cost_key)
        if runs:
            runs_by_method[method] = runs
    if len(runs_by_method) < 2:
        log.warning("%s: fewer than two methods with runs at k=%s, skipping", dataset, k)
        return None

    out = {'dataset': dataset, 'k': k, 'cost_key': cost_key, 'methods': list(runs_by_method),
           'max_recall': {m: max(r[RECALL_KEY] for r in runs) for m, runs in runs_by_method.items()}}

    # Exact match: rounded recall shared by every method.
    for dp in decimals:
        sets = [{_level(r[RECALL_KEY], dp) for r in runs} for runs in runs_by_method.values()]
        common = sorted(set.intersection(*sets))
        if common:
            break
    out['dp'] = dp
    out['common_levels'] = [c / 10 ** dp for c in common]
    if common:
        out['lowest_common'] = {'recall': common[0] / 10 ** dp,
                                'methods': _pick_exact(runs_by_method, common[0], dp, cost_key)}
        out['highest_common'] = {'recall': common[-1] / 10 ** dp,
                                 'methods': _pick_exact(runs_by_method, common[-1], dp, cost_key)}
    else:
        log.warning("%s: no recall value shared by every method at %s dp", dataset, decimals)

    # At-least match: the highest level every method can reach.
    dp_reach = decimals[0]
    target = math.floor(min(out['max_recall'].values()) * 10 ** dp_reach + 1e-9) / 10 ** dp_reach
    out['reachable'] = {'recall': target, 'dp': dp_reach,
                        'methods': _pick_at_least(runs_by_method, target, cost_key)}
    return out


def fixed_recall(nested_data, k=g.K_MAIN, cost_key=COST_KEY, decimals=DECIMALS, write=True,
                 methods=None, datasets=None):
    results = {}
    for dataset in datasets or g.DATASETS:
        res = fixed_recall_dataset(nested_data, dataset, k, cost_key, decimals, methods)
        if res:
            results[dataset] = res
            print_table(res)
    if write and results:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        path = os.path.join(g.FIGURE_DIR, f'fixed_recall_k{k}.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        log.info("saved %s", path)
    return results


def _fmt(value):
    if value is None:
        return '-'
    if isinstance(value, float) and not value.is_integer():
        return f'{value:.4g}' if abs(value) < 1000 else f'{value:.0f}'
    return str(int(value))


def print_table(res):
    cols = ['recall', 'mrr', res['cost_key']] + [c for c in ('comm_kb', 'client_storage_mb', 'pir_rounds')
                                                  if c != res['cost_key']]
    print(f"\n=== {res['dataset']}, k={res['k']} ===")
    print("max recall: " + ', '.join(f"{g.METHOD_LABELS.get(m, m)} {v:.4f}" for m, v in res['max_recall'].items()))
    shown = res['common_levels'] if len(res['common_levels']) <= 12 else \
        res['common_levels'][:6] + ['...'] + res['common_levels'][-6:]
    print(f"recalls shared by every method at {res['dp']}dp: {shown or 'none'}")
    for name in ('lowest_common', 'highest_common', 'reachable'):
        if name not in res:
            continue
        block = res[name]
        print(f"\n  {name} (recall {'>=' if name == 'reachable' else '='} {block['recall']})")
        print('  ' + f"{'method':<14}" + ''.join(f'{c:>18}' for c in cols) + '  folder')
        for method, row in block['methods'].items():
            print('  ' + f'{g.METHOD_LABELS.get(method, method):<14}' +
                  ''.join(f'{_fmt(row[c]):>18}' for c in cols) + f"  {row['folder']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, default=g.K_MAIN)
    ap.add_argument('--cost', default=COST_KEY, help='run key used to pick the cheapest config')
    ap.add_argument('--dp', type=int, nargs='+', default=DECIMALS, help='decimal places to try, in order')
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--datasets', nargs='+', default=g.DATASETS)
    args = ap.parse_args()

    g.setup_logging()
    nested = load_results.load_results(g.RESULTS_DIR)
    fixed_recall(nested, k=args.k, cost_key=args.cost, decimals=args.dp,
                 methods=args.methods, datasets=args.datasets)
