"""
One frozen value across every dataset (rebuttal item A, replaces fixed_score.py's
per-dataset targets). Tiptoe and Coeus are left out.

    python frozen_values.py
    python frozen_values.py --datasets msmarco nq scifact fever
    python frozen_values.py --mrr 0.20 --recall 0.40 --wan 1.5 2.0   # your own values

Two kinds of freeze, each done for three method groups (all three; PACMANN +
PILLAR-Bin; PACMANN + PILLAR-Tree):

  Frozen score  one MRR@10 (and one Recall@10) target T for every dataset:
                T = floor(min over methods and datasets of the method's best
                score), so every method in the group reaches T on every dataset.
                Each method then uses its cheapest config (lowest WAN) with
                score >= T on each dataset.
  Frozen WAN    one WAN budget W (s/query) for every dataset: W = the smallest
                budget at which every method has a config on every dataset
                (max over methods and datasets of the method's fastest run,
                rounded up). Each method then uses its best-MRR@10 (and best-
                Recall@10) config with WAN <= W on each dataset.

A dataset joins a group only if every method of the group has runs there
(others are listed as skipped). The (method, dataset) that sets each value is
reported (the bottleneck), plus who wins on each dataset and the per-method mean
over datasets, so it is easy to see whether one method wins everywhere.

Writes figures/frozen_values_k<K>.md and .json.
"""

import argparse
import json
import logging
import math
import os

import globals as g
import load_results
import plot_utils as pu

log = logging.getLogger('frozen_values')

DATASETS = ['msmarco', 'scifact', 'nq', 'fever', 'msmarco-v2']
GROUPS = [('all three', ['bins', 'tree', 'pacmann']),
          ('PACMANN + PILLAR-Bin', ['pacmann', 'bins']),
          ('PACMANN + PILLAR-Tree', ['pacmann', 'tree'])]
SCORE_KEYS = ['mrr', 'recall']
COST_KEY = 'wan_time'
SCORE_DP = 2      # targets are floored to this many decimals
WAN_DP = 2        # budgets are rounded up to this many decimals
SCORE_LABEL = {'mrr': 'MRR@10', 'recall': 'Recall@10'}
REPORT_KEYS = ['mrr', 'recall', 'wan_time', 'lan_time', 'comm_kb', 'client_storage_mb']


def L(m):
    return g.METHOD_LABELS.get(m, m)


def D(d):
    return g.DATASET_LABELS.get(d, d)


def floor_dp(x, dp):
    return math.floor(x * 10 ** dp + 1e-9) / 10 ** dp


def ceil_dp(x, dp):
    return math.ceil(x * 10 ** dp - 1e-9) / 10 ** dp


def load_runs(nested_data, methods, datasets, k):
    """{method: {dataset: [runs with numeric MRR, Recall and WAN]}}"""
    need = SCORE_KEYS + [COST_KEY]
    return {m: {d: [r for r in nested_data.get(m, {}).get(d, [])
                    if r['k'] == k and all(pu.is_number(r.get(x)) for x in need)]
                for d in datasets} for m in methods}


def _row(run):
    row = {'config': run['config'], 'folder': run['folder'], 'source': run.get('source', 'real')}
    for key in REPORT_KEYS:
        v = run.get(key, math.nan)
        row[key] = v if pu.is_number(v) else None
    return row


def group_datasets(by, group, datasets):
    used = [d for d in datasets if all(by[m][d] for m in group)]
    skipped = {d: [m for m in group if not by[m][d]] for d in datasets if d not in used}
    return used, skipped


def frozen_score(by, group, datasets, score_key, target=None):
    """Cheapest config per (method, dataset) at one score target shared by every dataset."""
    best = {(m, d): max(r[score_key] for r in by[m][d]) for m in group for d in datasets}
    bottleneck = min(best, key=best.get)
    t = target if target is not None else floor_dp(best[bottleneck], SCORE_DP)
    picks = {}
    for m in group:
        for d in datasets:
            ok = [r for r in by[m][d] if r[score_key] >= t - 1e-9]
            picks[(m, d)] = min(ok, key=lambda r: (r[COST_KEY], -r[score_key])) if ok else None
    return {'kind': 'score', 'score': score_key, 'value': t, 'bottleneck': bottleneck,
            'bottleneck_value': best[bottleneck], 'picks': picks}


def frozen_wan(by, group, datasets, score_key, budget=None):
    """Best-score config per (method, dataset) within one WAN budget shared by every dataset."""
    fastest = {(m, d): min(r[COST_KEY] for r in by[m][d]) for m in group for d in datasets}
    bottleneck = max(fastest, key=fastest.get)
    w = budget if budget is not None else ceil_dp(fastest[bottleneck], WAN_DP)
    picks = {}
    for m in group:
        for d in datasets:
            ok = [r for r in by[m][d] if r[COST_KEY] <= w + 1e-9]
            picks[(m, d)] = max(ok, key=lambda r: (r[score_key], -r[COST_KEY])) if ok else None
    return {'kind': 'wan', 'score': score_key, 'value': w, 'bottleneck': bottleneck,
            'bottleneck_value': fastest[bottleneck], 'picks': picks}


def _f(v, fmt):
    return 'n/a' if v is None else format(v, fmt)


def _table(res, group, datasets):
    out = ['| Dataset | Method | Config | MRR@10 | Recall@10 | WAN (s/query) | LAN (s/query) | Comm (KB/query) '
           '| Client storage (MB) | Source |', '|' + '---|' * 10]
    for d in datasets:
        for m in group:
            p = res['picks'][(m, d)]
            if p is None:
                out.append(f'| {D(d)} | {L(m)} | ' + ('does not reach it' if res['kind'] == 'score' else
                                                     'nothing within the budget') + ' |' * 8)
                continue
            out.append(f"| {D(d)} | {L(m)} | `{p['config']}` | {p['mrr']:.4f} | {p['recall']:.4f} | "
                       f"{p['wan_time']:.3f} | {_f(p.get('lan_time'), '.3f')} | {_f(p.get('comm_kb'), '.1f')} | "
                       f"{_f(p.get('client_storage_mb'), '.1f')} | {p.get('source', 'real')} |")
    return out


def _summary(res, group, datasets):
    """Who wins on each dataset, and the per-method mean over datasets."""
    key, score = res['kind'], res['score']
    if key == 'score':   # same quality: lower WAN wins
        metric, better, what = COST_KEY, min, 'lowest WAN'
    else:                # same budget: higher score wins
        metric, better, what = score, max, f'highest {SCORE_LABEL[score]}'
    wins, per_d = {m: 0 for m in group}, []
    for d in datasets:
        vals = {m: res['picks'][(m, d)][metric] for m in group if res['picks'][(m, d)]}
        if vals:
            w = better(vals, key=vals.get)
            wins[w] += 1
            per_d.append(f'{D(d)} {L(w)}')
    out = [f'- {what} per dataset: ' + ', '.join(per_d),
           '- datasets won: ' + ', '.join(f'{L(m)} {n}/{len(datasets)}' for m, n in wins.items())]
    sweep = [m for m, n in wins.items() if n == len(datasets)]
    if sweep:
        out[-1] += f'  (**{L(sweep[0])} wins every dataset**)'
    means = []
    for m in group:
        ps = [res['picks'][(m, d)] for d in datasets]
        if all(ps):
            means.append(f"{L(m)} MRR@10 {sum(p['mrr'] for p in ps) / len(ps):.4f}, Recall@10 "
                         f"{sum(p['recall'] for p in ps) / len(ps):.4f}, WAN {sum(p['wan_time'] for p in ps) / len(ps):.3f} s")
        else:
            means.append(f'{L(m)} n/a (misses a dataset)')
    out.append('- mean over datasets: ' + '; '.join(means))
    return out


def _header(res):
    m, d = res['bottleneck']
    if res['kind'] == 'score':
        return (f"#### Frozen {SCORE_LABEL[res['score']]} = {res['value']:.{SCORE_DP}f} (cheapest config at or above it)",
                f"Set by {L(m)} on {D(d)} (its best {SCORE_LABEL[res['score']]} there: {res['bottleneck_value']:.4f}).")
    return (f"#### Frozen WAN = {res['value']:.{WAN_DP}f} s/query (best {SCORE_LABEL[res['score']]} within it)",
            f"Set by {L(m)} on {D(d)} (its fastest run there: {res['bottleneck_value']:.3f} s).")


def frozen_values(nested_data, datasets=None, k=g.K_MAIN, groups=None, mrr_targets=None, recall_targets=None,
                  wan_budgets=None, write=True, quiet=False):
    datasets = datasets or DATASETS
    groups = groups or GROUPS
    methods = sorted({m for _, ms in groups for m in ms})
    by = load_runs(nested_data, methods, datasets, k)
    given = {'mrr': mrr_targets, 'recall': recall_targets}
    lines = [f'# A. Frozen values across datasets (k={k}; Tiptoe and Coeus excluded)', '',
             'One value shared by every dataset: a score target (each method\'s cheapest config reaching it) or a WAN '
             'budget (each method\'s best config within it). Computed values: score = floor of the lowest best score '
             'over methods and datasets; WAN = the smallest budget where every method has a run on every dataset.', '']
    summary = ['## Frozen values used', '', '| Group | Datasets | Frozen MRR@10 | Frozen Recall@10 | Frozen WAN (s/query) |',
               '|---|---|---|---|---|']
    js = {}
    for name, group in groups:
        used, skipped = group_datasets(by, group, datasets)
        lines += [f'## {name}: {", ".join(L(m) for m in group)}', '']
        if skipped:
            lines.append('Skipped (no runs at k=%d): ' % k +
                         '; '.join(f"{D(d)} (no {', '.join(L(m) for m in ms)})" for d, ms in skipped.items()))
            lines.append('')
        if not used:
            lines += ['No dataset where every method of this group has runs.', '']
            continue
        lines += [f'Datasets: {", ".join(D(d) for d in used)}', '']
        results = []
        for sk in SCORE_KEYS:
            for t in (given[sk] or [None]):
                results.append(frozen_score(by, group, used, sk, t))
        for sk in SCORE_KEYS:
            for w in (wan_budgets or [None]):
                results.append(frozen_wan(by, group, used, sk, w))
        for res in results:
            h, note = _header(res)
            lines += [h, '', note, ''] + _table(res, group, used) + [''] + _summary(res, group, used) + ['']
        fmt = lambda kind, sk, dp: ', '.join(f"{r['value']:.{dp}f}" for r in results if r['kind'] == kind and r['score'] == sk)
        summary.append(f"| {name} | {', '.join(D(d) for d in used)} | {fmt('score', 'mrr', SCORE_DP)} | "
                       f"{fmt('score', 'recall', SCORE_DP)} | {fmt('wan', 'mrr', WAN_DP)} |")
        js[name] = {'methods': group, 'datasets': used, 'skipped': skipped, 'results': [
            {'kind': r['kind'], 'score': r['score'], 'value': r['value'], 'bottleneck': list(r['bottleneck']),
             'bottleneck_value': r['bottleneck_value'],
             'picks': {f'{m}/{d}': (_row(p) if p else None) for (m, d), p in r['picks'].items()}}
            for r in results]}
    text = '\n'.join(lines[:4] + summary + [''] + lines[4:]) + '\n'
    if not quiet:
        print(text)
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'frozen_values_k{k}')
        open(base + '.md', 'w').write(text)
        json.dump(js, open(base + '.json', 'w'), indent=2)
        log.info("saved %s.md / .json", base)
    return js, text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('-k', type=int, default=g.K_MAIN)
    ap.add_argument('--mrr', nargs='+', type=float, help='MRR@10 targets instead of the computed one')
    ap.add_argument('--recall', nargs='+', type=float, help='Recall@10 targets instead of the computed one')
    ap.add_argument('--wan', nargs='+', type=float, help='WAN budgets (s/query) instead of the computed one')
    args = ap.parse_args()
    g.setup_logging()
    frozen_values(load_results.load_results(), args.datasets, args.k, mrr_targets=args.mrr,
                  recall_targets=args.recall, wan_budgets=args.wan)
