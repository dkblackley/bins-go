"""
Equal-quality comparison (rebuttal item A): for a target T, every method uses its
cheapest config (lowest WAN time) with Recall@10 >= T; the same for MRR@10.

    python fixed_score.py
    python fixed_score.py --datasets msmarco nq --cost lan_time
    python fixed_score.py --targets 0.15 0.30          # your own Recall/MRR targets instead

Two targets per dataset and score (floored to DP=2 decimals, so the method
that sets them still reaches them):

  T1  the highest score EVERY method with runs reaches: floor(min over methods
      of each method's best score).
  T2  a higher one that only the strongest methods reach: floor(min of PILLAR-
      Tree's and PACMANN's best scores), if that is above T1. Methods below it
      are listed as "does not reach".

For each method at each target: config, Recall@10, MRR@10, WAN time, comm KB,
client storage MB, real/simulated. Then the speed-ups at T1 (PACMANN, Tiptoe,
Coeus WAN / Bin WAN; PACMANN WAN / Tree WAN), every method's best score, and
which method is alone at the top ("only private method above X" = the second
best method's best score).

Bins runs dropped by the client storage filter (load_results, 2/3 of the
embedding DB) are not used; when they reach higher scores than the kept ones
the report says so (FILTER_CLIENT_STORAGE=0 to include them).

Writes figures/fixed_score_k<K>.json and .md.
"""

import argparse
import json
import logging
import math
import os

import globals as g
import load_results
import plot_utils as pu

log = logging.getLogger('fixed_score')

METHODS = ['bins', 'tree', 'pacmann', 'tiptoe', 'coeus']
DATASETS = ['msmarco', 'scifact', 'nq', 'fever', 'msmarco-v2']
SCORE_KEYS = ['recall', 'mrr']
COST_KEY = 'wan_time'
DP = 2
REPORT_KEYS = ['recall', 'mrr', 'wan_time', 'lan_time', 'comm_kb', 'client_storage_mb']
SCORE_LABEL = {'recall': 'Recall@10', 'mrr': 'MRR@10'}


def _label(m):
    return g.METHOD_LABELS.get(m, m)


def floor_dp(x, dp=DP):
    return math.floor(x * 10 ** dp + 1e-9) / 10 ** dp


def _runs(nested_data, method, dataset, k, score_key, cost_key):
    return [r for r in nested_data.get(method, {}).get(dataset, [])
            if r['k'] == k and pu.is_number(r.get(score_key)) and pu.is_number(r.get(cost_key))]


def _row(run):
    row = {'config': run['config'], 'folder': run['folder'], 'source': run.get('source', 'real')}
    for key in REPORT_KEYS:
        v = run.get(key, math.nan)
        row[key] = v if pu.is_number(v) else None
    return row


def cheapest_at(runs, score_key, target, cost_key):
    ok = [r for r in runs if r[score_key] >= target - 1e-9]
    return min(ok, key=lambda r: (r[cost_key], -r[score_key])) if ok else None


def score_block(nested_data, dataset, k, score_key, cost_key, methods, targets=None):
    by_m = {m: _runs(nested_data, m, dataset, k, score_key, cost_key) for m in methods}
    by_m = {m: rs for m, rs in by_m.items() if rs}
    if not by_m:
        return None
    best = {m: max(rs, key=lambda r: r[score_key]) for m, rs in by_m.items()}
    maxes = {m: best[m][score_key] for m in by_m}
    if targets:
        tlist = [('given', t) for t in targets]
    else:
        t1 = floor_dp(min(maxes.values()))
        tlist = [('T1: every method reaches it', t1)]
        top = [maxes[m] for m in ('tree', 'pacmann') if m in maxes]
        if top and floor_dp(min(top)) > t1:
            tlist.append(('T2: reached by Tree and PACMANN', floor_dp(min(top))))
    out = {'score': score_key, 'methods': list(by_m),
           'best': {m: dict(_row(best[m]), score=maxes[m]) for m in by_m}, 'targets': []}
    for name, t in tlist:
        picks = {m: cheapest_at(rs, score_key, t, cost_key) for m, rs in by_m.items()}
        entry = {'name': name, 'target': t, 'methods': {m: (_row(p) if p else None) for m, p in picks.items()}}
        c = {m: p[cost_key] for m, p in picks.items() if p and p[cost_key] > 0}
        sp = {}
        for a, b in (('pacmann', 'bins'), ('tiptoe', 'bins'), ('coeus', 'bins'), ('pacmann', 'tree'), ('tree', 'bins')):
            if a in c and b in c:
                sp[f'{a}/{b}'] = c[a] / c[b]
        entry['ratios'] = sp
        out['targets'].append(entry)
    ranked = sorted(maxes.items(), key=lambda kv: -kv[1])
    if len(ranked) >= 2:
        out['top'] = {'method': ranked[0][0], 'score': ranked[0][1], 'runner_up': ranked[1][0],
                      'runner_up_score': ranked[1][1]}
    dropped = [r for r in load_results.DROPPED_BINS.get(dataset, [])
               if r['k'] == k and pu.is_number(r.get(score_key))]
    if dropped and 'bins' in maxes:
        hidden = max(dropped, key=lambda r: r[score_key])
        if hidden[score_key] > maxes['bins']:
            out['bins_filtered_note'] = {'n': len(dropped), 'best_hidden': hidden[score_key],
                                         'config': hidden['config'], 'client_storage_mb': hidden['client_storage_mb'],
                                         'kept_best': maxes['bins']}
    return out


def _f(v, fmt):
    return 'n/a' if v is None else format(v, fmt)


def report_dataset(res, cost_key):
    lines = [f"\n## {g.DATASET_LABELS.get(res['dataset'], res['dataset'])} (k={res['k']}, cost = {cost_key})"]
    for score_key in SCORE_KEYS:
        b = res.get(score_key)
        if not b:
            lines.append(f"\n{SCORE_LABEL[score_key]}: no runs")
            continue
        lines.append(f"\n### {SCORE_LABEL[score_key]}")
        lines.append('best per method: ' + ', '.join(f"{_label(m)} {v['score']:.4f} ({v['config']})"
                                                     for m, v in sorted(b['best'].items(), key=lambda kv: -kv[1]['score'])))
        if 'top' in b:
            t = b['top']
            lines.append(f"top: {_label(t['method'])} {t['score']:.4f}; next {_label(t['runner_up'])} "
                         f"{t['runner_up_score']:.4f}  ->  \"{_label(t['method'])} is the only private method above "
                         f"{SCORE_LABEL[score_key]} {floor_dp(t['runner_up_score'], 3):.3f}\"")
        n = b.get('bins_filtered_note')
        if n:
            lines.append(f"NOTE: {n['n']} PILLAR-Bin runs dropped by the client storage filter reach up to "
                         f"{n['best_hidden']:.4f} ({n['config']}, {n['client_storage_mb']:.0f} MB client storage) vs "
                         f"{n['kept_best']:.4f} kept. FILTER_CLIENT_STORAGE=0 to include them.")
        for t in b['targets']:
            lines.append(f"\n{t['name']}: {SCORE_LABEL[score_key]} >= {t['target']:.2f}")
            hdr = f"  {'method':<13}{'config':<34}{'Recall@10':>10}{'MRR@10':>9}{'WAN s':>9}{'LAN s':>8}" \
                  f"{'comm KB':>10}{'client MB':>11}  source"
            lines.append(hdr)
            for m, r in t['methods'].items():
                if r is None:
                    lines.append(f"  {_label(m):<13}does not reach it")
                    continue
                lines.append(f"  {_label(m):<13}{r['config']:<34}{_f(r['recall'], '.4f'):>10}{_f(r['mrr'], '.4f'):>9}"
                             f"{_f(r['wan_time'], '.3f'):>9}{_f(r['lan_time'], '.3f'):>8}{_f(r['comm_kb'], '.1f'):>10}"
                             f"{_f(r['client_storage_mb'], '.1f'):>11}  {r['source']}")
            for key, ratio in t['ratios'].items():
                slow, fast = key.split('/')   # ratio = WAN of `slow` / WAN of `fast`
                if ratio >= 1:
                    lines.append(f"  {_label(fast)} is {ratio:.2f}x faster than {_label(slow)} (WAN)")
                else:
                    lines.append(f"  {_label(fast)} is {1 / ratio:.2f}x slower than {_label(slow)} (WAN)")
    return '\n'.join(lines)


def fixed_score(nested_data, datasets=None, k=g.K_MAIN, cost_key=COST_KEY, methods=None, targets=None, write=True,
                quiet=False):
    results, texts = {}, []
    for dataset in datasets or DATASETS:
        res = {'dataset': dataset, 'k': k}
        for score_key in SCORE_KEYS:
            res[score_key] = score_block(nested_data, dataset, k, score_key, cost_key, methods or METHODS, targets)
        results[dataset] = res
        texts.append(report_dataset(res, cost_key))
    text = '# A. Equal-quality comparison (cheapest config at or above each target)\n' + '\n'.join(texts)
    if not quiet:
        print(text)
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'fixed_score_k{k}')
        json.dump(results, open(base + '.json', 'w'), indent=2)
        open(base + '.md', 'w').write('```\n' + text + '\n```\n')
        log.info("saved %s.json / .md", base)
    return results, text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('-k', type=int, default=g.K_MAIN)
    ap.add_argument('--cost', default=COST_KEY, help='run key minimised at each target (wan_time, lan_time, ...)')
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--targets', nargs='+', type=float, help='use these targets (same for Recall and MRR)')
    args = ap.parse_args()
    g.setup_logging()
    fixed_score(load_results.load_results(), args.datasets, args.k, args.cost, args.methods, args.targets)
