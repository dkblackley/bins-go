"""
One frozen value, identical on every dataset (rebuttal item A, replaces
fixed_score.py's per-dataset targets). Tiptoe and Coeus are left out. Every
value is rounded DOWN to DP (2) decimals (0.199 -> 0.19), and "identical" means
identical after that, up to a window of +/- TOL (0.01, --tol; 0 = exact): a
frozen MRR@10 of 0.37 +/- 0.01 picks, for every method on every dataset, a
config whose rounded-down MRR@10 is 0.36, 0.37 or 0.38 -- not any one above
0.37. WAN is matched on the raw time instead: W is a multiple of 0.1 s
(WAN_DP = 1 decimal) and a run matches when |WAN - W| <= WAN_TOL (0.05 s,
--wan-tol).

    python frozen_values.py
    python frozen_values.py --datasets msmarco nq scifact fever
    python frozen_values.py --mrr 0.20 --recall 0.40 --wan 1.5 2.0   # your own values

Two kinds of freeze, each done separately for two pairs, PACMANN vs PILLAR-Bin
and PACMANN vs PILLAR-Tree, so each pair gets its own frozen values:

  Frozen score  one MRR@10 (and one Recall@10) value T: the highest value that
                every method in the group has a run at, on every dataset.
                Among the runs at exactly T, each method uses its cheapest:
                lowest PICK_BY (WAN by default; --pick-by client_storage_mb
                for the lowest client storage instead, ties to lower WAN).
  Frozen WAN    hardcoded per pair in FIXED_WAN (picked by scanning W = 0.1 ..
                2.0 s on the local results, 2026-11-11):
                  PACMANN vs Tree: W = 0.6 +/- 0.05 s, each method's run
                    closest to W (ties: best MRR@10, then Recall@10). Both
                    sides land at 0.57-0.60 s on mid-range configs.
                  PACMANN vs Bin: a budget, each method's best MRR@10 with
                    WAN <= 0.3 s. Bin's whole non-dominated frontier is at or
                    below ~0.2 s and PACMANN's starts at 0.26 s, so a matched
                    WAN would always pit PACMANN's weakest config against Bin.
                --wan replaces the value (the pair's mode stays); a pair not
                in FIXED_WAN gets the lowest W shared within +/- WAN_TOL.

When no value is shared (within the window) by every (method, dataset), the
value shared by the most is used and the pairs without a run at it are shown as missing. Every
shared value is listed, so another one can be passed with --mrr / --recall /
--wan.

A dataset joins a group only if every method of the group has runs there
(others are listed as skipped). After each table: who wins on each dataset
(ties at DP decimals are ties) and the per-method mean over datasets, so it is
easy to see whether one method wins everywhere.

Grid-search leftovers are dropped before anything is frozen (counts in the
output), otherwise a fast config that returns nothing gets picked:
  MIN_MRR (0.01, --min-mrr)  rounded-down MRR@10 must be at least this, i.e. not
                             0.00 (0 = no floor)
  dominated runs             WAN freeze only: a run is dropped when another run of
                             the same method on the same dataset is at least as fast
                             with at least as high MRR@10 and Recall@10 (nobody
                             would report it; --keep-dominated turns this off). The
                             score freeze already picks the cheapest run at T, so it
                             keeps them (more runs = more values to match).

Dataset selection (--auto-datasets, on by default; --no-auto-datasets uses
--datasets as given): the largest set of datasets on which BOTH pairs have a
fully shared value for every metric in AUTO_METRICS (MRR@10 and Recall@10 by
default; add wan with --auto-metrics mrr recall wan). That one set is then used
for every table of both pairs. Candidates: --datasets (default: every dataset in
g.ALL_DATASETS where Bin, Tree and PACMANN all have runs). Ties between sets of
the same size go to the one with the highest frozen scores. The report also
gives the largest set for each metric on its own.

Writes figures/frozen_values_k<K>.md and .json (frozen_values_k<K>_<pick-by>.md / .json
when PICK_BY is not wan_time, so both versions can be handed in).
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
GROUPS = [('PACMANN vs PILLAR-Bin', ['pacmann', 'bins']),
          ('PACMANN vs PILLAR-Tree', ['pacmann', 'tree'])]
SCORE_KEYS = ['mrr', 'recall']
COST_KEY = 'wan_time'   # the WAN freeze matches on this
PICK_BY = 'wan_time'    # what "cheapest" means in the score freeze; 'client_storage_mb' = lowest client storage
PICK_BY_CHOICES = ['wan_time', 'client_storage_mb', 'lan_time', 'comm_kb']
COST_LABEL = {'wan_time': 'WAN', 'client_storage_mb': 'client storage', 'lan_time': 'LAN', 'comm_kb': 'communication'}
DP = 2            # every value is rounded down (truncated) to this many decimals; frozen values match at this precision
TOL = 0.01        # a run matches a frozen MRR@10 / Recall@10 value T when its rounded value is within T ± TOL
WAN_DP = 1        # WAN values are rounded down to this many decimals
WAN_TOL = 0.05    # a run matches a frozen WAN value W when |raw WAN - W| <= WAN_TOL (s/query)
# hardcoded WAN comparison per pair: ('closest', W) = run closest to W within W ± WAN_TOL;
# ('budget', W) = best MRR@10 with WAN <= W. See the docstring for why these values.
FIXED_WAN = {'PACMANN vs PILLAR-Bin': ('budget', 0.3),
             'PACMANN vs PILLAR-Tree': ('closest', 0.6)}
MIN_MRR = 0.01    # runs whose rounded-down MRR@10 is below this are dropped (0.00 = the config doesn't work)
AUTO_DATASETS = True    # pick the largest dataset set where every AUTO_METRICS value is shared in both pairs
AUTO_METRICS = ['mrr', 'recall']   # 'wan' can be added; PACMANN vs Bin rarely share a WAN
DROP_DOMINATED = True   # drop runs another run of the same method/dataset beats on WAN, MRR@10 and Recall@10
SCORE_LABEL = {'mrr': 'MRR@10', 'recall': 'Recall@10'}
REPORT_KEYS = ['mrr', 'recall', 'wan_time', 'lan_time', 'comm_kb', 'client_storage_mb']


def L(m):
    return g.METHOD_LABELS.get(m, m)


def D(d):
    return g.DATASET_LABELS.get(d, d)


def rkey(x, dp=DP):
    """x rounded down to dp decimals (0.199 -> 19), as an int (x 10^dp) so equal values compare equal.
    The 1e-9 keeps floats like 0.29 (stored as 0.28999...) from dropping a step."""
    return int(math.floor(x * 10 ** dp + 1e-9))


def r2(x, dp=DP):
    """x rounded down to dp decimals, as text (so tables show the value that was matched)."""
    return 'n/a' if x is None else f'{rkey(x, dp) / 10 ** dp:.{dp}f}'


def dominated(r, runs):
    """another run is at least as fast and at least as good on MRR and Recall, and strictly better on one"""
    return any(o is not r and o[COST_KEY] <= r[COST_KEY] and o['mrr'] >= r['mrr'] and o['recall'] >= r['recall']
               and (o[COST_KEY] < r[COST_KEY] or o['mrr'] > r['mrr'] or o['recall'] > r['recall']) for o in runs)


def load_runs(nested_data, methods, datasets, k, min_mrr=MIN_MRR, drop_dominated=DROP_DOMINATED):
    """runs at or above the MRR floor {method: {dataset: [runs]}}, the same without dominated runs (for the WAN
    freeze), and {(method, dataset): (dropped below the floor, dropped as dominated)}"""
    need = SCORE_KEYS + [COST_KEY]
    by, front, dropped = {}, {}, {}
    for m in methods:
        by[m], front[m] = {}, {}
        for d in datasets:
            runs = [r for r in nested_data.get(m, {}).get(d, [])
                    if r['k'] == k and all(pu.is_number(r.get(x)) for x in need)]
            ok = [r for r in runs if rkey(r['mrr']) >= rkey(min_mrr)] if min_mrr else runs
            keep = [r for r in ok if not dominated(r, ok)] if drop_dominated else ok
            by[m][d], front[m][d] = ok, keep
            if len(keep) < len(runs):
                dropped[(m, d)] = (len(runs) - len(ok), len(ok) - len(keep))
    return by, front, dropped


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


def _cost(r, key):
    v = r.get(key)
    return v if pu.is_number(v) else math.inf   # a run without this cost is never the cheapest


def exact_freeze(by, group, datasets, match_key, pick_key, target=None, tol=0.0, dp=DP, pick_by=None):
    """One value T of match_key (rounded down to dp) for every (method, dataset); a run matches when its rounded
    value is within tol of T. Among the matching runs: the cheapest (score freeze) or the one closest to T, ties to
    the best pick_key (WAN freeze)."""
    kind = 'wan' if match_key == COST_KEY else 'score'
    if kind == 'wan':
        return _wan_freeze(by, group, datasets, pick_key, target, tol, dp)
    steps = rkey(tol, dp)
    pairs = [(m, d) for m in group for d in datasets]
    avail = {p: {rkey(r[match_key], dp) for r in by[p[0]][p[1]]} for p in pairs}
    cands = set().union(*avail.values())
    cands = set().union(*({v + i for i in range(-steps, steps + 1)} for v in cands))
    cover = {v: sum(1 for vals in avail.values() if any(abs(x - v) <= steps for x in vals)) for v in cands}
    common = sorted(v for v in cands if cover[v] == len(pairs) and v >= 0)
    if target is not None:
        t = rkey(target, dp)
    elif common:   # highest shared score / lowest shared WAN
        t = common[0] if kind == 'wan' else common[-1]
    else:          # nothing shared by every pair: the value shared by the most (highest score / lowest WAN among those)
        t = max((v for v in cands if v >= 0), key=lambda v: (cover[v], -v if kind == 'wan' else v))
    picks = {}
    for m, d in pairs:
        at = [r for r in by[m][d] if abs(rkey(r[match_key], dp) - t) <= steps]
        if not at:
            picks[(m, d)] = None
        elif kind == 'score':
            picks[(m, d)] = min(at, key=lambda r: (_cost(r, pick_by or PICK_BY), r[COST_KEY], -r[match_key]))
        else:
            w = t / 10 ** dp
            picks[(m, d)] = min(at, key=lambda r: (abs(r[COST_KEY] - w), -r[pick_key], -r['recall']))
    return {'kind': kind, 'match': match_key, 'score': pick_key, 'pick_by': pick_by or PICK_BY,
            'value': t / 10 ** dp, 'tol': steps / 10 ** dp,
            'dp': dp, 'common': [c / 10 ** dp for c in common], 'covered': sum(1 for p in picks.values() if p),
            'pairs': len(pairs), 'picks': picks}


def _wan_freeze(by, group, datasets, pick_key, target, tol, dp, mode='closest'):
    """WAN on the raw time, W on a 10^-dp grid. mode 'closest': a run matches when |WAN - W| <= tol and each method
    takes the run closest to W (ties: best pick_key, then Recall). mode 'budget': a run matches when WAN <= W and
    each method takes its best pick_key (then Recall, then lowest WAN)."""
    pairs = [(m, d) for m in group for d in datasets]
    if mode == 'budget':
        near = lambda r, v: r[COST_KEY] <= v / 10 ** dp + 1e-9
    else:
        near = lambda r, v: abs(r[COST_KEY] - v / 10 ** dp) <= tol + 1e-9
    top = max(r[COST_KEY] for m, d in pairs for r in by[m][d])
    grid = range(0, int(math.ceil((top + tol) * 10 ** dp)) + 1)
    cover = {v: sum(1 for m, d in pairs if any(near(r, v) for r in by[m][d])) for v in grid}
    common = [v for v in grid if cover[v] == len(pairs)]
    if target is not None:
        t = int(round(target * 10 ** dp))
    elif common:
        t = common[0]
    else:
        t = max(grid, key=lambda v: (cover[v], -v))
    w = t / 10 ** dp
    picks = {}
    for m, d in pairs:
        at = [r for r in by[m][d] if near(r, t)]
        if not at:
            picks[(m, d)] = None
        elif mode == 'budget':
            picks[(m, d)] = max(at, key=lambda r: (r[pick_key], r['recall'], -r[COST_KEY]))
        else:
            picks[(m, d)] = min(at, key=lambda r: (abs(r[COST_KEY] - w), -r[pick_key], -r['recall']))
    return {'kind': 'wan', 'match': COST_KEY, 'score': pick_key, 'value': w, 'tol': 0 if mode == 'budget' else tol,
            'dp': dp, 'mode': mode,
            'common': [c / 10 ** dp for c in common], 'covered': sum(1 for p in picks.values() if p),
            'pairs': len(pairs), 'picks': picks}


def _f(v, fmt):
    return 'n/a' if v is None else format(v, fmt)


def _table(res, group, datasets):
    out = ['| Dataset | Method | Config | MRR@10 | Recall@10 | WAN (s/query) | LAN (s/query) | Comm (KB/query) '
           '| Client storage (MB) | Source |', '|' + '---|' * 10]
    for d in datasets:
        for m in group:
            p = res['picks'][(m, d)]
            if p is None:
                out.append(f'| {D(d)} | {L(m)} | no run at this value (within the window) |' + ' |' * 7)
                continue
            out.append(f"| {D(d)} | {L(m)} | `{p['config']}` | {r2(p['mrr'])} | {r2(p['recall'])} | "
                       f"{r2(p['wan_time'])} | {r2(p.get('lan_time'))} | {_f(p.get('comm_kb'), '.1f')} | "
                       f"{_f(p.get('client_storage_mb'), '.1f')} | {p.get('source', 'real')} |")
    return out


def _summary(res, group, datasets):
    """Who wins on each dataset (compared at DP decimals, so equal roundings tie), and the per-method mean."""
    if res['kind'] == 'score':   # same quality: lower WAN wins
        metric, sign = res.get('pick_by', COST_KEY), -1
        what = f"lowest {COST_LABEL.get(metric, metric)}"
    else:                        # same WAN: higher score wins
        metric, sign, what = res['score'], 1, f"highest {SCORE_LABEL[res['score']]}"
    wins, per_d = {m: 0 for m in group}, []
    for d in datasets:
        vals = {m: sign * rkey(res['picks'][(m, d)][metric]) for m in group
                if res['picks'][(m, d)] and pu.is_number(res['picks'][(m, d)].get(metric))}
        if len(vals) < 2:
            continue
        top = [m for m in vals if vals[m] == max(vals.values())]
        if len(top) == 1:
            wins[top[0]] += 1
            per_d.append(f'{D(d)} {L(top[0])}')
        else:
            per_d.append(f"{D(d)} tie ({' = '.join(L(m) for m in top)})")
    out = [f'- {what} per dataset: ' + (', '.join(per_d) or 'n/a'),
           '- datasets won: ' + ', '.join(f'{L(m)} {n}/{len(datasets)}' for m, n in wins.items())]
    sweep = [m for m, n in wins.items() if n == len(datasets)]
    if sweep:
        out[-1] += f'  (**{L(sweep[0])} wins every dataset**)'
    means = []
    for m in group:
        ps = [res['picks'][(m, d)] for d in datasets]
        if all(ps):
            mean = lambda key: r2(sum(p[key] for p in ps) / len(ps))
            st = [p['client_storage_mb'] for p in ps if pu.is_number(p.get('client_storage_mb'))]
            means.append(f"{L(m)} MRR@10 {mean('mrr')}, Recall@10 {mean('recall')}, WAN {mean('wan_time')} s"
                         + (f", client storage {sum(st) / len(st):.0f} MB" if len(st) == len(ps) else ''))
        else:
            means.append(f'{L(m)} n/a (no run at this value on every dataset)')
    out.append('- mean over datasets: ' + '; '.join(means))
    return out


def _vt(res):
    dp = res.get('dp', DP)
    if res.get('mode') == 'budget':
        return '≤ ' + r2(res['value'], dp)
    return r2(res['value'], dp) + (f" ± {res['tol']:g}" if res['tol'] else '')


def _header(res):
    v = _vt(res)
    if res['kind'] == 'score':
        h = f"#### Frozen {SCORE_LABEL[res['match']]} = {v} (every pick is at {v}; lowest {COST_LABEL.get(res['pick_by'], res['pick_by'])} among those runs)"
        name = SCORE_LABEL[res['match']]
    elif res.get('mode') == 'budget':
        h = f"#### WAN budget {v} s/query (each method's best {SCORE_LABEL[res['score']]} within it)"
        name = 'WAN budget'
    else:
        h = (f"#### Frozen WAN = {v} s/query (every pick is at {v} s; the run closest to "
             f"{r2(res['value'], res['dp'])} s, ties to best {SCORE_LABEL[res['score']]})")
        name = 'WAN'
    dp = res.get('dp', DP)
    within = f" (within ± {res['tol']:g})" if res['tol'] else ''
    if res['common']:
        note = f"{name} values every method has a run at{within}, on every dataset: {', '.join(r2(c, dp) for c in res['common'])}."
    else:
        note = f"No {name} value is shared{within} by every method on every dataset."
    if res['covered'] < res['pairs']:
        note += f" {v} is hit by {res['covered']}/{res['pairs']} method-dataset pairs."
    return h, note


def _freeze(by, front, group, datasets, metric, tol, wan_tol, target=None):
    if metric == 'wan':
        return exact_freeze(front, group, datasets, COST_KEY, 'mrr', target, wan_tol, WAN_DP)
    return exact_freeze(by, group, datasets, metric, metric, target, tol)


def _fully_shared(by, front, groups, datasets, metric, tol, wan_tol, target=None):
    """the frozen values (one per group) if every group has full coverage on these datasets, else None"""
    out = []
    for _, group in groups:
        r = _freeze(by, front, group, datasets, metric, tol, wan_tol, target)
        if r['covered'] < r['pairs']:
            return None
        out.append(r['value'])
    return out


def choose_datasets(by, front, groups, candidates, metrics, tol, wan_tol, targets):
    """largest subset of candidates where every metric is fully shared in every group (ties: highest scores)"""
    from itertools import combinations
    for size in range(len(candidates), 0, -1):
        best = None
        for ds in combinations(candidates, size):
            vals = [_fully_shared(by, front, groups, list(ds), mt, tol, wan_tol, targets.get(mt)) for mt in metrics]
            if any(v is None for v in vals):
                continue
            score = sum(sum(v) if mt != 'wan' else -sum(v) for mt, v in zip(metrics, vals))
            if best is None or score > best[0]:
                best = (score, list(ds))
        if best:
            return best[1]
    return []


def frozen_values(nested_data, datasets=None, k=g.K_MAIN, groups=None, mrr_targets=None, recall_targets=None,
                  wan_budgets=None, tol=None, wan_tol=None, min_mrr=MIN_MRR, drop_dominated=DROP_DOMINATED,
                  auto_datasets=AUTO_DATASETS, auto_metrics=None, pick_by=None, write=True, quiet=False):
    pick_by = pick_by or PICK_BY
    tol = TOL if tol is None else tol
    wan_tol = WAN_TOL if wan_tol is None else wan_tol
    auto_metrics = auto_metrics or AUTO_METRICS
    datasets = datasets or (g.ALL_DATASETS if auto_datasets else DATASETS)
    groups = groups or GROUPS
    methods = sorted({m for _, ms in groups for m in ms})
    by, front, dropped = load_runs(nested_data, methods, datasets, k, min_mrr, drop_dominated)
    given = {'mrr': mrr_targets, 'recall': recall_targets}
    auto_lines = []
    if auto_datasets:
        firsts = {'mrr': (mrr_targets or [None])[0], 'recall': (recall_targets or [None])[0],
                  'wan': (wan_budgets or [None])[0]}
        cands = [d for d in datasets if all(by[m][d] for m in methods)]
        chosen = choose_datasets(by, front, groups, cands, auto_metrics, tol, wan_tol, firsts)
        name = lambda mt: 'WAN' if mt == 'wan' else SCORE_LABEL[mt]
        auto_lines = ['## Dataset selection', '',
                      f"Candidates (Bin, Tree and PACMANN all have runs): {', '.join(D(d) for d in cands) or 'none'}"
                      + (f"; no runs for every method: {', '.join(D(d) for d in datasets if d not in cands)}"
                         if len(cands) < len(datasets) else '') + '.',
                      f"Largest set with a fully shared {' and '.join(name(mt) for mt in auto_metrics)} in both pairs: "
                      f"**{', '.join(D(d) for d in chosen) or 'none'}**"
                      + (f" (dropped: {', '.join(D(d) for d in cands if d not in chosen)})" if chosen else '') + '.']
        for mt in ['mrr', 'recall', 'wan']:
            alone = choose_datasets(by, front, groups, cands, [mt], tol, wan_tol, firsts)
            auto_lines.append(f"- {name(mt)} alone: {', '.join(D(d) for d in alone) or 'no dataset'}")
        auto_lines.append('')
        datasets = chosen or cands
    lines = [f'# A. Frozen values across datasets (k={k}; Tiptoe and Coeus excluded)', '',
             f'One value for every method on every dataset (values rounded down to {DP} decimals; a run matches when '
             f'it is within ± {tol:.{DP}f} of an MRR@10 / Recall@10 value, ± {wan_tol:g} s of a WAN value (raw time; WAN values are multiples of 0.1 s)): an '
             f'MRR@10 / Recall@10 value (each method\'s matching run with the lowest {COST_LABEL.get(pick_by, pick_by)}) or a WAN latency (each method\'s matching '
             'run closest to it). '
             'Computed values: the highest score / lowest WAN that every method has a run at on every dataset (if '
             'none, the value shared by the most method-dataset pairs). Coverage in brackets when not complete.', '']
    rules = ([f'MRR@{k} below {min_mrr:.{DP}f}'] if min_mrr else []) + (['dominated'] if drop_dominated else [])
    if rules:
        lines += [f"Runs dropped ({' / '.join(rules)}, the latter for the WAN freeze only): " +
                  (', '.join(f'{L(m)} {D(d)} {a} / {b}' for (m, d), (a, b) in sorted(dropped.items())) or 'none') + '.', '']
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
                results.append(exact_freeze(by, group, used, sk, sk, t, tol, pick_by=pick_by))
        mode, fixed = FIXED_WAN.get(name, ('closest', None))
        for w in (wan_budgets or [fixed]):   # one table per W (the pick doesn't depend on MRR vs Recall)
            results.append(_wan_freeze(front, group, used, 'mrr', w, wan_tol, WAN_DP, mode))
        for res in results:
            h, note = _header(res)
            lines += [h, '', note, ''] + _table(res, group, used) + [''] + _summary(res, group, used) + ['']
        fmt = lambda kind, sk: ', '.join(_vt(r) + ('' if r['covered'] == r['pairs'] else f" ({r['covered']}/{r['pairs']})")
                                         for r in results if r['kind'] == kind and r['score'] == sk)
        summary.append(f"| {name} | {', '.join(D(d) for d in used)} | {fmt('score', 'mrr')} | "
                       f"{fmt('score', 'recall')} | {fmt('wan', 'mrr')} |")
        js[name] = {'methods': group, 'datasets': used, 'skipped': skipped, 'results': [
            {'kind': r['kind'], 'match': r['match'], 'score': r['score'], 'value': r['value'], 'tol': r['tol'],
             'common': r['common'],
             'covered': r['covered'], 'pairs': r['pairs'],
             'picks': {f'{m}/{d}': (_row(p) if p else None) for (m, d), p in r['picks'].items()}}
            for r in results]}
    text = '\n'.join(lines[:4] + auto_lines + summary + [''] + lines[4:]) + '\n'
    if not quiet:
        print(text)
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'frozen_values_k{k}' + ('' if pick_by == 'wan_time' else f'_{pick_by}'))
        open(base + '.md', 'w').write(text)
        json.dump(js, open(base + '.json', 'w'), indent=2)
        log.info("saved %s.md / .json", base)
    return js, text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', help='datasets (with auto selection: the candidates; default every '
                    'dataset in g.ALL_DATASETS, else ' + ' '.join(DATASETS) + ')')
    ap.add_argument('--no-auto-datasets', action='store_true', help='use --datasets as given, no selection')
    ap.add_argument('--auto-metrics', nargs='+', choices=['mrr', 'recall', 'wan'], default=AUTO_METRICS,
                    help='metrics that must be fully shared in both pairs on the selected datasets')
    ap.add_argument('-k', type=int, default=g.K_MAIN)
    ap.add_argument('--mrr', nargs='+', type=float, help='MRR@10 values (matched at 2 dp) instead of the computed one')
    ap.add_argument('--recall', nargs='+', type=float, help='Recall@10 values (matched at 2 dp) instead of the computed one')
    ap.add_argument('--wan', nargs='+', type=float, help='WAN latencies (s/query, multiples of 0.1) instead of the computed one')
    ap.add_argument('--tol', type=float, default=TOL, help='MRR@10 / Recall@10 match window, T +/- tol (0 = exact)')
    ap.add_argument('--wan-tol', type=float, default=WAN_TOL, help='WAN match window in s, W +/- tol (0 = exact)')
    ap.add_argument('--min-mrr', type=float, default=MIN_MRR, help='drop runs with MRR@10 below this (0 = no floor)')
    ap.add_argument('--pick-by', choices=PICK_BY_CHOICES, default=PICK_BY,
                    help='score freeze: pick the matching run with the lowest of this (default WAN)')
    ap.add_argument('--keep-dominated', action='store_true', help='keep runs that another run beats on WAN, MRR and Recall')
    args = ap.parse_args()
    g.setup_logging()
    frozen_values(load_results.load_results(), args.datasets, args.k, mrr_targets=args.mrr,
                  recall_targets=args.recall, wan_budgets=args.wan, tol=args.tol, wan_tol=args.wan_tol,
                  min_mrr=args.min_mrr, drop_dominated=not args.keep_dominated,
                  auto_datasets=not args.no_auto_datasets, auto_metrics=args.auto_metrics,
                  pick_by=args.pick_by)
