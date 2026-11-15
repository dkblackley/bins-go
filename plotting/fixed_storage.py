"""
Fixed client storage comparison (like frozen_values.py, but the frozen value is the client storage). For each
storage target S (GB) every method's runs with client storage in S +/- TOL_GB are the candidates, and two tables
are made:

  best MRR@10   the matching run with the highest MRR@10 (ties: Recall@10, then lowest WAN)
  lowest WAN    the matching run with the lowest WAN rounded down to 2 dp; runs at the same rounded WAN go to the
                highest MRR@10 (then Recall@10)

Two separate comparisons (GROUPS), since Tree and Bin clients are very different sizes:
  PILLAR-Bin vs PACMANN    0.5 / 1.5 / 2.5 GB
  PILLAR-Tree vs PACMANN   4.5 / 5.0 / 5.5 GB

    python fixed_storage.py                                   # both comparisons, MS MARCO, +/- 0.1 GB
    python fixed_storage.py --groups bin --bin-storage 0.5 1 1.5 2 2.5
    python fixed_storage.py --groups tree --tree-storage 4.5 5 5.5 6 --tol 0.25
    python fixed_storage.py --mode budget                     # client storage <= S instead of S +/- tol
    python fixed_storage.py --datasets msmarco msmarco-v2-25m

Client storage is ClientStorageMB from metadata.json (MiB); 1 GB = MB_PER_GB MB (1024, so GB = GiB). Runs are
loaded from the simulated results (g.ONLY_SIMULATED) with the client-storage filter off, so no run is dropped for
being larger than the embedding DB. Runs with rounded-down MRR@10 below MIN_MRR (grid-search leftovers) are dropped.
When a method has no run in the window, its row says so and gives its nearest run's client storage.

Writes figures/fixed_storage_k<K>[_budget].md and .json.
"""

import argparse
import json
import logging
import math
import os

import globals as g
import load_results
import plot_utils as pu
import frozen_values as fv

log = logging.getLogger('fixed_storage')

DATASETS = ['msmarco']
# separate comparisons (Tree's client is >= 4.5 GB on MS MARCO, Bin's mostly far smaller): name, methods, targets (GB)
GROUPS = {'bin': ('PILLAR-Bin vs PACMANN', ['bins', 'pacmann'], [0.5, 1.5, 2.5]),
          'tree': ('PILLAR-Tree vs PACMANN', ['tree', 'pacmann'], [4.5, 5.0, 5.5])}
TOL_GB = 0.1          # a run matches S when |storage - S| <= TOL_GB (0.4..0.6 GB for 0.5 GB)
MB_PER_GB = 1024      # ClientStorageMB is MiB
MIN_MRR = 0.01
MODE = 'window'       # 'window': S +/- TOL_GB; 'budget': storage <= S
PICKS = {
    'mrr': ('Best MRR@10', lambda r: (-r['mrr'], -r['recall'], r['wan_time'])),
    # WAN compared rounded down to 2 dp, so runs at the same WAN (e.g. 0.05 s) go to the highest MRR@10
    'wan': ('Lowest WAN', lambda r: (fv.rkey(r['wan_time']), -r['mrr'], -r['recall'], r['wan_time'])),
}
COLS = '| Storage target | Method | Config | Client storage (GB) | MRR@10 | Recall@10 | WAN (s/query) | LAN (s/query) | Comm (KB/query) | Matches |'

L, D, r2 = fv.L, fv.D, fv.r2


def gb(r):
    return r['client_storage_mb'] / MB_PER_GB


def load_runs(nd, methods, datasets, k):
    need = ['mrr', 'recall', 'wan_time', 'client_storage_mb']
    by = {}
    for m in methods:
        for d in datasets:
            by[(m, d)] = [r for r in nd.get(m, {}).get(d, []) if r['k'] == k
                          and all(pu.is_number(r.get(x)) for x in need)
                          and (not MIN_MRR or fv.rkey(r['mrr']) >= fv.rkey(MIN_MRR))]
    return by


def matching(runs, s, tol, mode):
    if mode == 'budget':
        return [r for r in runs if gb(r) <= s + 1e-9]
    return [r for r in runs if abs(gb(r) - s) <= tol + 1e-9]


def target_label(s, tol, mode):
    return f'≤ {s:g} GB' if mode == 'budget' else f'{s:g} ± {tol:g} GB'


def table(by, methods, d, targets, tol, mode, pick):
    title, key = PICKS[pick]
    out = [f'### {D(d)}: {title} at fixed client storage ({" vs ".join(L(m) for m in methods)})', '', COLS, '|' + '---|' * 10]
    js = {}
    for s in targets:
        for m in methods:
            runs = by[(m, d)]
            at = matching(runs, s, tol, mode)
            lab = target_label(s, tol, mode)
            if not runs:
                out.append(f'| {lab} | {L(m)} | — (no runs) |' + ' |' * 7)
                js[f'{s:g}/{m}'] = None
            elif not at:
                near = min(runs, key=lambda r: abs(gb(r) - s))
                out.append(f'| {lab} | {L(m)} | no run in range (nearest: {gb(near):.2f} GB, `{near["config"]}`) |'
                           + ' |' * 6 + ' 0 |')
                js[f'{s:g}/{m}'] = None
            else:
                p = min(at, key=key)
                lan = r2(p['lan_time']) if pu.is_number(p.get('lan_time')) else 'n/a'
                comm = f"{p['comm_kb']:.1f}" if pu.is_number(p.get('comm_kb')) else 'n/a'
                out.append(f"| {lab} | {L(m)} | `{p['config']}` | {gb(p):.2f} | {r2(p['mrr'])} | {r2(p['recall'])} | "
                           f"{r2(p['wan_time'])} | {lan} | {comm} | {len(at)} |")
                js[f'{s:g}/{m}'] = dict(fv._row(p), client_storage_gb=gb(p), n_matching=len(at))
    out.append('')
    return out, js


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--groups', nargs='+', choices=list(GROUPS), default=list(GROUPS))
    ap.add_argument('--bin-storage', nargs='+', type=float, help='Bin vs PACMANN targets in GB')
    ap.add_argument('--tree-storage', nargs='+', type=float, help='Tree vs PACMANN targets in GB')
    ap.add_argument('--tol', type=float, default=TOL_GB, help='match window in GB (S +/- tol)')
    ap.add_argument('--mode', choices=['window', 'budget'], default=MODE)
    ap.add_argument('-k', type=int, default=g.K_MAIN)
    args = ap.parse_args()
    g.setup_logging()
    given = {'bin': args.bin_storage, 'tree': args.tree_storage}
    nd = load_results.load_results(filter_client=False)
    methods = sorted({m for gk in args.groups for m in GROUPS[gk][1]})
    by = load_runs(nd, methods, args.datasets, args.k)
    win = (f'client storage ≤ S' if args.mode == 'budget' else
           f'client storage within S ± {args.tol:g} GB')
    lines = [f'# Fixed client storage (k={args.k})', '',
             f'Each method\'s runs with {win} (1 GB = {MB_PER_GB} MB, ClientStorageMB). Best MRR@10: highest MRR@10 '
             'among them (ties: Recall@10, lowest WAN). Lowest WAN: lowest WAN (rounded down to 2 dp) among them, ties to the highest MRR@10. '
             f'MRR@10 / Recall@10 / WAN rounded down to 2 dp. Runs with MRR@10 below {MIN_MRR:.2f} dropped. '
             'Matches = number of runs in range. PILLAR-Bin and PILLAR-Tree are compared with PACMANN separately, '
             'each at its own storage targets.', '']
    for d in args.datasets:
        rng = '; '.join(f"{L(m)} {min(map(gb, by[(m, d)])):.2f}–{max(map(gb, by[(m, d)])):.2f} GB ({len(by[(m, d)])} runs)"
                        if by[(m, d)] else f'{L(m)} no runs' for m in methods)
        lines += [f'Client storage range on {D(d)}: {rng}.', '']
    js = {}
    for gk in args.groups:
        name, group, targets = GROUPS[gk]
        targets = given[gk] or targets
        lines += [f'## {name} ({", ".join(f"{t:g}" for t in targets)} GB)', '']
        for pick in PICKS:
            for d in args.datasets:
                t, j = table(by, group, d, targets, args.tol, args.mode, pick)
                lines += t
                js.setdefault(name, {}).setdefault(pick, {})[d] = j
    text = '\n'.join(lines) + '\n'
    print(text)
    os.makedirs(g.FIGURE_DIR, exist_ok=True)
    base = os.path.join(g.FIGURE_DIR, f'fixed_storage_k{args.k}' + ('_budget' if args.mode == 'budget' else ''))
    open(base + '.md', 'w').write(text)
    json.dump(js, open(base + '.json', 'w'), indent=2)
    log.info('saved %s.md / .json', base)


if __name__ == '__main__':
    main()
