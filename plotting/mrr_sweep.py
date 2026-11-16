"""
Fixed-MRR@10 sweep: for every target T in a range, each method's lowest-WAN config at that MRR@10, on every
dataset, to find the targets where PILLAR-Bin / PILLAR-Tree vs PACMANN / Tiptoe / Coeus tell the same story
everywhere. Runs are loaded exactly as in working_rebuttal.py (same CONFIG: storage filter, Tiptoe token, MIN_MRR).

    python mrr_sweep.py                                  # T = 0.05 .. 0.25, rule atleast, filter off
    python mrr_sweep.py --from 0.10 --to 0.20 --show 0.10 0.16
    python mrr_sweep.py --rule window --storage-filter on

Rules (MRR@10 rounded down to 2 dp, as everywhere else):
  atleast  the lowest-WAN config with MRR@10 >= T (a method whose cheapest config is already above T keeps it)
  window   the lowest-WAN config with |MRR@10 - T| <= TOL (0.01)
A method with no config meeting the rule shows '—' with the best MRR@10 it reaches.

Writes figures/mrr_sweep_k<K>[_window][_filter_on].md.
"""

import argparse
import os

import globals as g
import frozen_values as fv
import working_rebuttal as wr

DATASETS = ['msmarco', 'scifact', 'fever', 'nq']
METHODS = ['bins', 'tree', 'pacmann', 'tiptoe', 'coeus']
BASE = ['pacmann', 'tiptoe', 'coeus']
L = wr.L
D = {'msmarco': 'MS MARCO', 'scifact': 'SciFact', 'fever': 'FEVER', 'nq': 'NQ'}


def picks(R, T, rule):
    out = {}
    for d in DATASETS:
        for m in METHODS:
            rs = R.get(m, d)
            if rule == 'window':
                ok = [r for r in rs if abs(fv.rkey(r['mrr']) - fv.rkey(T)) <= fv.rkey(wr.TOL)]
            else:
                ok = [r for r in rs if fv.rkey(r['mrr']) >= fv.rkey(T)]
            best = max(rs, key=lambda r: r['mrr']) if rs else None
            out[(d, m)] = (min(ok, key=lambda r: (r['wan_time'], -r['mrr'])) if ok else None, best)
    return out


def wins(p, d, m, against):
    r = p[(d, m)][0]
    others = [p[(d, b)][0] for b in against if p[(d, b)][0]]
    return r is not None and all(r['wan_time'] < o['wan_time'] for o in others)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='lo', type=float, default=0.05)
    ap.add_argument('--to', dest='hi', type=float, default=0.25)
    ap.add_argument('--rule', choices=['atleast', 'window'], default='atleast')
    ap.add_argument('--storage-filter', choices=['off', 'on'], default=wr.STORAGE_FILTER)
    ap.add_argument('--show', nargs='*', type=float, help='targets to print as a per-method WAN table '
                    '(default: the highest T every method reaches on every dataset)')
    args = ap.parse_args()
    g.setup_logging()
    R = wr.Runs(wr.load([args.storage_filter], [])[args.storage_filter])
    ts = [x / 100 for x in range(fv.rkey(args.lo), fv.rkey(args.hi) + 1)]
    P = {T: picks(R, T, args.rule) for T in ts}
    f2 = fv.r2
    out = [f'# Fixed-MRR@10 sweep (k={wr.K}, rule {args.rule}, client-storage filter {args.storage_filter}, '
           f"Tiptoe token {'in' if wr.TIPTOE_TOKEN else 'out of'} its WAN)", '',
           'Best MRR@10 each method reaches: ' + '; '.join(
               f"{D[d]} " + ', '.join(f"{L(m)} {f2(P[ts[0]][(d, m)][1]['mrr']) if P[ts[0]][(d, m)][1] else 'n/a'}"
                                      for m in METHODS) for d in DATASETS), '',
           '## Where the story holds', '',
           '| T | method-dataset pairs at T | Bin fastest of all | Tree beats Tiptoe + Coeus | Tree beats PACMANN | cannot reach T |',
           '|---|---|---|---|---|---|']
    full = []
    for T in ts:
        p = P[T]
        miss = [f'{L(m)} {D[d]}' for d in DATASETS for m in METHODS if p[(d, m)][0] is None]
        n = len(DATASETS) * len(METHODS) - len(miss)
        yn = lambda m, ag: (lambda bad: 'yes' if not bad else 'no: ' + ', '.join(bad))(
            [D[d] for d in DATASETS if not wins(p, d, m, ag)])
        out.append(f"| {T:.2f} | {n}/{len(DATASETS) * len(METHODS)} | {yn('bins', ['tree'] + BASE)} | "
                   f"{yn('tree', ['tiptoe', 'coeus'])} | {yn('tree', ['pacmann'])} | {', '.join(miss) or '-'} |")
        if not miss:
            full.append(T)
    show = args.show if args.show is not None else full[-1:]
    for T in show:
        p = P.get(T) or picks(R, T, args.rule)
        out += ['', f'## WAN (s/query) at MRR@10 {">=" if args.rule == "atleast" else "="} {T:.2f}  '
                     '(MRR@10 of the pick in brackets; config below)', '',
                '| Method | ' + ' | '.join(D[d] for d in DATASETS) + ' |', '|---' * (len(DATASETS) + 1) + '|']
        for m in METHODS:
            cells = []
            for d in DATASETS:
                r, best = p[(d, m)]
                c = f"— (max {f2(best['mrr'])})" if r is None else f"{r['wan_time']:.2f} ({f2(r['mrr'])})"
                if r is not None and m in ('bins', 'tree') and wins(p, d, m, BASE):
                    c = f'**{c}**'
                cells.append(c)
            out.append(f'| {L(m)} | ' + ' | '.join(cells) + ' |')
        out += [''] + [f"- {D[d]}: " + '; '.join(f"{L(m)} `{p[(d, m)][0]['config']}`" for m in METHODS if p[(d, m)][0])
                       for d in DATASETS]
    out += ['', '## Full sweep: WAN (MRR@10 of the pick)', '',
            '| Dataset | Method | ' + ' | '.join(f'{T:.2f}' for T in ts) + ' |', '|---' * (2 + len(ts)) + '|']
    for d in DATASETS:
        for m in METHODS:
            row = []
            for T in ts:
                r, best = P[T][(d, m)]
                row.append('—' if r is None else f"{r['wan_time']:.2f} ({f2(r['mrr'])})")
            out.append(f'| {D[d]} | {L(m)} | ' + ' | '.join(row) + ' |')
    text = '\n'.join(out) + '\n'
    print(text)
    os.makedirs(g.FIGURE_DIR, exist_ok=True)
    path = os.path.join(g.FIGURE_DIR, f'mrr_sweep_k{wr.K}' + ('_window' if args.rule == 'window' else '')
                        + ('_filter_on' if args.storage_filter == 'on' else '') + '.md')
    open(path, 'w').write(text)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
