"""
Which sweep values can be dropped without losing anything? For one method, every value of every parameter is
removed in turn (all runs with that value), and the Pareto frontiers are rebuilt: best quality at or below each
cost, for every (quality, cost) pair below, per dataset. The loss of a removal is the largest relative quality drop
at any point of any of those frontiers. A value is "removable" when that loss stays within TOL on every dataset.
Removable values are then removed TOGETHER, greedily (most runs saved first), keeping only those whose combined
removal still stays within TOL, because two values can each be redundant only thanks to the other.

    python useless_params.py bins            # or tree
    python useless_params.py tree --tol 0.02 --datasets msmarco nq fever
    python useless_params.py bins --show bs  # the per-dataset losses of every bs value

Results come from load_results (simulated runs by default), k = g.K_MAIN, the vec0 batch/rounds sweep and the bins
client-storage filter left out (the filter would hide whole regions of a frontier).
"""

import argparse
import math
from collections import defaultdict

import globals as g
import load_results

QUALITY = ['recall', 'mrr']
COSTS = ['client_storage_mb', 'wan_time', 'lan_time', 'comm_kb']
PARAMS = {'bins': ['vec', 'bs', 'dpb'], 'tree': ['b', 'r', 's', 'L', 'q', 'k_candidates']}
TOL = 0.01   # relative quality drop allowed anywhere on a frontier
MIN_FRAC = 0.0   # frontier points below this share of the dataset's best quality don't count (--min-frac)


def num(x):
    return isinstance(x, (int, float)) and not math.isnan(x)


def frontier(runs, q, c):
    """[(cost, quality)] of the runs nothing else beats on both, sorted by cost."""
    pts = sorted(((r[c], -r[q]) for r in runs if num(r.get(c)) and num(r.get(q))))
    out, best = [], -math.inf
    for cost, nq in pts:
        if -nq > best:
            out.append((cost, -nq)); best = -nq
    return out


def best_at(front, cost):
    """Best quality at a cost <= `cost` on a frontier (0 if none)."""
    b = 0.0
    for c, qv in front:
        if c > cost:
            break
        b = qv
    return b


def loss(full, kept):
    """Largest relative quality drop at any frontier point of `full`, over every (quality, cost) pair, and where.
    Points below MIN_FRAC x the best quality of that frontier are ignored (the near-useless cheap end)."""
    worst, where = 0.0, None
    for q in QUALITY:
        for c in COSTS:
            ff, fk = frontier(full, q, c), frontier(kept, q, c)
            top = max((qv for _, qv in ff), default=0)
            for cost, qv in ff:
                if qv <= 0 or qv < MIN_FRAC * top:
                    continue
                drop = (qv - best_at(fk, cost)) / qv
                if drop > worst:
                    worst, where = drop, (q, c, cost, qv, best_at(fk, cost))
    return worst, where


def main():
    global MIN_FRAC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('method', choices=list(PARAMS))
    ap.add_argument('--tol', type=float, default=TOL)
    ap.add_argument('--datasets', nargs='+')
    ap.add_argument('--min-runs', type=int, default=40, help='skip datasets with fewer runs than this (partial sweeps)')
    ap.add_argument('--show', help='print every value of this parameter with its loss per dataset')
    ap.add_argument('--min-frac', type=float, default=MIN_FRAC,
                    help="ignore frontier points below this share of the frontier's best quality (e.g. 0.25)")
    ap.add_argument('--pairs', action='store_true', help='also try removing pairs of values (p1=v1 AND p2=v2)')
    ap.add_argument('--loo', action='store_true', help='validate: choose removals without each dataset, test on it')
    args = ap.parse_args()
    MIN_FRAC = args.min_frac

    g.INCLUDE_VEC0_SWEEP = False
    g.setup_logging()
    nd = load_results.load_results(filter_client=False)
    per_ds = {d: [r for r in rs if r['k'] == g.K_MAIN] for d, rs in nd.get(args.method, {}).items()}
    per_ds = {d: rs for d, rs in per_ds.items() if (not args.datasets or d in args.datasets) and len(rs) >= args.min_runs}
    print(f"{args.method}: datasets {', '.join(f'{d} ({len(rs)})' for d, rs in sorted(per_ds.items()))}; tolerance {args.tol:.1%}\n")

    if args.loo:
        return leave_one_out(per_ds, args)
    removed, kept, cand = pick(per_ds, args)
    report(per_ds, removed, kept, cand)


def units(per_ds, args):
    """Candidate removals: (p, v) per parameter value, plus (p1, p2), (v1, v2) cells with --pairs."""
    out = []
    ps = PARAMS[args.method]
    for p in ps:
        for v in sorted({r.get(p) for rs in per_ds.values() for r in rs if r.get(p) is not None}):
            out.append((p, v))
    if args.pairs:
        for i, p1 in enumerate(ps):
            for p2 in ps[i + 1:]:
                for v1, v2 in sorted({(r.get(p1), r.get(p2)) for rs in per_ds.values() for r in rs
                                      if r.get(p1) is not None and r.get(p2) is not None}):
                    out.append(((p1, p2), (v1, v2)))
    return out


def hit(r, p, v):
    return all(r.get(pp) == vv for pp, vv in zip(p, v)) if isinstance(p, tuple) else r.get(p) == v


def name(p, v):
    return '&'.join(f'{pp}={vv}' for pp, vv in zip(p, v)) if isinstance(p, tuple) else f'{p}={v}'


def pick(per_ds, args):
    """(removed units, kept runs per dataset, single-removal candidates) chosen on per_ds."""
    cand = []
    for p, v in units(per_ds, args):
        worst = [loss(rs, [r for r in rs if not hit(r, p, v)])[0] for rs in per_ds.values() if any(hit(r, p, v) for r in rs)]
        if worst and max(worst) <= args.tol:
            cand.append((sum(1 for rs in per_ds.values() for r in rs if hit(r, p, v)), p, v))
    removed, kept = [], {d: list(rs) for d, rs in per_ds.items()}
    for saved, p, v in sorted(cand, key=lambda x: -x[0]):
        trial = {d: [r for r in rs if not hit(r, p, v)] for d, rs in kept.items()}
        if max(loss(per_ds[d], trial[d])[0] for d in per_ds) <= args.tol:
            removed.append((p, v)); kept = trial
    return removed, kept, cand


def report(per_ds, removed, kept, cand):
    print(f"removable one at a time (loss <= tol on every dataset): {len(cand)}")
    print("removable together:", ', '.join(name(p, v) for p, v in removed) or 'none')
    tot = sum(len(rs) for rs in per_ds.values()); left = sum(len(rs) for rs in kept.values())
    print(f"runs: {tot} -> {left} ({1 - left / tot:.0%} fewer)")
    for d in sorted(per_ds):
        w, where = loss(per_ds[d], kept[d])
        print(f"  {d:16} {len(per_ds[d]):4} -> {len(kept[d]):4} runs, worst frontier drop {w:.2%}"
              + (f" ({where[0]} vs {where[1]} at {where[2]:.4g}: {where[3]:.4f} -> {where[4]:.4f})" if where else ''))


def leave_one_out(per_ds, args):
    """For each dataset: pick the removals on the others, apply them to it, report its frontier loss."""
    print("leave-one-dataset-out: removals chosen on the other datasets, applied to the held-out one")
    for d in sorted(per_ds):
        others = {o: rs for o, rs in per_ds.items() if o != d}
        removed, _, _ = pick(others, args)
        rs = per_ds[d]
        kept = [r for r in rs if not any(hit(r, p, v) for p, v in removed)]
        w, where = loss(rs, kept)
        print(f"  held out {d:16} {len(rs):4} -> {len(kept):4} runs ({1 - len(kept) / len(rs):.0%} fewer), "
              f"worst frontier drop {w:.2%}" + (f" ({where[0]} vs {where[1]} at {where[2]:.4g}: {where[3]:.4f} -> {where[4]:.4f})" if where else ''))


def _unused():
    # 1. one value at a time
    cand = []
    for p in PARAMS[args.method]:
        values = sorted({r.get(p) for rs in per_ds.values() for r in rs if r.get(p) is not None})
        for v in values:
            worst = {}
            for d, rs in per_ds.items():
                if not any(r.get(p) == v for r in rs):
                    continue
                worst[d] = loss(rs, [r for r in rs if r.get(p) != v])
            if not worst:
                continue
            mx = max(w for w, _ in worst.values())
            saved = sum(1 for rs in per_ds.values() for r in rs if r.get(p) == v)
            if args.show == p:
                print(f"  {p}={v}: max loss {mx:.2%}  " + ', '.join(f"{d} {w:.1%}" for d, (w, _) in sorted(worst.items())))
            if mx <= args.tol:
                cand.append((saved, p, v))
    if args.pairs:   # combinations: the runs with p1=v1 and p2=v2, e.g. vec=0 and dpb=10
        ps = PARAMS[args.method]
        for i, p1 in enumerate(ps):
            for p2 in ps[i + 1:]:
                combos = sorted({(r.get(p1), r.get(p2)) for rs in per_ds.values() for r in rs
                                 if r.get(p1) is not None and r.get(p2) is not None})
                for v1, v2 in combos:
                    hit = lambda r: r.get(p1) == v1 and r.get(p2) == v2
                    worst = [loss(rs, [r for r in rs if not hit(r)])[0] for rs in per_ds.values() if any(hit(r) for r in rs)]
                    if worst and max(worst) <= args.tol:
                        cand.append((sum(1 for rs in per_ds.values() for r in rs if hit(r)), (p1, p2), (v1, v2)))
    if args.show:
        print()

    # 2. together, greedily
    removed, kept = [], {d: list(rs) for d, rs in per_ds.items()}
    def drop(r, p, v):
        return all(r.get(pp) == vv for pp, vv in zip(p, v)) if isinstance(p, tuple) else r.get(p) == v

    for saved, p, v in sorted(cand, key=lambda x: -x[0]):
        trial = {d: [r for r in rs if not drop(r, p, v)] for d, rs in kept.items()}
        worst = {d: loss(per_ds[d], trial[d]) for d in per_ds}
        mx = max(w for w, _ in worst.values())
        if mx <= args.tol:
            removed.append((p, v)); kept = trial
    def name(p, v):
        return '&'.join(f'{pp}={vv}' for pp, vv in zip(p, v)) if isinstance(p, tuple) else f'{p}={v}'

    print(f"removable one at a time (loss <= tol on every dataset): {len(cand)}")
    print("removable together:", ', '.join(name(p, v) for p, v in removed) or 'none')
    tot = sum(len(rs) for rs in per_ds.values()); left = sum(len(rs) for rs in kept.values())
    print(f"runs: {tot} -> {left} ({1 - left / tot:.0%} fewer)")
    for d in sorted(per_ds):
        w, where = loss(per_ds[d], kept[d])
        print(f"  {d:16} {len(per_ds[d]):4} -> {len(kept[d]):4} runs, worst frontier drop {w:.2%}"
              + (f" ({where[0]} vs {where[1]} at {where[2]:.4g}: {where[3]:.4f} -> {where[4]:.4f})" if where else ''))


if __name__ == '__main__':
    main()
