"""
Picks a handful of representative configs out of a full parameter sweep, the
same way the BM25-Tree section of the paper does.

    python select_configs.py                 # a table of picks for every method/dataset/k
    python select_configs.py --full          # ... plus every key of every picked run
    python select_configs.py -m bins -d msmarco -k 100
    python select_configs.py -m bins --fixed vec=1 -n 8
    python select_configs.py --cost comm_kb  # select on communication instead of latency

Run it directly to inspect the picks by hand: each table row is one config with
its folder, so `python load_results.py <folder>` or the folder itself under
RESULTS_DIR is one step away. --full prints every parsed key instead.

THE PROCEDURE
-------------
A "grid search" just means running every combination of the parameters (all
bs x dpb for bins, all steps x neighb for pacmann, all B x r x s x L for the
tree) and then choosing from the results. The choosing is the part worth
copying, and it goes:

1. Effectiveness E. Min-max normalise each quality metric over the pool
   (MRR and Recall by default) so both land in [0, 1], then average them:
       E = (MRR_norm + Recall_norm) / 2
2. Pareto frontier. A config is dominated if another is at least as good on
   every quality metric AND no more expensive, with one strict inequality.
   The frontier is everything not dominated: the configs where you cannot
   improve one number without giving up another.
3. Effectiveness-cost envelope. The same idea in the (E, cost) plane.
4. Knee point. Normalise cost on a log scale and E linearly, both to [0, 1],
   and take the envelope point that maximises (y - x): the best effectiveness
   per unit of cost, i.e. where the curve stops paying off.
5. The picks, in order: the knee (C1), the cheapest frontier config reaching
   90% of the best E (C2), the same at 95% (C3), the best MRR (C4) and the
   best Recall (C5). Duplicates are dropped and the list is topped up from the
   envelope by descending E, so you always get N configs when N exist.

COST
----
The paper uses PIR calls per query. That count isn't in metadata.json for
every method, so COST_KEY defaults to per-query LAN latency, which every
method reports. Set it to 'pir_rounds' for tree-only plots, or to 'comm_kb' /
'preproc_rounds' if you want to select on those instead.
"""

import logging
import math

import globals as g
import plot_utils as pu

log = logging.getLogger('select')

COST_KEY = 'lan_time'                  # the "cheaper is better" axis
EFFECT_KEYS = ['mrr', 'recall']        # averaged into E after min-max
QUALITY_KEYS = ['mrr', 'recall']       # dominance test in step 2
THRESHOLDS = [0.90, 0.95]              # C2 and C3
N_BEST = 5                             # how many configs to keep


# ==========================================
# THE PIECES
# ==========================================

def usable(runs, keys):
    """Runs that have a real number for every key (a NaN can't be normalised)."""
    good = [r for r in runs if all(pu.is_number(r.get(key)) for key in keys)]
    if len(good) < len(runs):
        log.debug("dropped %d of %d runs missing one of %s", len(runs) - len(good), len(runs), keys)
    return good


def minmax(values):
    """Values scaled to [0, 1]. A flat list becomes all zeros."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def effectiveness(runs, effect_keys=EFFECT_KEYS):
    """E per run: the mean of the min-max normalised quality metrics."""
    columns = [minmax([r[key] for r in runs]) for key in effect_keys]
    return [sum(col[i] for col in columns) / len(columns) for i in range(len(runs))]


def pareto_front(runs, cost_key=COST_KEY, quality_keys=QUALITY_KEYS):
    """Runs not dominated on (quality_keys higher, cost_key lower)."""
    front = []
    for run in runs:
        dominated = any(
            all(other[key] >= run[key] for key in quality_keys)
            and other[cost_key] <= run[cost_key]
            and (any(other[key] > run[key] for key in quality_keys)
                 or other[cost_key] < run[cost_key])
            for other in runs if other is not run)
        if not dominated:
            front.append(run)
    return front


def envelope(runs, scores, cost_key=COST_KEY):
    """The same test in the (E, cost) plane. Returns [(run, E), ...]."""
    pairs = list(zip(runs, scores))
    return [(run, e) for run, e in pairs
            if not any(oe >= e and other[cost_key] <= run[cost_key]
                       and (oe > e or other[cost_key] < run[cost_key])
                       for other, oe in pairs if other is not run)]


def knee_point(pairs, cost_key=COST_KEY):
    """
    Kneedle: normalise log(cost) and E to [0, 1] over the envelope, then take
    the point with the largest (y - x). Costs must be positive for the log.
    """
    pairs = [(run, e) for run, e in pairs if run[cost_key] > 0]
    if not pairs:
        return None
    xs = minmax([math.log(run[cost_key]) for run, _ in pairs])
    ys = minmax([e for _, e in pairs])
    return max(zip(pairs, xs, ys), key=lambda item: item[2] - item[1])[0][0]


# ==========================================
# THE SELECTION
# ==========================================

def select_with_reasons(runs, n=N_BEST, cost_key=COST_KEY, effect_keys=EFFECT_KEYS,
                        quality_keys=QUALITY_KEYS, thresholds=THRESHOLDS, name=''):
    """
    Like select_configs(), but returns [(reason, run, E), ...] instead of just
    the runs, so a caller can show *why* each one was picked (used by the
    table this file prints when run directly). Each pick is also logged.
    """
    pool = usable(runs, set(effect_keys) | set(quality_keys) | {cost_key})
    if not pool:
        pu.warn_once("%s: no runs with all of %s, nothing to select",
                     name, sorted(set(effect_keys) | {cost_key}))
        return []
    if len(pool) <= n:
        log.info("%s: only %d candidate runs, keeping all of them", name, len(pool))
        scores = dict(zip(map(id, pool), effectiveness(pool, effect_keys)))
        return [('(all)', run, scores[id(run)]) for run in pool]

    scores = dict(zip(map(id, pool), effectiveness(pool, effect_keys)))
    front = pareto_front(pool, cost_key, quality_keys)
    env = envelope(pool, [scores[id(r)] for r in pool], cost_key)
    best_e = max(scores.values())

    picks = []                     # [(reason, run), ...], duplicates dropped below
    knee = knee_point(env, cost_key)
    if knee is not None:
        picks.append(('knee', knee))
    for threshold in thresholds:
        reaching = [r for r in front if scores[id(r)] >= threshold * best_e]
        if reaching:
            picks.append((f'E>={threshold:.0%}', min(reaching, key=lambda r: r[cost_key])))
    for key in effect_keys:
        picks.append((f'best {key}', max(pool, key=lambda r: r[key])))
    # Top up with the most effective envelope points not already chosen.
    picks += [('envelope', r) for r, _ in sorted(env, key=lambda p: -p[1])]

    chosen, seen = [], set()
    for reason, run in picks:
        if id(run) in seen:
            continue
        seen.add(id(run))
        chosen.append((reason, run, scores[id(run)]))
        log.info("%s: C%d %-10s %-22s E=%.3f %s=%.4g mrr=%.4f recall=%.4f",
                 name, len(chosen), reason, run['config'], scores[id(run)],
                 cost_key, run[cost_key], run['mrr'], run['recall'])
        if len(chosen) == n:
            break
    return chosen


def select_configs(runs, n=N_BEST, cost_key=COST_KEY, effect_keys=EFFECT_KEYS,
                   quality_keys=QUALITY_KEYS, thresholds=THRESHOLDS, name=''):
    """
    The n representative configs, in pick order (knee first). Returns fewer
    than n only when fewer distinct runs are available.
    """
    picks = select_with_reasons(runs, n, cost_key, effect_keys, quality_keys, thresholds, name)
    return [run for _, run, _ in picks]


def select_from_nested(nested_data, method, dataset, k=None, n=N_BEST, **fixed):
    """
    select_configs() straight off the loader's nested_data. Extra keyword
    arguments filter the pool first, e.g. vec=1 to select among single-DB
    bins runs only.
    """
    runs = pu.get_runs(nested_data, method, dataset, k=k, **fixed)
    return select_configs(runs, n=n, name=f'{method}/{dataset}/k{k}')


# ==========================================
# COMMAND LINE (inspect the picks by hand)
# ==========================================

def print_table(name, picks, cost_key, full=False):
    """One row per pick: reason, config, score, cost, quality, and the folder
    to hand to `python load_results.py <folder>` for the full metadata."""
    print(f"\n{name}")
    if not picks:
        print("  (nothing to select)")
        return
    header = f"{'#':<3} {'reason':<10} {'config':<24} {'E':>6} {cost_key:>12} {'mrr':>8} {'recall':>8}  folder"
    print(header)
    print('-' * len(header))
    for i, (reason, run, e) in enumerate(picks, 1):
        print(f"{i:<3} {reason:<10} {run['config']:<24} {e:>6.3f} {run[cost_key]:>12.4g} "
              f"{run['mrr']:>8.4f} {run['recall']:>8.4f}  {run['folder']}")
        if full:
            for key, value in sorted(run.items()):
                print(f"      {key:<24} {value}")


if __name__ == '__main__':
    import argparse

    import load_results

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--method', choices=g.METHOD_ORDER,
                        help='only this method (default: all of %(choices)s)')
    parser.add_argument('-d', '--dataset', choices=g.DATASETS,
                        help='only this dataset (default: all of %(choices)s)')
    parser.add_argument('-k', type=int, help=f'only this k (default: {g.K_ABLATION} and {g.K_MAIN})')
    parser.add_argument('-n', type=int, default=N_BEST, help=f'how many picks to print (default {N_BEST})')
    parser.add_argument('--cost', default=COST_KEY, metavar='KEY',
                        help=f'run key to select on, cheaper is better (default {COST_KEY})')
    parser.add_argument('--fixed', action='append', default=[], metavar='KEY=VALUE',
                        help='keep only runs where KEY==VALUE, repeatable, e.g. --fixed vec=1')
    parser.add_argument('--full', action='store_true',
                        help='also print every parsed key of each picked run')
    args = parser.parse_args()

    fixed = {}
    for item in args.fixed:
        key, _, raw = item.partition('=')
        try:
            value = float(raw)
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = raw
        fixed[key] = value

    g.setup_logging()
    data = load_results.load_results()

    methods = [args.method] if args.method else g.METHOD_ORDER
    datasets = [args.dataset] if args.dataset else g.DATASETS
    ks = [args.k] if args.k is not None else [g.K_ABLATION, g.K_MAIN]

    for method in methods:
        for dataset in datasets:
            for k in ks:
                runs = pu.get_runs(data, method, dataset, k=k, **fixed)
                picks = select_with_reasons(runs, n=args.n, cost_key=args.cost,
                                            name=f'{method}/{dataset}/k{k}')
                print_table(f'{method}/{dataset}/k{k}', picks, args.cost, full=args.full)
