"""
PILLAR-Bin before / after the stemmed-index fix (2026-11-13): the old runs (bluge index analyzed with Bluge's
standard analyzer, so stemmed vocabulary words found few or no documents) against the reruns with the index built
by the same strictEnglishAnalyzer as the vocabulary.

    python compare_bins_rerun.py                                   # old = datasets/results_simulated_old,
                                                                   # new = datasets/results_simulated
    python compare_bins_rerun.py --old A --new B -k 10 --targets 0.05 0.1 0.2

WORKFLOW (how this is meant to be used; for you or an agent picking this up later)
  1. The pre-fix bins results and Bluge indexes are kept ONLY on this machine (hopper ran out of space, so they were
     deleted there). Before syncing any new results down, move the old ones aside locally, once:
         cd <datasets>        # the folder holding results_simulated/ and index_*/
         mkdir results_simulated_old && mv results_simulated/bins_* results_simulated_old/
     (pacmann_* / tree_* stay in results_simulated: only PILLAR-Bin changed. Keep the old local index_*/ folders too:
     they are the pre-fix indexes, needed to reproduce the old numbers.)
  2. On hopper, the reruns write to /scratch/dblackle/results_simulated/bins_*. Sync those down into the local
     results_simulated/ (not results_simulated_old/).
  3. python compare_bins_rerun.py      -> figures/compare_bins_rerun_k10.md (+ .csv). Rerun it whenever more runs land:
     only configs present on both sides are compared, and "Rerun / old" shows how far the reruns are.
  4. If the new numbers are worse or broken, the old results are still in results_simulated_old/ (and the old
     indexes in index_*/): move them back to undo.
  What to look at: "Better / same / worse" should be nearly all better or same; anything in "Biggest MRR@10 drops"
  means something besides the index changed (or a run is broken). Client storage / comm / WAN ratios should stay
  near 1.00x (bins get a little fuller at large bs).
  Check on scifact before the rerun (8 configs, same binary, old vs new index): MRR@10 +0.03 to +0.08 everywhere.

Configs are matched by (dataset, config, k). Per dataset (and per layout vec0 / vec1 / text): how many configs are
rerun so far, mean / median change in MRR@10 and Recall@10, how many got better / worse, the best MRR@10 before and
after, and the median ratio new / old of client storage, comm and WAN. Then, per MRR@10 target, the smallest client
storage reaching it before and after among the matched configs (what the scale tables pick), and the biggest MRR@10 drops (worth a look: a
drop means something besides the index changed, or a run is broken).

No client storage filter (every run on both sides is compared). Writes figures/compare_bins_rerun_k<K>.md and a CSV
of every matched config.
"""

import argparse
import csv
import os
import statistics as st

import globals as g
import load_results
import plot_utils as pu

_DS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../datasets'))
OLD_DIR = os.path.join(_DS, 'results_simulated_old')
NEW_DIR = os.path.join(_DS, 'results_simulated')
LAYOUTS = {0: 'vec0', 1: 'vec1', 2: 'text'}
SAME = 1e-3   # |change in MRR@10| below this counts as unchanged


def load_bins(d, k):
    nd = load_results.load_results(simulated_dirs=[d], only_simulated=True, filter_client=False)
    return {(r['dataset'], r['config']): r for rs in nd.get('bins', {}).values() for r in rs
            if r['k'] == k and pu.is_number(r.get('mrr'))}


def f(v, spec='.4f'):
    return '-' if v is None or not pu.is_number(v) else format(v, spec)


def med(xs):
    xs = [x for x in xs if pu.is_number(x)]
    return st.median(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if pu.is_number(x)]
    return st.fmean(xs) if xs else None


def ratio(n, o, key):
    a, b = n.get(key), o.get(key)
    return a / b if pu.is_number(a) and pu.is_number(b) and b else None


def summary_row(label, pairs, n_old):
    dm = [n['mrr'] - o['mrr'] for o, n in pairs]
    dr = [n['recall'] - o['recall'] for o, n in pairs if pu.is_number(n.get('recall')) and pu.is_number(o.get('recall'))]
    up = sum(d > SAME for d in dm)
    down = sum(d < -SAME for d in dm)
    best_o = max((o['mrr'] for o, _ in pairs), default=None)
    best_n = max((n['mrr'] for _, n in pairs), default=None)
    return (f'| {label} | {len(pairs)} / {n_old} | {f(mean(dm), "+.4f")} | {f(med(dm), "+.4f")} | '
            f'{f(mean(dr), "+.4f")} | {up} / {len(pairs) - up - down} / {down} | {f(best_o)} -> {f(best_n)} | '
            f'{f(med([ratio(n, o, "client_storage_mb") for o, n in pairs]), ".2f")}x | '
            f'{f(med([ratio(n, o, "comm_kb") for o, n in pairs]), ".2f")}x | '
            f'{f(med([ratio(n, o, "wan_time") for o, n in pairs]), ".2f")}x |')


def cheapest(runs, t):
    ok = [r for r in runs if r['mrr'] >= t and pu.is_number(r.get('client_storage_mb'))]
    return min(ok, key=lambda r: (r['client_storage_mb'], r['wan_time'])) if ok else None


def compare(old_dir=OLD_DIR, new_dir=NEW_DIR, k=g.K_MAIN, targets=(0.05, 0.10, 0.20), write=True):
    old, new = load_bins(old_dir, k), load_bins(new_dir, k)
    keys = sorted(set(old) & set(new))
    datasets = [d for d in g.ALL_DATASETS if any(key[0] == d for key in old) or any(key[0] == d for key in new)]
    datasets += sorted({key[0] for key in list(old) + list(new)} - set(datasets))

    out = [f'# PILLAR-Bin before / after the stemmed-index fix (k={k})', '',
           f'old: `{old_dir}` ({len(old)} runs), new: `{new_dir}` ({len(new)} runs), matched: {len(keys)}. '
           f'Changes are new - old; better / same / worse at |change in MRR@10| > {SAME}. Ratios are medians of '
           f'new / old over the matched configs. No client storage filter.', '',
           '## Per dataset', '',
           '| Dataset | Rerun / old | Mean change in MRR@10 | Median change in MRR@10 | Mean change in Recall@10 | '
           'Better / same / worse | Best MRR@10 (old -> new) | Client storage | Comm | WAN |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for d in datasets:
        pairs = [(old[key], new[key]) for key in keys if key[0] == d]
        n_old = sum(key[0] == d for key in old)
        out.append(summary_row(g.DATASET_LABELS.get(d, d), pairs, n_old) if pairs else
                   f'| {g.DATASET_LABELS.get(d, d)} | 0 / {n_old} | not rerun yet | | | | | | | |')

    out += ['', '## Per dataset and layout', '',
            '| Dataset | Layout | Rerun / old | Mean change in MRR@10 | Median change in MRR@10 | '
            'Mean change in Recall@10 | Better / same / worse | Best MRR@10 (old -> new) | Client storage | Comm | WAN |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for d in datasets:
        for v, lay in LAYOUTS.items():
            pairs = [(old[key], new[key]) for key in keys if key[0] == d and old[key]['vec'] == v]
            n_old = sum(key[0] == d and r['vec'] == v for key, r in old.items())
            if pairs:
                out.append(summary_row(f'{g.DATASET_LABELS.get(d, d)} | {lay}', pairs, n_old))

    out += ['', '## Smallest client storage reaching an MRR@10 target', '',
            'As the scale tables pick them, over the matched configs only (so a half-finished rerun compares like '
            'with like).', '',
            '| Dataset | Target | Old config | Old MB (MRR@10) | New config | New MB (MRR@10) | New / old |',
            '|---|---|---|---|---|---|---|']
    for d in datasets:
        ro = [old[key] for key in keys if key[0] == d]
        rn = [new[key] for key in keys if key[0] == d]
        if not ro:
            continue
        for t in targets:
            a, b = cheapest(ro, t), cheapest(rn, t)
            cell = lambda r: (f'`{r["config"]}`', f'{r["client_storage_mb"]:,.1f} ({r["mrr"]:.3f})') if r else \
                ('does not reach it', '-')
            ca, cb = cell(a), cell(b)
            rat = b['client_storage_mb'] / a['client_storage_mb'] if a and b and a['client_storage_mb'] else None
            out.append(f'| {g.DATASET_LABELS.get(d, d)} | {t:.2f} | {ca[0]} | {ca[1]} | {cb[0]} | {cb[1]} | '
                       f'{f(rat, ".2f") + ("x" if rat else "")} |')

    drops = sorted(((new[key]['mrr'] - old[key]['mrr'], key) for key in keys), key=lambda x: x[0])[:15]
    drops = [x for x in drops if x[0] < -SAME]
    out += ['', '## Biggest MRR@10 drops', '']
    if drops:
        out += ['| Dataset | Config | Old MRR@10 | New MRR@10 | Change |', '|---|---|---|---|---|']
        out += [f'| {g.DATASET_LABELS.get(key[0], key[0])} | `{key[1]}` | {old[key]["mrr"]:.4f} | '
                f'{new[key]["mrr"]:.4f} | {dm:+.4f} |' for dm, key in drops]
    else:
        out.append(f'None: no matched config lost more than {SAME} MRR@10.')

    text = '\n'.join(out) + '\n'
    print(text)
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'compare_bins_rerun_k{k}')
        open(base + '.md', 'w').write(text)
        cols = ['mrr', 'recall', 'client_storage_mb', 'comm_kb', 'wan_time', 'db_size_mb', 'empty_bins']
        with open(base + '.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['dataset', 'config', 'layout'] + [f'{c}_{s}' for c in cols for s in ('old', 'new')] +
                       ['mrr_change'])
            for key in keys:
                o, n = old[key], new[key]
                w.writerow([key[0], key[1], LAYOUTS.get(o['vec'], o['vec'])] +
                           [r.get(c) for c in cols for r in (o, n)] + [n['mrr'] - o['mrr']])
        print(f'saved {base}.md / .csv')
    return text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', default=OLD_DIR, help=f'results before the fix (default {OLD_DIR})')
    ap.add_argument('--new', default=NEW_DIR, help=f'results after the fix (default {NEW_DIR})')
    ap.add_argument('-k', type=int, default=g.K_MAIN)
    ap.add_argument('--targets', type=float, nargs='+', default=[0.05, 0.10, 0.20], help='MRR@10 targets')
    args = ap.parse_args()
    g.setup_logging()
    compare(args.old, args.new, args.k, args.targets)
