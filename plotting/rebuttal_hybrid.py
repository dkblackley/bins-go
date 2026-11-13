"""
"Is PILLAR just Coeus + Tiptoe?" -- Tiptoe, Coeus and a naive Tiptoe+Coeus hybrid against
PILLAR-Bin and PILLAR-Tree, at one frozen MRR@10, one frozen Recall@10 and one WAN budget
(the same freezes as frozen_values.py, which does PACMANN vs Bin / Tree).

    python rebuttal_hybrid.py                          # every dataset with Tiptoe, Coeus and the PILLAR method
    python rebuttal_hybrid.py -d scifact nq
    python rebuttal_hybrid.py --mrr 0.40 --recall 0.55 --wan 0.4
    python rebuttal_hybrid.py --lists reranked         # merge the cosine-reranked lists instead

THE HYBRID. A Tiptoe config and a Coeus config are run one after the other. Their returned lists are
merged round robin (t1, c1, t2, c2, ..., duplicates skipped) and cut to k (10). Both orders are tried
(T>C starts with Tiptoe, C>T with Coeus) and each counts as a candidate, which helps the hybrid. Cost
of a hybrid = the sum of the two runs: WAN, LAN, communication, client storage. MRR@10 / Recall@10 are
recomputed from the per-query lists with the formulas baselines/common.py uses (calc_mrr / calc_recall).
Every Tiptoe x Coeus pair (above the MRR floor) is a hybrid candidate.

--lists returned (the default) merges results.json, the lists Tiptoe and Coeus return, which is also
what their MRR in the paper is (g.MRR_FROM_PRE_RERANK). --lists reranked merges results_reRank.json
(after the local cosine rerank) and uses the reranked MRR for the Tiptoe / Coeus rows too.

THE FREEZES, one set per PILLAR method (Bin, Tree), over Tiptoe, Coeus, Hybrid and PILLAR-X, each dataset at
its own values. Within one row of the headline only the method changes:
  Frozen MRR@10 / Recall@10  T = the highest value (rounded down to 2 dp, matched within +/- --tol) that all
                             four have a run at on that dataset; each takes its matching run with the lowest
                             WAN (--pick-by). The hybrid's pick is the cheapest PAIR whose merged list hits T.
  WAN budget                 W = the lowest budget (0.1 s steps) under which all four have a run on that
                             dataset (--wan to choose one); each takes its best MRR@10 within W. A hybrid
                             counts its summed WAN.
A freeze with no value all four reach on a dataset is not shown (listed as "no shared value"). Datasets
where one of the methods has no runs at all are listed and skipped.
Under each frozen score table: "picks merged", the hybrid of exactly the Tiptoe and Coeus configs that
table picked (better of the two orders on MRR@10) -- it is not at T, it shows what merging those two gives.

OFFLINE COSTS (left out of WAN / LAN for every method by default, as in the paper's other plots). Tiptoe
downloads a single-use token per query (34-44 MB, ~98% of its bytes; counted in comm_kb, not in WAN / LAN).
PILLAR / PACMANN's Piano hints run out after ~sqrt(N) ln N lookups per partition and every refresh streams the
whole DB to the client again; the tree's refreshes alone come to ~2-4 MB per query on SciFact and 60-250 MB
per query on NQ / FEVER / DBPedia / MS MARCO (Stage DB size x (refreshes - 1) / queries), i.e. as much as
Tiptoe's token or more. --tiptoe-token adds Tiptoe's token to its WAN / LAN alone (one-sided; for reference
only). Coeus: all its per-query traffic is in its WAN already (its Galois keys are a one-off setup, counted
once); its server time is not counted (CoeusTimeSource none), which flatters it.

Runs are filtered like frozen_values.py (MRR@10 floor --min-mrr; nothing else). Tiptoe's WAN is the
online part only (its per-query token is fetched offline), as everywhere else in the paper.

Inputs: load_results() (metadata.json; Tiptoe / Coeus are read from g.RESULTS_DIRS if it found none of
them, e.g. with ONLY_SIMULATED on hopper), results.json / results_reRank.json of every Tiptoe and Coeus
run, and the qrels the baselines used (g.DATASETS_ROOT/<d>/qrels/test.tsv, qrels.dev.tsv for msmarco;
also tried next to the results folders, e.g. /scratch/dblackle/datasets;
fallback g.TREE_DATA_DIR/<d>_tree/qrels.trec). Each run's recomputed MRR is checked against its
metadata.json (mismatches are reported: a sign of the wrong qrels). Missing per-query files are listed
at the end with an rsync command.

Writes figures/rebuttal_hybrid_k<K>.md and .json.
"""

import argparse
import json
import logging
import math
import os
from collections import defaultdict

import globals as g
import load_results
import plot_utils as pu
import frozen_values as fv

log = logging.getLogger('rebuttal_hybrid')

GROUPS = [('PILLAR-Bin', ['bins']), ('PILLAR-Tree', ['tree'])]
PILLARS = sorted({m for _, ms in GROUPS for m in ms})
BASE = ['tiptoe', 'coeus']
HYB = 'hybrid'
g.METHOD_LABELS.setdefault(HYB, 'Tiptoe+Coeus (RR merge)')
LIST_FILE = {'returned': 'results.json', 'reranked': 'results_reRank.json'}
SUM_KEYS = ['wan_time', 'lan_time', 'comm_kb', 'client_storage_mb']
CHECK_TOL = 5e-4   # recomputed MRR vs metadata.json (written with 4 dp)
NEEDED = defaultdict(set)

L, D, r2 = fv.L, fv.D, fv.r2


# ------------------------------------------------------------------------------------------ per-query metrics

_QRELS = {}


def qrels(dataset):
    """qid -> set of relevant doc ids (score > 0), the file baselines/common.py uses."""
    if dataset not in _QRELS:
        name = 'qrels.dev.tsv' if dataset == 'msmarco' else 'test.tsv'
        # g.DATASETS_ROOT, then a datasets/ folder next to the results folders (hopper: /scratch/dblackle/datasets)
        roots = [g.DATASETS_ROOT]
        for rd in g.RESULTS_DIRS + g.SIMULATED_RESULTS_DIRS:
            roots += [os.path.dirname(os.path.abspath(rd)), os.path.join(os.path.dirname(os.path.abspath(rd)), 'datasets')]
        paths = list(dict.fromkeys(os.path.join(r, dataset, 'qrels', name) for r in roots))
        paths.append(os.path.join(g.TREE_DATA_DIR, f'{dataset}_tree', 'qrels.trec'))
        path = next((p for p in paths if os.path.exists(p)), None)
        rel = defaultdict(set)
        if path is None:
            log.error("no qrels for %s (looked for %s)", dataset, ', '.join(paths))
        else:
            for line in open(path):
                p = line.split()
                if len(p) < 3:
                    continue
                qid, doc, score = (p[0], p[2], p[3]) if len(p) > 3 else (p[0], p[1], p[2])
                if score.lstrip('-').isdigit() and int(score) > 0:
                    rel[qid].add(doc)
            log.info("qrels %s: %s (%d queries)", dataset, path, len(rel))
        _QRELS[dataset] = (rel, path)
    return _QRELS[dataset][0]


def mrr_recall(results, rel):
    """calc_mrr / calc_recall of baselines/common.py: mean over the queries that have qrels."""
    mrr = rec = 0.0
    n = 0
    for qid, ranked in results.items():
        r = rel.get(qid)
        if not r:
            continue
        for i, d in enumerate(ranked):
            if d in r:
                mrr += 1.0 / (i + 1)
                break
        rec += len(set(ranked) & r) / len(r)
        n += 1
    return (mrr / n, rec / n) if n else (math.nan, math.nan)


def round_robin(a, b, k):
    out, seen = [], set()
    for i in range(max(len(a), len(b))):
        for lst in (a, b):
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i])
                out.append(lst[i])
                if len(out) == k:
                    return out
    return out


_LISTS = {}   # (results_dir, folder, file) -> (key of identical-list group, {qid: [doc, ...]}) or None


def lists(run, which):
    p = os.path.join(run['results_dir'], run['folder'], LIST_FILE[which])
    if p not in _LISTS:
        if not os.path.exists(p):
            NEEDED[run['results_dir']].add(run['folder'])
            _LISTS[p] = None
        else:
            res = json.load(open(p))
            # configs that only change cost (e.g. Tiptoe's hint size) return the same lists: merge them once
            _LISTS[p] = (hash(json.dumps(res, sort_keys=True)), res)
    return _LISTS[p]


def check_run(run, which, rel):
    """recomputed MRR vs the one in metadata.json; returns the difference (or None if no file)"""
    got = lists(run, which)
    if got is None:
        return None
    mrr, _ = mrr_recall(got[1], rel)
    want = run['mrr_reranked'] if which == 'reranked' else run['mrr_pre_rerank']
    return mrr - want if pu.is_number(want) else None


# ------------------------------------------------------------------------------------------------- hybrids

def hybrids(by, dataset, which, k):
    """every Tiptoe x Coeus pair (x 2 orders) on this dataset as a run dict, plus the qids used"""
    rel = qrels(dataset)
    ts = [(t, lists(t, which)) for t in by['tiptoe'][dataset]]
    cs = [(c, lists(c, which)) for c in by['coeus'][dataset]]
    ts, cs = [x for x in ts if x[1]], [x for x in cs if x[1]]
    cache, out = {}, []
    for t, (tk, tres) in ts:
        for c, (ck, cres) in cs:
            qids = [q for q in tres if q in cres]
            for order in ('T>C', 'C>T'):
                key = (tk, ck, order)
                if key not in cache:
                    a, b = (tres, cres) if order == 'T>C' else (cres, tres)
                    cache[key] = mrr_recall({q: round_robin(a[q], b[q], k) for q in qids}, rel)
                mrr, rec = cache[key]
                h = {'method': HYB, 'dataset': dataset, 'k': k, 'mrr': mrr, 'recall': rec, 'order': order,
                     'config': f"{order} T:{t['config']} + C:{c['config']}",
                     'folder': f"{t['folder']} + {c['folder']}",
                     'source': 'simulated',   # Tiptoe and Coeus are always simulated
                     'tiptoe': t, 'coeus': c}
                for key2 in SUM_KEYS:
                    x, y = t.get(key2), c.get(key2)
                    h[key2] = x + y if pu.is_number(x) and pu.is_number(y) else math.nan
                out.append(h)
    log.info("%s: %d hybrid candidates from %d Tiptoe x %d Coeus runs with per-query lists (%d distinct merges)",
             dataset, len(out), len(ts), len(cs), len(cache))
    return out


def merged_pick(tp, cp, by_hyb):
    """the hybrid of exactly these two runs, better order on MRR@10"""
    if not (tp and cp):
        return None
    hs = [h for h in by_hyb if h['tiptoe'] is tp and h['coeus'] is cp]
    return max(hs, key=lambda h: (h['mrr'], h['recall'])) if hs else None


def add_tiptoe_token(nd):
    """WAN / LAN of the per-query token (TiptoeOfflineWANTime / TiptoeOfflineLANTime) added to Tiptoe's runs"""
    n = 0
    for rs in nd.get('tiptoe', {}).values():
        for r in rs:
            try:
                meta = json.load(open(os.path.join(r['results_dir'], r['folder'], 'metadata.json')))
                nq = r['num_queries']
                r['wan_time_online'], r['lan_time_online'] = r['wan_time'], r['lan_time']
                r['wan_time'] += float(meta['TiptoeOfflineWANTime']) / nq
                r['lan_time'] += float(meta['TiptoeOfflineLANTime']) / nq
                n += 1
            except (OSError, KeyError, ValueError) as err:
                log.warning("%s: no token time (%s), Tiptoe WAN/LAN left online-only", r['folder'], err)
    log.info("Tiptoe token WAN / LAN added to %d runs", n)


def add_baselines(nd):
    """Tiptoe / Coeus runs that load_results() didn't find (ONLY_SIMULATED reads results_simulated plus
    g.EXTRA_RESULTS_DIRS only, and on hopper the latter don't exist) are read from g.RESULTS_DIRS, where the
    baseline scripts write them. They are simulated runs (simulate_tiptoe.py, run_coeus.py) and marked so.
    Nothing else is taken from there: PILLAR-Bin / Tree stay exactly what load_results() gave (simulated only
    under ONLY_SIMULATED)."""
    dirs = [d for d in g.RESULTS_DIRS if os.path.isdir(d)]
    missing = [m for m in BASE if not any(r['k'] == g.K_MAIN for rs in nd.get(m, {}).values() for r in rs)]
    if not missing or not dirs:
        return
    log.info("no %s runs loaded; reading them from %s", '/'.join(missing), ', '.join(dirs))
    extra = load_results.load_results(dirs, fallback_simulated=False, only_simulated=False, filter_client=False)
    for m in missing:
        nd[m] = extra.get(m, {})
        for rs in nd[m].values():
            for r in rs:
                r['source'] = 'simulated'   # Tiptoe / Coeus runs are always simulated, wherever they are stored
        log.info("%s: %s", m, ', '.join(f'{d} {len(rs)}' for d, rs in sorted(nd[m].items())) or 'none found')


# ------------------------------------------------------------------------------------------------- report

def cell(p, kind):
    """compact cell: at a frozen score, the cost; at a WAN budget, the scores"""
    if p is None:
        return '-'
    if kind == 'score':
        return f"{p['wan_time']:.3f} s ({r2(p['mrr'])} / {r2(p['recall'])})"
    return f"{r2(p['mrr'])} / {r2(p['recall'])} ({p['wan_time']:.3f} s)"


def freeze_title(res):
    if res['kind'] == 'score':
        return f"{fv.SCORE_LABEL[res['match']]} = {fv._vt(res)}"
    return f"WAN ≤ {res['value']:.1f} s"


def run_group(by, group, datasets, args, metric, target):
    if metric == 'wan':
        return fv._wan_freeze(by, group, datasets, 'mrr', target, args.wan_tol, fv.WAN_DP, 'budget')
    return fv.exact_freeze(by, group, datasets, metric, metric, target, args.tol, pick_by=args.pick_by)


def pick_datasets(by, group, cands, metric, target, args):
    """the largest subset of cands where every method of the group has a run at ONE shared value (ties: the
    highest score / lowest WAN), and that freeze; (cands, None) if not even one dataset works"""
    from itertools import combinations
    for size in range(len(cands), 0, -1):
        best = None
        for ds in combinations(cands, size):
            res = run_group(by, group, list(ds), args, metric, target)
            if res['covered'] < res['pairs']:
                continue
            score = -res['value'] if metric == 'wan' else res['value']
            if best is None or score > best[0]:
                best = (score, list(ds), res)
        if best:
            return best[1], best[2]
    return cands, None


def report(nd, args):
    k = args.k
    datasets = args.datasets or g.ALL_DATASETS
    methods = BASE + PILLARS
    by, _, dropped = fv.load_runs(nd, methods, datasets, k, args.min_mrr, drop_dominated=False)
    if args.lists == 'reranked':   # Tiptoe / Coeus rows use the MRR of the lists that are merged
        for m in BASE:
            for d in datasets:
                by[m][d] = [dict(r, mrr=r['mrr_reranked']) for r in by[m][d] if pu.is_number(r.get('mrr_reranked'))]
    have = [d for d in datasets if all(by[m][d] for m in BASE)]
    by[HYB] = {d: [] for d in datasets}
    checks = []
    for d in have:
        rel = qrels(d)
        if not rel:   # no qrels here: the hybrid can't be scored (Tiptoe / Coeus / PILLAR still compared)
            checks.append(f"{D(d)}: NO QRELS (looked under DATASETS_ROOT={g.DATASETS_ROOT}, next to the results "
                          f"folders, and in {g.TREE_DATA_DIR}; set DATASETS_ROOT), so no hybrid on this dataset")
            continue
        diffs = [x for x in (check_run(r, args.lists, rel) for m in BASE for r in by[m][d]) if x is not None]
        bad = [x for x in diffs if abs(x) > CHECK_TOL]
        checks.append(f"{D(d)}: {len(diffs)} runs with per-query lists, {len(bad)} whose recomputed MRR@{k} differs "
                      f"from metadata.json by > {CHECK_TOL}" + (f" (max {max(map(abs, bad)):.4f}: wrong qrels?)" if bad else '')
                      + f"; qrels {_QRELS[d][1]}")
        by[HYB][d] = hybrids(by, d, args.lists, k)

    lines = [f'# Tiptoe, Coeus and a Tiptoe+Coeus hybrid vs PILLAR (k={k})', '',
             f'Hybrid = a Tiptoe run then a Coeus run, the two {"returned" if args.lists == "returned" else "reranked"} '
             f'lists merged round robin and cut to {k}; cost = sum of both (WAN, LAN, comm, client storage). '
             'Every Tiptoe x Coeus pair and both merge orders are candidates. '
             + ("Tiptoe's WAN / LAN INCLUDE its per-query token (--tiptoe-token; PILLAR's Piano refreshes are NOT "
                "included, so this is not like-for-like). " if args.tiptoe_token else
                "WAN / LAN = online rounds only for every method (Tiptoe's token and PILLAR's Piano refreshes are "
                "both offline and left out). ")
             + 'Coeus: no server time counted (lower bound). ' +
             f'Scores rounded down to {fv.DP} dp, matched within ± {args.tol:g}. '
             'Frozen score cells: WAN (MRR@10 / Recall@10), lowest WAN at that score. '
             'WAN budget cells: MRR@10 / Recall@10 (WAN), best MRR@10 within the budget.', '']
    if dropped:
        lines += [f"Runs below MRR@{k} {args.min_mrr}: " + ', '.join(f'{L(m)} {D(d)} {a}' for (m, d), (a, _) in sorted(dropped.items())), '']
    details, js, headline = [], {}, ['## Headline', '']
    targets = {'mrr': args.mrr, 'recall': args.recall, 'wan': args.wan}
    for gname, pillars in GROUPS:
        group = BASE + [HYB] + pillars
        cands = [d for d in datasets if all(by[m][d] for m in group)]
        details += [f'## {gname}: ' + ', '.join(L(m) for m in group), '',
                    f"Datasets (every method has runs, hybrid included): {', '.join(D(d) for d in cands) or 'none'}"
                    + (f"; not shown: " + '; '.join(f"{D(d)} (no {', '.join(L(m) for m in group if not by[m][d])})"
                                                    for d in datasets if d not in cands and any(by[m][d] for m in group))
                       if len(cands) < len(datasets) else '') + '.', '']
        cols = ['Tiptoe', 'Coeus', L(HYB), 'picks merged'] + [L(m) for m in pillars]
        headline += [f'### {gname}', '', '| Dataset | Freeze | ' + ' | '.join(cols) + ' |', '|' + '---|' * (2 + len(cols))]
        js[gname] = []
        for d in cands:   # every dataset at its own values; within a row only the method changes
            details += [f'### {D(d)}', '']
            for metric in ('mrr', 'recall', 'wan'):
                for t in (targets[metric] or [None]):
                    name = 'WAN budget' if metric == 'wan' else fv.SCORE_LABEL[metric]
                    _, res = pick_datasets(by, group, [d], metric, t, args)
                    if res is None:   # no value every method reaches: nothing shown
                        details += [f'#### {name}: no value that every method has a run at, not shown', '']
                        headline.append(f'| {D(d)} | {name}: no shared value | ' + ' | ' * len(cols))
                        continue
                    h, note = fv._header(res)
                    details += [h, ''] + fv._table(res, group, [d])
                    mp = None
                    if res['kind'] == 'score':
                        mp = merged_pick(res['picks'][('tiptoe', d)], res['picks'][('coeus', d)], by[HYB][d])
                        if mp:
                            details.append(f"| {D(d)} | picks merged (not at the frozen value) | `{mp['config']}` | "
                                           f"{r2(mp['mrr'])} | {r2(mp['recall'])} | {r2(mp['wan_time'])} | "
                                           f"{r2(mp.get('lan_time'))} | {fv._f(mp.get('comm_kb'), '.1f')} | "
                                           f"{fv._f(mp.get('client_storage_mb'), '.1f')} | {mp['source']} |")
                    details += [''] + fv._summary(res, group, [d])[:1] + ['']
                    pk = res['picks']
                    headline.append(f"| {D(d)} | {freeze_title(res)} | "
                                    + ' | '.join([cell(pk[(m, d)], res['kind']) for m in BASE + [HYB]]
                                                 + [cell(mp, 'score') if res['kind'] == 'score' else '']
                                                 + [cell(pk[(m, d)], res['kind']) for m in pillars]) + ' |')
                    js[gname].append({'dataset': d, 'kind': res['kind'], 'match': res['match'], 'value': res['value'],
                                      'tol': res['tol'],
                                      'picks': {m: (fv._row(pk[(m, d)]) if pk[(m, d)] else None) for m in group},
                                      'picks_merged': fv._row(mp) if mp else None})
        headline.append('')
    out = lines + headline + [''] + ['## Per-query check', ''] + [f'- {c}' for c in checks] + [''] + details
    if NEEDED:
        out += ['## Per-query files to download (then run again)', '']
        for rd, folders in NEEDED.items():
            server = '/scratch/dblackle/results_simulated/' if 'simulated' in rd else '/scratch/dblackle/results/'
            inc = ' '.join(f"--include='{fo}/' --include='{fo}/results*.json'" for fo in sorted(folders))
            out.append(f"rsync -am {inc} --exclude='*' hopper:{server} {rd}/")
            out.append('')
    text = '\n'.join(out) + '\n'
    print(text)
    if args.write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'rebuttal_hybrid_k{k}' + ('' if args.lists == 'returned' else '_reranked'))
        open(base + '.md', 'w').write(text)
        json.dump(js, open(base + '.json', 'w'), indent=2)
        print(f'saved {base}.md / .json')
    return js, text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--datasets', nargs='+', help='default: g.ALL_DATASETS (those without the runs are skipped)')
    ap.add_argument('-k', type=int, default=g.K_MAIN)
    ap.add_argument('--lists', choices=list(LIST_FILE), default='returned',
                    help='merge the returned lists (results.json) or the cosine-reranked ones (results_reRank.json)')
    ap.add_argument('--mrr', nargs='+', type=float, help='frozen MRR@10 values instead of the computed one')
    ap.add_argument('--recall', nargs='+', type=float, help='frozen Recall@10 values instead of the computed one')
    ap.add_argument('--wan', nargs='+', type=float, help='WAN budgets (s/query) instead of the computed one')
    ap.add_argument('--tol', type=float, default=fv.TOL, help='score match window, T +/- tol')
    ap.add_argument('--wan-tol', type=float, default=fv.WAN_TOL)
    ap.add_argument('--min-mrr', type=float, default=fv.MIN_MRR)
    ap.add_argument('--pick-by', choices=fv.PICK_BY_CHOICES, default=fv.PICK_BY,
                    help='frozen score: the matching run with the lowest of this')
    ap.add_argument('--tiptoe-token', action='store_true',
                    help="add Tiptoe's per-query token to its WAN / LAN (NOT like-for-like: PILLAR / PACMANN's Piano "
                         "refreshes, which stream the DB, are left out)")
    ap.add_argument('--no-write', dest='write', action='store_false')
    args = ap.parse_args()
    g.setup_logging()
    nd = load_results.load_results()
    add_baselines(nd)
    if args.tiptoe_token:
        add_tiptoe_token(nd)
    report(nd, args)
