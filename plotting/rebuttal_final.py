"""
Every table for the camera-ready / final rebuttal numbers, recomputed from the results (nothing typed in by hand),
written as ONE Markdown file that is meant to be pasted into an LLM:

    figures/rebuttal_final.md

    python rebuttal_final.py                    # every table
    python rebuttal_final.py --only best fixed_mrr_bin scale

The file has two parts, split by a dashed line (SEPARATOR):
  1. RESULTS: the clean tables, numbers only (no configs, no selection notes). Paste this part.
  2. DETAILS: what each number is (config, exact MRR@10, which selection tier was used, folder), for you.
A fixed-MRR@10 pick that only gets within TOL under the target is shown as a plain number in part 1 (the detail is
in part 2). A method that never reaches the target is marked UNREACHED_MARK in part 1 (MARK_UNREACHED = False drops
the mark too).

The same functions feed the figures (plot_best_mrr.py, plot_rebuttal_bars.py, plot_query_terms.py), so a bar is
always the number in the table.

Filters (defaults, see globals.py): simulated results only (g.ONLY_SIMULATED), runs whose client storage is >= the
dataset's embedding DB are never loaded (g.FILTER_CLIENT_STORAGE, strict: g.KEEP_IF_ALL_DROPPED off) except on
SciFact (g.CLIENT_FILTER_EXEMPT), runs with MRR@10 < 0.01 ignored (working_rebuttal.MIN_MRR).
Selection at a fixed MRR@10 is working_rebuttal.pick(): MRR@10 (rounded down to 2 dp) >= T, then the lowest
PICK cost; if none, within TOL below T; if none, the highest-MRR@10 config (UNREACHED).
"""

import argparse
import logging
import math
import os
import statistics

import globals as g
import load_results
import plot_utils as pu
import frozen_values as fv
import fixed_storage as fs
import rebuttal_hybrid as rh
import scale_v2 as sv
import working_rebuttal as wr

log = logging.getLogger('rebuttal_final')

# ===================================================================================================== CONFIG
K = g.K_MAIN
MAIN4 = ['msmarco', 'scifact', 'fever', 'nq']
SCALE_SIZES = ['msmarco', 'msmarco-v2-25m', 'msmarco-v2-50m', 'msmarco-v2']

# Table "best": each method's best MRR@10 config per dataset, under the non-private references
BEST_DATASETS = g.REBUTTAL_DATASETS
BEST_METHODS = ['bins', 'tree', 'pacmann', 'tiptoe', 'coeus']
# (key in baselines.json, label). 'HybridRRF' is the fused hybrid the rebuttal reported (MS MARCO 0.28);
# 'Hybrid' is the alpha-weighted score fusion
NONPRIVATE = [('BM25', 'BM25'), ('Dense', 'Exact dense'), ('HybridRRF', 'Fused hybrid')]

# Fixed-MRR@10 tables: one table (and one bar figure) each. 'pick' = what "cheapest" means among the configs
# reaching the target; 'costs' = the columns shown. The Tiptoe+Coeus hybrid ('hybrid') appears in ONE table only.
FIXED_MRR = {
    'fixed_mrr_bin':  dict(title='WAN latency at fixed MRR@10', target=0.10, pick='wan_time', costs=['wan_time'],
                           methods=['bins', 'pacmann', 'tiptoe', 'coeus'], datasets=MAIN4),
    'fixed_mrr_tree': dict(title='WAN latency at fixed MRR@10', target=0.15, pick='wan_time', costs=['wan_time'],
                           methods=['tree', 'pacmann', 'tiptoe', 'coeus'], datasets=MAIN4),
    'fixed_mrr_hybrid': dict(title='WAN latency at fixed MRR@10, with the Tiptoe+Coeus hybrid', target=0.17,
                             pick='wan_time', costs=['wan_time'],
                             methods=['tiptoe', 'coeus', 'hybrid', 'bins', 'tree'],
                             datasets=['scifact', 'nq', 'fever', 'msmarco']),
    'scale': dict(title='Cost vs corpus size at fixed MRR@10 (lowest client storage)', target=0.05,
                  pick='client_storage_mb', costs=['wan_time', 'client_storage_mb'],
                  methods=['pacmann', 'bins', 'tree'], datasets=SCALE_SIZES),
}

# Fixed config across corpus sizes: one config per method, the smallest client storage reaching FIXED_CONFIG_MRR on
# FIXED_CONFIG_BASE among the configs run at every size the method has runs at (scale_v2.pick_fixed), kept unchanged.
# FIXED_CONFIG_USE = {'bins': 'vec0_bs1000000_dpb50_vb10_vr10_ib24'} forces a config.
FIXED_CONFIG_MRR = 0.10
FIXED_CONFIG_METHODS = ['bins', 'pacmann']
FIXED_CONFIG_USE = {}

# Fixed client storage (MS MARCO): each method's runs at client storage S, the lowest WAN (rounded down to 2 dp; ties
# to the highest MRR@10), as fixed_storage.py. STORAGE_MODE 'window' = S +/- STORAGE_TOL_GB (what the posted rebuttal
# numbers used: PACMANN 0.25 s / 0.02 at 0.5 GB); 'budget' = storage <= S (then PACMANN has nothing under 0.5 GB and
# keeps its 593 MB, MRR@10 0.02 config at every larger budget, since it is the fastest)
STORAGE_DATASET = 'msmarco'
STORAGE_METHODS = ['bins', 'pacmann']
STORAGE_BUDGETS_GB = [0.5, 1.0, 1.5]
STORAGE_MODE = 'window'
STORAGE_TOL_GB = 0.1

# Query-term ablation (PILLAR-Tree q*): per dataset the (b, r, s, L, c) family swept over the most q* values
QT_DATASETS = g.REBUTTAL_DATASETS
QT_METHOD = 'tree'
QT_MIN_VALUES = 3          # a dataset with fewer q* values than this is left out

# RAGAS: each method's config with a RAGAS CSV and the highest MRR@10 (working_rebuttal.ragas_runs)
RAGAS_DATASETS = ['msmarco', 'scifact']
RAGAS_METHODS = ['tree', 'bins', 'pacmann']

# Document-text fetch (the extra PIR round after the top-10 IDs), from the docfetch runs
DOCFETCH_DATASETS = ['scifact', 'nq', 'fever', 'msmarco']

MB_PER_GB = 1024           # ClientStorageMB is MiB, so 1024 gives GiB. The posted rebuttal's GB columns used 1000
                           # (4,677 MB -> 4.7 GB); set 1000 to match them. Also used for the fixed-storage targets
MARK_UNREACHED = True
UNREACHED_MARK = '†'
SEPARATOR = '- ' * 40
OUT = os.path.join(g.FIGURE_DIR, 'rebuttal_final.md')
# ================================================================================================= end CONFIG

L, D = wr.L, wr.D
COST_HEAD = {'wan_time': 'WAN (s)', 'lan_time': 'LAN (s)', 'comm_kb': 'Comm (KB)',
             'client_storage_mb': 'Storage (GB)'}


def score(v):
    return fv.r2(v) if pu.is_number(v) else 'n/a'


def cost(v, key):
    if not pu.is_number(v):
        return 'n/a'
    if key == 'client_storage_mb':
        return f'{v / MB_PER_GB:.2f}'
    if key == 'comm_kb':
        return f'{v:,.1f}'
    return wr.f_s(v)


def md(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)] + \
           ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]


def describe(run):
    """one line about a run for the DETAILS part"""
    if run is None:
        return 'no run'
    bits = [f"`{run['config']}`", f"MRR@10 {run['mrr']:.4f}", f"Recall@10 {run.get('recall', math.nan):.4f}",
            f"WAN {run.get('wan_time', math.nan):.3f} s"]
    if pu.is_number(run.get('client_storage_mb')):
        bits.append(f"client {run['client_storage_mb']:,.1f} MB")
    bits.append(f"[{run.get('source', 'real')}] {run.get('folder', '')}")
    return ', '.join(bits)


FLAG_TEXT = {'': 'reaches the target', '~': 'within TOL below the target (closest config, shown as a plain number)',
             '*': 'NEVER reaches the target: its highest-MRR@10 config', 'pending': 'no runs'}


class Result:
    """one table: clean lines (RESULTS part), detail lines (DETAILS part), and the data the figures read"""

    def __init__(self, key, title):
        self.key, self.title, self.clean, self.details, self.data = key, title, [], [], {}

    def text(self, part):
        lines = self.clean if part == 'clean' else self.details
        return '\n'.join([f'## {self.title}', ''] + lines + ['']) if lines else ''


# --------------------------------------------------------------------------------------------------- loading

def runs_from(nested_data, tiptoe_token=None):
    """working_rebuttal.Runs over a COPY of nested_data (so the Tiptoe token / storage tweaks don't leak into the
    other figures): Tiptoe / Coeus from RESULTS_DIRS when missing, Tiptoe's per-query token added to its WAN / LAN
    (working_rebuttal.TIPTOE_TOKEN), and Tiptoe / Coeus client storage back to what they report."""
    nd = {m: {d: [dict(r) for r in rs] for d, rs in by_d.items()} for m, by_d in nested_data.items()}
    rh.add_baselines(nd)
    if wr.TIPTOE_TOKEN if tiptoe_token is None else tiptoe_token:
        rh.add_tiptoe_token(nd)
    for by_d in nd.values():
        for rs in by_d.values():
            for r in rs:
                if not pu.is_number(r.get('client_storage_mb')) and pu.is_number(r.get('client_storage_mb_reported')):
                    r['client_storage_mb'] = r['client_storage_mb_reported']
    return wr.Runs(nd)


_R = {}


def get_runs(nested_data=None):
    """the Runs every table and figure shares (built once per process)"""
    if 'R' not in _R:
        _R['R'] = runs_from(nested_data if nested_data is not None else load_results.load_results())
    return _R['R']


# ---------------------------------------------------------------------------------------------------- tables

def best_scores(R, datasets=None, methods=None):
    datasets, methods = datasets or BEST_DATASETS, methods or BEST_METHODS
    res = Result('best', 'Best MRR@10 per method, with non-private references')
    best = {(m, d): wr.best_mrr(R.get(m, d)) for m in methods for d in datasets}
    nonpriv = {}
    for d in datasets:
        js = wr.nonprivate(d) or {}
        for key, _ in NONPRIVATE:
            v = js.get(key)
            nonpriv[(key, d)] = v.get('MRR@10', v.get('MRR')) if isinstance(v, dict) else None
    rows = [[label] + [score(nonpriv[(key, d)]) for d in datasets] for key, label in NONPRIVATE]
    rows += [[L(m)] + [score(best[(m, d)]['mrr']) if best[(m, d)] else 'n/a' for d in datasets] for m in methods]
    res.clean = md(['Method'] + [f'{D(d)} MRR@10' for d in datasets], rows) + \
        ['', f'The first {len(NONPRIVATE)} rows are non-private. n/a = not run.']
    res.details = [f'- {D(d)}, {L(m)}: {describe(best[(m, d)])}' for d in datasets for m in methods] + \
                  [f'- {D(d)}, {label} ({key}): ' + (f'{nonpriv[(key, d)]:.4f}' if nonpriv[(key, d)] else 'missing')
                   for d in datasets for key, label in NONPRIVATE] + \
                  [f'- non-private source: {wr._NONPRIVATE.get("_path")}']
    res.data = dict(datasets=datasets, methods=methods, best=best, nonprivate=nonpriv,
                    nonprivate_labels=NONPRIVATE)
    return res


def fixed_mrr(R, key, spec=None):
    spec = spec or FIXED_MRR[key]
    t, ms, ds, costs = spec['target'], spec['methods'], spec['datasets'], spec['costs']
    res = Result(key, f"{spec['title']} = {t:.2f}")
    picks = {(m, d): wr.pick(R.get(m, d), t, cost=spec['pick']) for m in ms for d in ds}

    def cell(m, d, c):
        run, flag = picks[(m, d)]
        if run is None:
            return 'n/a'
        mark = UNREACHED_MARK if MARK_UNREACHED and flag == '*' else ''
        return cost(run.get(c), c) + mark

    head = ['Method'] + [f'{D(d)} {COST_HEAD.get(c, c)}' for d in ds for c in costs]
    rows = [[L(m)] + [cell(m, d, c) for d in ds for c in costs] for m in ms]
    res.clean = md(head, rows)
    unreached = [(m, d) for (m, d), (_, f) in picks.items() if f == '*']
    if unreached and MARK_UNREACHED:
        res.clean += ['', f'{UNREACHED_MARK} = no configuration reaches MRR@10 {t:.2f}; its highest-MRR@10 '
                          'configuration is shown.']
    res.clean += ['', 'n/a = not run.'] if any(picks[p][0] is None for p in picks) else []
    res.details = [f"Rule: MRR@10 (rounded down to 2 dp) >= {t:.2f}, lowest {spec['pick']}; else within "
                   f"{wr.TOL:.2f} below; else the highest MRR@10."] + \
                  [f'- {D(d)}, {L(m)}: {FLAG_TEXT[picks[(m, d)][1]]}; {describe(picks[(m, d)][0])}'
                   for d in ds for m in ms]
    hyb = [f'- Tiptoe+Coeus on {D(d)}: {R.hyb_status[d]}' for d in ds if 'hybrid' in ms and d in R.hyb_status]
    res.details += hyb
    res.data = dict(spec, picks=picks)
    return res


def fixed_config(nested, R, f=None, methods=None, use=None):
    f = FIXED_CONFIG_MRR if f is None else f
    methods, use = methods or FIXED_CONFIG_METHODS, use if use is not None else FIXED_CONFIG_USE
    res = Result('fixed_config', f'Client storage of one fixed configuration as the corpus grows '
                                 f'(configs reach MRR@10 {f:.2f} on {D(SCALE_SIZES[0])})')
    by = {m: {d: sv.runs(nested, m, d) for d in SCALE_SIZES} for m in methods}
    sv_sizes = sv.SIZES
    sv.SIZES = [(d, n) for d, n in sv.SIZES if d in SCALE_SIZES]   # pick_fixed reads SIZES[0] as the base
    try:
        cfg = {m: use.get(m) or sv.pick_fixed(by[m], f) for m in methods}
    finally:
        sv.SIZES = sv_sizes
    sel = {(m, d): next((r for r in by[m][d] if r['config'] == cfg[m]), None) for m in methods for d in SCALE_SIZES}
    head = ['Corpus', 'Documents', 'Vocabulary']
    for m in methods:
        head += [f'{L(m)} storage (MB)'] + ([f'{L(m)} ID DB part (MB)', f'{L(m)} embedding DB part (MB)']
                                            if m == 'bins' else []) + [f'{L(m)} MRR@10']
    rows = []
    for d in SCALE_SIZES:
        n, _ = wr.doc_count(d)
        v = sv.vocab(nested, d)
        row = [D(d), f'{n / 1e6:.1f}M' if n else 'n/a', f'{v:,}' if v else 'n/a']
        for m in methods:
            r = sel[(m, d)]
            mb = lambda key: f'{r[key]:,.1f}' if r and pu.is_number(r.get(key)) else 'n/a'
            row += [mb('client_storage_mb')] + ([mb('client_storage_mb_bm25'), mb('client_storage_mb_vec')]
                                                if m == 'bins' else []) + [score(r['mrr']) if r else 'n/a']
        rows.append(row)
    res.clean = md(head, rows) + ['', 'Each method keeps the same configuration at every corpus size. n/a = not run.']
    res.details = [f'- {L(m)}: `{cfg[m]}`' + (' (forced in FIXED_CONFIG_USE)' if m in use else '') for m in methods] + \
                  [f'- {D(d)}, {L(m)}: {describe(sel[(m, d)])}' for d in SCALE_SIZES for m in methods] + \
                  ['- PILLAR-Bin ID DB = the BM25 bins DB (bs rows, fixed by the config); embedding DB = the second '
                   '(vec0) DB, one row per stored document, so it grows with the corpus until every bin is full.']
    res.data = dict(configs=cfg, sel=sel, sizes=SCALE_SIZES, methods=methods)
    return res


def fixed_storage(R, budgets=None, methods=None, dataset=None):
    budgets, methods, d = budgets or STORAGE_BUDGETS_GB, methods or STORAGE_METHODS, dataset or STORAGE_DATASET
    res = Result('fixed_storage', f'Fixed client storage ({D(d)}): lowest WAN at each client storage')
    _, key = fs.PICKS['wan']
    fs.MB_PER_GB = MB_PER_GB
    picks = {}
    for b in budgets:
        for m in methods:
            runs = [r for r in R.get(m, d) if pu.is_number(r.get('client_storage_mb'))]
            at = fs.matching(runs, b, STORAGE_TOL_GB, STORAGE_MODE)
            picks[(b, m)] = min(at, key=key) if at else None
    rows = [[f'{b:g} GB', L(m)] + ([cost(picks[(b, m)]['wan_time'], 'wan_time'), score(picks[(b, m)]['mrr'])]
                                    if picks[(b, m)] else ['n/a', 'n/a']) for b in budgets for m in methods]
    res.clean = md(['Client storage', 'Method', 'WAN (s)', 'MRR@10'], rows)
    if any(p is None for p in picks.values()):
        res.clean += ['', 'n/a = no configuration at this storage.']
    rule = 'client storage <= S' if STORAGE_MODE == 'budget' else f'client storage within S +/- {STORAGE_TOL_GB:g} GB'
    res.details = [f'Rule: {rule} (1 GB = {MB_PER_GB} MB), lowest WAN rounded down to 2 dp, ties to the highest MRR@10.'] + [f'- {b:g} GB, {L(m)}: {describe(picks[(b, m)])}' for b in budgets for m in methods]
    res.data = dict(budgets=budgets, methods=methods, dataset=d, picks=picks, mode=STORAGE_MODE, tol=STORAGE_TOL_GB)
    return res


def query_terms(nested, datasets=None, method=None):
    datasets, method = datasets or QT_DATASETS, method or QT_METHOD
    res = Result('query_terms', f'Number of query terms used by {L(method)} (q*) vs MRR@10')
    fams, chosen = {}, {}
    for d in datasets:
        fam = {}
        for r in nested.get(method, {}).get(d, []):
            if r['k'] == K and r.get('q') is not None and pu.is_number(r.get('mrr')):
                fam.setdefault((r['b'], r['r'], r['s'], r['L'], r.get('k_candidates')), {})[r['q']] = r
        if not fam:
            continue
        key, byq = max(fam.items(), key=lambda kv: (len(kv[1]), max(x['mrr'] for x in kv[1].values())))
        if len(byq) >= QT_MIN_VALUES:
            fams[d], chosen[d] = key, byq
    ds = list(chosen)
    qs = sorted({q for byq in chosen.values() for q in byq})
    rows = [[q] + [f"{chosen[d][q]['mrr']:.3f}" if q in chosen[d] else '' for d in ds] for q in qs]
    res.clean = md([f'Query terms (q*)'] + [f'{D(d)} MRR@10' for d in ds], rows) + \
        ['', 'Every other parameter is fixed per dataset; an empty cell was not run.']
    res.details = [f'- {D(d)}: family b={k[0]} r={k[1]} s={k[2]} L={k[3]} c={k[4]}; ' +
                   ', '.join(f"q{q} {r['mrr']:.4f} ({r['folder']})" for q, r in sorted(chosen[d].items()))
                   for d, k in fams.items()] + \
                  [f'- left out (fewer than {QT_MIN_VALUES} q* values): ' +
                   (', '.join(D(d) for d in datasets if d not in chosen) or 'none')]
    res.data = dict(method=method, datasets=ds, q_values=qs, runs=chosen, families=fams)
    return res


def ragas(R, datasets=None, methods=None):
    datasets, methods = datasets or RAGAS_DATASETS, methods or RAGAS_METHODS
    res = Result('ragas', 'RAGAS answer relevancy and faithfulness with 95% confidence intervals')
    per = {}
    for m, d, run, path in wr.ragas_runs():
        per.setdefault((m, d), []).append((run, path))
    choose = lambda xs: max(xs, key=lambda x: (x[0]['mrr'] if x[0] and pu.is_number(x[0].get('mrr')) else -1, x[1]))
    head = ['Method']
    for d in datasets:
        head += [f'{D(d)} {name}' for name in ('Answer relevancy', 'Faithfulness')] + [f'{D(d)} WAN (s)']
    rows, picked = [], {}
    for m in methods:
        row = [L(m)]
        for d in datasets:
            if (m, d) not in per:
                row += ['n/a'] * 3
                continue
            run, path = picked[(m, d)] = choose(per[(m, d)])
            for metric in ('answer_relevancy', 'faithfulness'):
                q = wr._per_q(path, metric)
                row.append(_ci2(f'{statistics.mean(q.values())} {wr.mean_ci(q.values())}') if q else 'n/a')
            row.append(cost(run.get('wan_time'), 'wan_time') if run else 'n/a')
        rows.append(row)
    res.clean = md(head, rows) + ['', 'Mean [95% bootstrap CI] over the evaluated queries.']
    res.details = [f'- {D(d)}, {L(m)}: {describe(run)}; CSV {path} '
                   f'({len(wr._per_q(path))} queries scored for answer relevancy)'
                   for (m, d), (run, path) in picked.items()] + \
                  [f'- bootstrap: {wr.BOOTSTRAP_N} resamples, seed {wr.BOOTSTRAP_SEED}. RAGAS runs are the older real '
                   'runs (one per select_configs pick); WAN is that run\'s.']
    res.data = dict(picked=picked)
    return res


def _ci2(text):
    """'0.7512 [0.693, 0.811]' -> '0.75 [0.69, 0.81]'"""
    mean, ci = text.split(' ', 1)
    lo, hi = ci.strip('[]').split(', ')
    return f'{float(mean):.2f} [{float(lo):.2f}, {float(hi):.2f}]'


def docfetch(R, datasets=None):
    datasets = datasets or DOCFETCH_DATASETS
    res = Result('docfetch', 'Extra cost of fetching the text of the top-10 documents (one PIR round, same for every method)')
    rows, det = [], []
    for d in datasets:
        meta, path = wr.docfetch_meta(d)
        if meta is None:
            rows.append([D(d), 'n/a', 'n/a'])
            det.append(f'- {D(d)}: no docfetch_{d}_k{K}_flate/metadata.json')
            continue
        rows.append([D(d), f"{float(meta['WANTimePerQuery']):.3f}", f"{float(meta['BytesPerQuery']) / 1024:,.1f}"])
        det.append(f"- {D(d)}: LAN {float(meta['LANTimePerQuery']):.3f} s, client storage "
                   f"{float(meta['ClientStorageMB']):,.0f} MB, compression {meta.get('Compression')} ({path})")
    res.clean = md(['Dataset', 'WAN (s/query)', 'Comm (KB/query)'], rows)
    res.details = det
    return res


TABLES = ['best', 'fixed_mrr_bin', 'fixed_mrr_tree', 'fixed_mrr_hybrid', 'scale', 'fixed_config', 'fixed_storage',
          'query_terms', 'ragas', 'docfetch']


def compute(key, R):
    if key == 'best':
        return best_scores(R)
    if key in FIXED_MRR:
        return fixed_mrr(R, key)
    if key == 'fixed_config':
        return fixed_config(R.nd, R)
    if key == 'fixed_storage':
        return fixed_storage(R)
    if key == 'query_terms':
        return query_terms(R.nd)
    if key == 'ragas':
        return ragas(R)
    if key == 'docfetch':
        return docfetch(R)
    raise KeyError(key)


def report(nested_data=None, only=None, out=OUT, quiet=False):
    R = get_runs(nested_data)
    results = []
    for key in only or TABLES:
        try:
            results.append(compute(key, R))
        except Exception as err:   # one broken table must not lose the others
            log.exception('table %s failed', key)
            r = Result(key, f'{key}: FAILED')
            r.details = [repr(err)]
            results.append(r)
    filt = ('client storage >= embedding DB dropped' + (f" (not on {', '.join(g.CLIENT_FILTER_EXEMPT)})"
                                                        if g.CLIENT_FILTER_EXEMPT else '')
            + (', '.join([''] + [f'not for {m} on {d}' for m, d in sorted(g.CLIENT_FILTER_EXEMPT_PAIRS)])
               if g.CLIENT_FILTER_EXEMPT_PAIRS else '')
            if g.FILTER_CLIENT_STORAGE else 'client storage filter OFF')
    text = '\n'.join(['# Results', ''] + [r.text('clean') for r in results if r.clean] +
                     ['', SEPARATOR, '', '# Details (not for the paper)', '',
                      f"Runs: {'simulated only' if g.ONLY_SIMULATED else 'real, else simulated'}; {filt}; "
                      f"MRR@10 < {wr.MIN_MRR:.2f} ignored; Tiptoe WAN/LAN "
                      f"{'include' if wr.TIPTOE_TOKEN else 'exclude'} its per-query token. Scores rounded down to "
                      f"2 dp, times rounded to 2 dp, storage in GB = MB / {MB_PER_GB}.", ''] +
                     [r.text('details') for r in results if r.details])
    if not quiet:
        print(text)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, 'w').write(text)
        log.info('saved %s', out)
    return {r.key: r for r in results}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=TABLES)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    g.setup_logging()
    report(only=args.only, out=args.out)
