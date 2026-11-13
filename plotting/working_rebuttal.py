"""
Every table for the ICLR rebuttal, built from the raw results (metadata.json of every run, per-query files,
baselines.py JSONs, docfetch runs). Nothing is hard-coded except the CONFIG block below (targets, names, paths,
dataset facts such as document counts). Re-run whenever new results land.

    python working_rebuttal.py                        # every table
    python working_rebuttal.py --table 1              # one table (0 = target feasibility)
    python working_rebuttal.py --table 1 3 --md rebuttal_tables.md --csv figures/rebuttal_csv/
    python working_rebuttal.py --mrr-bin 0.15 --mrr-tree 0.30 --t1-datasets scifact nq fever
                                                      # ONE target per protocol on every dataset
    python working_rebuttal.py --match closest_above  # pick the config closest to (and above) the target
    python working_rebuttal.py --storage-filter both  # every table with the client-storage filter off AND on
    python working_rebuttal.py --storage-filter on --storage-exempt scifact

SELECTION RULE (everywhere a fixed MRR@10 appears), MRR@10 rounded DOWN to 2 dp as in frozen_values.py:
  tier 1  configs with MRR@10 >= T                 -> MATCH_MODE picks one (see below)
  tier 2  none: configs with MRR@10 >= T - TOL     -> same pick, row marked "~"  (TOL = 0 turns this off)
  tier 3  none: the highest-MRR@10 config          -> row marked "*", the MRR@10 it reaches is in the row
  no runs at all                                   -> row says "pending"
MATCH_MODE 'atleast' (default; the rebuttal's rule): lowest WAN in the tier. 'closest_above': MRR@10 closest to T
(ties: lowest WAN). 'window': like frozen_values.py, |MRR@10 - T| <= TOL, lowest WAN.

Table 0 shows each method's best MRR@10 per dataset and the highest single target every method reaches on a set
of datasets, so you can check whether one MRR@10 for every dataset is possible (usually: not with MS MARCO).
"""

import argparse
import csv
import glob
import json
import logging
import math
import multiprocessing as mp
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

import globals as g
import load_results
import plot_utils as pu
import frozen_values as fv
import rebuttal_hybrid as rh
import rebuttal_numbers as rn

log = logging.getLogger('working_rebuttal')

# ===================================================================================================== CONFIG
K = g.K_MAIN

# dataset keys (as in the results folder names) -> labels used in the tables
DATASET_LABELS = {'msmarco': 'MS MARCO v1 (8.8M)', 'scifact': 'SciFact', 'fever': 'FEVER (5.4M)', 'nq': 'NQ (2.7M)',
                  'msmarco-v2-25m': 'MS MARCO v2 25M', 'msmarco-v2-50m': 'MS MARCO v2 50M',
                  'msmarco-v2': 'MS MARCO v2 138M'}
METHOD_LABELS = {'bins': 'PILLAR-Bin', 'tree': 'PILLAR-Tree', 'pacmann': 'PACMANN', 'tiptoe': 'Tiptoe',
                 'coeus': 'Coeus', 'hybrid': 'Tiptoe+Coeus (merged)'}
PILLAR = {'bin': 'bins', 'tree': 'tree'}          # protocol name -> method key
BASELINES = ['pacmann', 'tiptoe', 'coeus', 'hybrid']

# Table 1: per-dataset targets {dataset: (Bin target, Tree target)}. None = auto: the highest MRR@10 that the PILLAR
# method and every baseline (hybrid excluded) reaches on that dataset. --mrr-bin / --mrr-tree override every dataset.
T1_DATASETS = ['msmarco', 'scifact', 'fever', 'nq']
T1_TARGETS = {'msmarco': (0.10, 0.20), 'fever': (0.54, 0.65), 'nq': (0.20, 0.40), 'scifact': (None, None)}
MATCH_MODE = 'atleast'     # 'atleast' | 'closest_above' | 'window'
TOL = 0.01                 # fuzzy window under the target (tier 2), in MRR@10 units at 2 dp
MIN_MRR = 0.01             # configs with MRR@10 below this are grid-search leftovers and ignored
PICK_COST = 'wan_time'     # "lowest WAN"; e.g. 'client_storage_mb' to pick the smallest client instead

# Client-storage filter: load_results() drops runs whose client storage is >= the dataset's embedding DB
# (globals.EMBED_DB_MB). On SciFact (3.8 MB of embeddings) that drops EVERY PILLAR-Bin config.
#   'off'  every run is kept (default)     'on'  the paper's filter (STORAGE_EXEMPT datasets still keep theirs)
#   'both' every table twice, filter off then on
STORAGE_FILTER = 'off'
STORAGE_EXEMPT = []

# Table 2: baselines.py output. Either one combined file {dataset: {"BM25": {...}, "Dense": ..., "Hybrid": ...,
# "HybridRRF": ..., "NumDocs", "Params"}} (hopper's bins-go/baselines.json) or per-dataset <d>_k10.json files.
BASELINES_JSON = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines.json'),
                  os.path.join(g.DATASETS_ROOT, 'baselines.json')]
T2_DATASETS = T1_DATASETS + ['msmarco-v2-25m', 'msmarco-v2-50m', 'msmarco-v2']
T2_METHODS = ['bins', 'tree', 'pacmann', 'tiptoe', 'coeus']
# (method, dataset) pairs with no runs that are known to be infeasible / not run, labelled "not run" instead of
# "pending" (e.g. Tiptoe / Coeus at MS MARCO v2 scale). Any other empty pair is still reported as pending.
T2_NOT_RUN = {(m, d) for m in ('tiptoe', 'coeus') for d in ('msmarco-v2-25m', 'msmarco-v2-50m', 'msmarco-v2')}
T2_NOT_RUN |= {('pacmann', 'msmarco-v2')}
NONPRIVATE = [('BM25', 'BM25'), ('Dense', 'Exact dense (e5, PCA-192)'), ('Hybrid', 'Fused hybrid (score)'),
              ('HybridRRF', 'Fused hybrid (RRF)')]

# Table 3 / 4
SCALE_DATASETS = ['msmarco', 'msmarco-v2-25m', 'msmarco-v2-50m', 'msmarco-v2']
SCALE_METHODS = ['bins', 'tree', 'pacmann']
T_SCALE = 0.10
T4_TARGET = 0.10
# Doc counts: read from baselines.json (NumDocs), else the docfetch run (NumDocs); this is only the last fallback
DOC_COUNTS_FALLBACK = {'msmarco': 8_841_823, 'msmarco-v2-25m': 25_000_000, 'msmarco-v2-50m': 50_000_000,
                       'msmarco-v2': 138_364_198, 'scifact': 5_183, 'nq': 2_681_468, 'fever': 5_416_568}

# Table 5: tree family (b, r, s, L, c) whose q* is varied; None = the msmarco family with the most of Q_VALUES
Q_VALUES = [2, 4, 6, 8]
Q_FAMILY = None
# The tree keeps the FIRST q* analysed query terms. The reviewer also asked for "highest-IDF q* terms", which needs a
# code change in the tree and new runs. Set SHOW_IDF_ROWS = False to drop those rows; if such runs are ever made, put
# IDF_FOLDER_MARK in their folder name and they fill in.
SHOW_IDF_ROWS = True
IDF_FOLDER_MARK = 'idf'

# Table 6 / 7 / 8
WINLOSS_PICK = 'best_recall'   # 'best_recall' or 'tree_target' (the Table 1 picks at the Tree target)
RAGAS_DATASETS = ['msmarco', 'scifact', 'fever', 'nq']
RAGAS_METHODS = ['tree', 'bins', 'pacmann']        # each PILLAR method is paired against PACMANN
# The RAGAS runs are older REAL runs (one per select_configs pick) with ragas_output_<method>_k10.csv in their folder.
RAGAS_DIRS = [os.path.join(g.DATASETS_ROOT, 'results'), g.SIMULATED_RESULTS_DIRS[0]] + g.RESULTS_DIRS
RAGAS_SKIP = 'local'       # ragas_output_local_* = the 15-query local test, left out
RAGAS_PAIR = 'best_mrr'    # which config of each method is paired in the headline: 'best_mrr' (of those with a CSV)
BOOTSTRAP_N, BOOTSTRAP_SEED = 1000, 0
E2E_DATASETS = T1_DATASETS  # Table 8: the Table 1 picks (same targets) + the measured doc-text fetch of the dataset
DOCFETCH_DIRS = [os.path.join(g.DATASETS_ROOT, 'results_docfetch')]
HYBRID_WORKERS = os.cpu_count() or 1   # processes for the Tiptoe x Coeus merges (1 = serial, rebuttal_hybrid's code)
# ================================================================================================= end CONFIG

MISSING = []
L = lambda m: METHOD_LABELS.get(m, g.METHOD_LABELS.get(m, m))
D = lambda d: DATASET_LABELS.get(d, g.DATASET_LABELS.get(d, d))
rkey = fv.rkey


# ----------------------------------------------------------------------------------------------- formatting

def f_score(v):
    return 'n/a' if not pu.is_number(v) else fv.r2(v)


def f_s(v):
    if not pu.is_number(v):
        return 'n/a'
    return f'{v:.3f}' if 0 < abs(v) < 0.01 else f'{v:.2f}'


def f_int(v):
    if not pu.is_number(v):
        return 'n/a'
    return f'{v:.1f}' if 0 < abs(v) < 1 else f'{v:,.0f}'


def f_x(v):
    if not pu.is_number(v):
        return 'n/a'
    return f'{v:.2f}×' if 0 < abs(v) < 0.1 else f'{v:.1f}×'


def f_pct(v):
    return 'n/a' if not pu.is_number(v) else f'{v:.0%}'


def f_str(v):
    return '' if v is None else str(v)


class Table:
    """rows of raw values; columns = (header, key, formatter). Markdown formats, CSV keeps the raw values."""

    def __init__(self, num, title, columns, intro=None):
        self.num, self.title, self.columns, self.intro = num, title, columns, intro or []
        self.rows, self.notes, self.blocks, self.suffix = [], [], [], ''

    def add(self, **row):
        self.rows.append(row)

    def markdown(self):
        out = [f'## Table {self.num}. {self.title}', ''] + [x for i in self.intro for x in (i, '')]
        for name, cols, rows in [(None, self.columns, self.rows)] + self.blocks:
            if name:
                out += [f'**{name}**', '']
            if not rows:
                out += ['(no rows)', '']
                continue
            out.append('| ' + ' | '.join(h for h, _, _ in cols) + ' |')
            out.append('|' + '---|' * len(cols))
            for r in rows:
                out.append('| ' + ' | '.join(fmt(r.get(k)) for _, k, fmt in cols) + ' |')
            out.append('')
        out += [f'- {n}' for n in self.notes]
        return '\n'.join(out).rstrip() + '\n'

    def write_csv(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        for i, (name, cols, rows) in enumerate([(None, self.columns, self.rows)] + self.blocks):
            path = os.path.join(outdir, f'table{self.num}{self.suffix}' + (f'_{i}' if i else '') + '.csv')
            with open(path, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow([h for h, _, _ in cols])
                for r in rows:
                    w.writerow(['' if r.get(k) is None else r.get(k) for _, k, _ in cols])


# --------------------------------------------------------------------------------------------------- loading

def load(modes, storage_exempt):
    """{'off': nested data with every run, 'on': the paper's client-storage filter (+ exempt datasets)}"""
    nd = load_results.load_results()   # filtered; what it dropped is in load_results.DROPPED
    rh.add_baselines(nd)
    for m in nd:   # Tiptoe / Coeus storage is NaN'd for the paper's plots; the tables show what they report
        for rs in list(nd[m].values()) + list(load_results.DROPPED.get(m, {}).values()):
            for r in rs:
                if not pu.is_number(r.get('client_storage_mb')) and pu.is_number(r.get('client_storage_mb_reported')):
                    r['client_storage_mb'] = r['client_storage_mb_reported']
    out = {}
    for mode in modes:
        back_to = None if mode == 'off' else set(storage_exempt)
        cur = {m: {d: list(rs) for d, rs in by_d.items()} for m, by_d in nd.items()}
        for m, by_d in load_results.DROPPED.items():
            for d, rs in by_d.items():
                if rs and (back_to is None or d in back_to):
                    cur.setdefault(m, {}).setdefault(d, []).extend(rs)
        out[mode] = cur
    return out


_HYB = {}   # what the forked workers share: qrels and the distinct Coeus lists of the dataset being merged


def _merge_one_tiptoe(path):
    """one Tiptoe results.json merged with every distinct Coeus list, both orders -> (hash, {(ci, order): (mrr, rec)})"""
    tres = json.load(open(path))
    rel, out = _HYB['rel'], {}
    for ci, cres in enumerate(_HYB['coeus']):
        qids = [q for q in tres if q in cres]
        for order in ('T>C', 'C>T'):
            a, b = (tres, cres) if order == 'T>C' else (cres, tres)
            out[(ci, order)] = rh.mrr_recall({q: rh.round_robin(a[q], b[q], K) for q in qids}, rel)
    return out


def hybrids_parallel(ts, cs, d, workers=None):
    """rebuttal_hybrid.hybrids(), with the merges spread over HYBRID_WORKERS forked processes (one Tiptoe run per
    task; the Coeus lists and qrels are loaded once, before the fork, and shared). Same output."""
    workers = workers or HYBRID_WORKERS
    if workers <= 1:
        return rh.hybrids({'tiptoe': {d: ts}, 'coeus': {d: cs}}, d, 'returned', K)
    cl = [(c, rh.lists(c, 'returned')) for c in cs]
    cl = [(c, x) for c, x in cl if x]
    keys = list(dict.fromkeys(x[0] for _, x in cl))           # distinct Coeus lists
    _HYB.update(rel=rh.qrels(d), coeus=[next(x[1] for _, x in cl if x[0] == k) for k in keys])
    ci_of = {k: i for i, k in enumerate(keys)}
    tpaths = []
    for t in ts:
        p = os.path.join(t['results_dir'], t['folder'], rh.LIST_FILE['returned'])
        if os.path.exists(p):
            tpaths.append((t, p))
        else:
            rh.NEEDED[t['results_dir']].add(t['folder'])
    if not (tpaths and cl):
        return []
    with mp.get_context('fork').Pool(min(workers, len(tpaths))) as pool:
        merged = pool.map(_merge_one_tiptoe, [p for _, p in tpaths], chunksize=1)
    out = []
    for (t, _), res in zip(tpaths, merged):
        for c, (ck, _) in cl:
            for order in ('T>C', 'C>T'):
                mrr, rec = res[(ci_of[ck], order)]
                h = {'method': 'hybrid', 'dataset': d, 'k': K, 'mrr': mrr, 'recall': rec, 'order': order,
                     'config': f"{order} T:{t['config']} + C:{c['config']}", 'folder': f"{t['folder']} + {c['folder']}",
                     'source': 'simulated', 'tiptoe': t, 'coeus': c}
                for key in rh.SUM_KEYS:
                    x, y = t.get(key), c.get(key)
                    h[key] = x + y if pu.is_number(x) and pu.is_number(y) else math.nan
                out.append(h)
    log.info("%s: %d hybrid candidates (%d Tiptoe x %d Coeus runs, %d workers)", d, len(out), len(tpaths), len(cl),
             min(workers, len(tpaths)))
    return out


_HYBRIDS = {}   # dataset -> (hybrid runs, status); the storage filter doesn't touch Tiptoe / Coeus, so shared by modes


class Runs:
    """runs[method][dataset] at k=K with a usable MRR@10 and WAN, plus the Tiptoe+Coeus hybrids (built lazily)"""

    def __init__(self, nd):
        self.nd = nd
        self.by = defaultdict(dict)
        for m, by_d in nd.items():
            for d, rs in by_d.items():
                self.by[m][d] = [r for r in rs if r['k'] == K and pu.is_number(r.get('mrr'))
                                 and pu.is_number(r.get('wan_time')) and rkey(r['mrr']) >= rkey(MIN_MRR)]
        self.hyb_status = {}

    def get(self, m, d):
        if m == 'hybrid':
            return self.hybrid(d)
        return self.by.get(m, {}).get(d, [])

    def hybrid(self, d):
        if d not in _HYBRIDS:
            _HYBRIDS[d] = self._hybrid(d)
        out, status = _HYBRIDS[d]
        if status:
            self.hyb_status[d] = status
        return out

    def _hybrid(self, d):
        self.hyb_status.pop(d, None)
        out = []
        t, c = self.get('tiptoe', d), self.get('coeus', d)
        if not (t and c):
            self.hyb_status[d] = f"no {'Tiptoe' if not t else 'Coeus'} runs"
        elif not rh.qrels(d):
            self.hyb_status[d] = 'no qrels found'
        else:
            out = [h for h in hybrids_parallel(t, c, d)
                   if pu.is_number(h['mrr']) and rkey(h['mrr']) >= rkey(MIN_MRR)]
            if not out:
                self.hyb_status[d] = 'per-query results.json of Tiptoe / Coeus not local (rsync at the end)'
        return out, self.hyb_status.get(d)


def pick(runs, target, mode=None, tol=None, cost=None):
    """(run, flag): flag '' (meets T), '~' (within TOL under T), '*' (highest MRR, T not reached), 'pending'"""
    mode, tol, cost = mode or MATCH_MODE, TOL if tol is None else tol, cost or PICK_COST
    if not runs:
        return None, 'pending'
    t, s = rkey(target), rkey(tol)
    c = lambda r: r.get(cost) if pu.is_number(r.get(cost)) else math.inf
    if mode == 'window':
        tiers = [([r for r in runs if abs(rkey(r['mrr']) - t) <= s], '')]
    else:
        tiers = [([r for r in runs if rkey(r['mrr']) >= t], ''),
                 ([r for r in runs if t - s <= rkey(r['mrr']) < t], '~')]
    for cand, flag in tiers:
        if cand:
            key = (lambda r: (abs(rkey(r['mrr']) - t), c(r), r['wan_time'])) if mode == 'closest_above' else \
                  (lambda r: (c(r), r['wan_time'], -r['mrr']))
            r = min(cand, key=key)
            return r, flag or ('~' if mode == 'window' and rkey(r['mrr']) < t else '')
    return max(runs, key=lambda r: (r['mrr'], r['recall'], -r['wan_time'])), '*'


def best_mrr(runs):
    return max(runs, key=lambda r: (r['mrr'], r['recall'], -r['wan_time'])) if runs else None


def auto_target(R, methods, d):
    """highest 2-dp MRR@10 that every method (with runs) reaches on d"""
    tops = [rkey(best_mrr(R.get(m, d))['mrr']) for m in methods if R.get(m, d)]
    return min(tops) / 100 if tops else None


# ------------------------------------------------------------------------------------------- shared columns

RUN_COLS = [('Config', 'config', f_str), ('MRR@10', 'mrr', f_score), ('Recall@10', 'recall', f_score),
            ('WAN (s)', 'wan_time', f_s), ('LAN (s)', 'lan_time', f_s), ('Comm (KB)', 'comm_kb', f_int),
            ('Client storage (MB)', 'client_storage_mb', f_int)]


def run_row(run, flag, **extra):
    row = dict(extra)
    if run is None:
        row['config'] = 'pending'
        return row
    for _, k, _ in RUN_COLS:
        row[k] = run.get(k)
    row['config'] = run['config'] + (f" {flag} (reaches {fv.r2(run['mrr'])})" if flag == '*' else
                                     f' {flag}' if flag else '')
    row['flag'] = flag
    row['source'] = run.get('source', 'real')
    return row


def note_pick(table, where, m, flag, target, R, d):
    if flag == 'pending':
        why = R.hyb_status.get(d, 'no runs') if m == 'hybrid' else 'no runs'
        MISSING.append(f'Table {table}, {where}, {L(m)}: pending ({why})')
    elif flag == '*':
        MISSING.append(f'Table {table}, {where}, {L(m)}: no config reaches MRR@10 {target:.2f}; highest-MRR config shown (*)')


FLAG_NOTE = ('`~` = no config reaches T, this one is within the tolerance below it (TOL {tol:.2f}); `*` = no config '
             'reaches T (even within TOL), the highest-MRR@10 config is shown with the MRR@10 it reaches; "pending" '
             '= no results yet. MRR@10 / Recall@10 rounded down to 2 dp (the value that was matched).')


# ----------------------------------------------------------------------------------------------------- tables

def table0(R, ctx):
    methods = ['bins', 'tree', 'pacmann', 'tiptoe', 'coeus']
    t = Table(0, 'Target feasibility: best MRR@10 per method and dataset',
              [('Dataset', 'dataset', f_str)] + [(L(m), m, f_str) for m in methods] +
              [('Bin target ceiling', 'cb', f_score), ('Tree target ceiling', 'ct', f_score)],
              ['Ceiling = highest MRR@10 every method of the protocol (PILLAR method + PACMANN, Tiptoe, Coeus with runs) '
               'reaches. One target for several datasets must be <= the smallest ceiling of those datasets.'])
    ds = ctx['t1_datasets']
    for d in ds:
        row = {'dataset': D(d)}
        for m in methods:
            b = best_mrr(R.get(m, d))
            row[m] = fv.r2(b['mrr']) if b else 'no runs'
        row['cb'] = auto_target(R, ['bins'] + BASELINES[:-1], d)
        row['ct'] = auto_target(R, ['tree'] + BASELINES[:-1], d)
        t.add(**row)
    for proto, key in (('Bin', 'cb'), ('Tree', 'ct')):
        for subset in [ds] + [[x for x in ds if x != y] for y in ds]:
            vals = [r[key] for r, d in zip(t.rows, ds) if d in subset and r[key] is not None]
            if vals and len(subset) > 1:
                t.notes.append(f'{proto} protocol, one target on {", ".join(D(x) for x in subset)}: at most '
                               f'{min(vals):.2f} (limited by {D(subset[[r[key] for r, d in zip(t.rows, ds) if d in subset].index(min(vals))])})')
    t.notes.append('Same ceilings with only PILLAR vs PACMANN: ' + '; '.join(
        f"{D(d)} Bin {fv.r2(auto_target(R, ['bins', 'pacmann'], d) or math.nan)} / Tree "
        f"{fv.r2(auto_target(R, ['tree', 'pacmann'], d) or math.nan)}" for d in ds))
    return t


def t1_targets(R, ctx, d):
    out = {}
    for i, (proto, m) in enumerate(PILLAR.items()):
        given = ctx['global_targets'][proto]
        tv = given if given is not None else T1_TARGETS.get(d, (None, None))[i]
        if tv is None:
            tv = auto_target(R, [m] + BASELINES[:-1], d)
        out[proto] = tv
    return out


def table1_picks(R, ctx):
    """{(dataset, protocol): (target, {method: (run, flag)})}, cached (Tables 1 and 8 use it)"""
    if 'picks' in ctx:
        return ctx['picks']
    picks = {}
    for d in ctx['t1_datasets']:
        for proto, tv in t1_targets(R, ctx, d).items():
            if tv is None:
                continue
            picks[(d, proto)] = (tv, {m: pick(R.get(m, d), tv) for m in [PILLAR[proto]] + BASELINES})
    ctx['picks'] = picks
    return picks


def table1(R, ctx):
    t = Table(1, 'Fixed-MRR@10 comparison (main results)',
              [('Dataset', 'dataset', f_str), ('Target', 'target', f_str), ('Method', 'method', f_str)] + RUN_COLS +
              [('Speedup vs PACMANN', 'speedup', f_x), ('Source', 'source', f_str)],
              [f'Selection: MRR@10 >= T, then {"lowest " + PICK_COST if MATCH_MODE != "closest_above" else "MRR@10 closest to T"} '
               f'(mode {MATCH_MODE}). Speedup = PACMANN WAN / method WAN at the same target. '
               + FLAG_NOTE.format(tol=TOL)])
    summary = defaultdict(lambda: defaultdict(list))
    for (d, proto), (tv, picks) in table1_picks(R, ctx).items():
        pac = picks['pacmann'][0]
        for m in [PILLAR[proto]] + BASELINES:
            run, flag = picks[m]
            note_pick(1, f'{D(d)} {proto} target {tv:.2f}', m, flag, tv, R, d)
            sp = pac['wan_time'] / run['wan_time'] if run and pac and run['wan_time'] > 0 else None
            t.add(**run_row(run, flag, dataset=D(d), target=f'{proto.capitalize()} {tv:.2f}', method=L(m),
                            speedup=sp if m != 'pacmann' else 1.0))
            if m == PILLAR[proto] and sp and flag == '' and picks['pacmann'][1] == '':
                summary[d][proto].append(sp)
    for d in ctx['t1_datasets']:
        parts = [f"{proto.capitalize()} speedup {f_x(v[0])}" if len(v) == 1 else f"{proto.capitalize()} {f_x(min(v))}–{f_x(max(v))}"
                 for proto in PILLAR for v in [summary[d][proto]] if v]
        t.notes.append(f'{D(d)}: ' + (', '.join(parts) if parts else 'no clean (unflagged) speedup'))
    if summary:
        for proto in PILLAR:
            allv = [x for d in summary for x in summary[d][proto]]
            if allv:
                t.notes.append(f'All datasets, PILLAR-{proto.capitalize()} speedup over PACMANN: min {f_x(min(allv))}, '
                               f'max {f_x(max(allv))} (unflagged rows only)')
    return t


def table2(R, ctx):
    t = Table(2, 'Best quality per dataset',
              [('Dataset', 'dataset', f_str), ('Method', 'method', f_str), ('Config', 'config', f_str),
               ('MRR@10', 'mrr', f_score), ('Recall@10', 'recall', f_score),
               ('% of exact dense Recall@10', 'pct', f_pct), ('Source', 'source', f_str)])
    params, not_run = None, []
    for d in T2_DATASETS:
        js = nonprivate(d)
        dense = js.get('Dense', {}).get('Recall') if js else None
        for key, label in NONPRIVATE:
            v = js.get(key) if js else None
            if isinstance(v, dict) and 'MRR' in v:
                t.add(dataset=D(d), method=label, config=key + (f" (alpha {v['alpha']})" if 'alpha' in v else ''),
                      mrr=v.get('MRR@10', v['MRR']), recall=v.get('Recall@10', v['Recall']), source='non-private',
                      pct=v.get('Recall@10', v['Recall']) / dense if dense else None)
                params = params or js.get('Params')
            else:
                t.add(dataset=D(d), method=label, config='pending', source='non-private')
                MISSING.append(f'Table 2, {D(d)}, {label}: pending (not in {" / ".join(BASELINES_JSON)})')
        for m in T2_METHODS:
            b = best_mrr(R.get(m, d))
            row = run_row(b, '', dataset=D(d), method=L(m), pct=(b['recall'] / dense) if b and dense else None)
            if b is None:
                if (m, d) in T2_NOT_RUN:
                    row['config'] = 'not run'
                    not_run.append(f'{L(m)} on {D(d)}')
                else:
                    MISSING.append(f'Table 2, {D(d)}, {L(m)}: pending (no runs)')
            t.add(**row)
    if params:
        t.notes.append(f"Non-private references (baselines.py): BM25 k1 = {params.get('k1')}, b = {params.get('b')}, analyzer "
                       f"{params.get('analyzer')}. Exact dense = the same e5 (PCA-192) embeddings PILLAR uses, exhaustive "
                       f"cosine search. Fused hybrid (score): union of the BM25 and dense top-{params.get('fusion_depth')} "
                       f"lists, scores min-max normalised per query, ranked by alpha · BM25 + (1 − alpha) · dense with "
                       f"alpha = {params.get('alpha')}. Fused hybrid (RRF): reciprocal rank fusion, "
                       f"1 / ({params.get('rrf_k')} + rank) summed over the two lists.")
    else:
        t.notes.append('Fusion rule: pending (no baselines.json)')
    t.notes.append('Highest-MRR@10 config of each private method (any cost). Tiptoe / Coeus: MRR of the list they return (before the local rerank).')
    if not_run:
        t.notes.append('Not run (no results; could not be run at this scale): ' + ', '.join(not_run) + '.')
    return t


_NONPRIVATE = {}


def nonprivate(d):
    """baselines.py output for d: the combined baselines.json, else rebuttal_numbers' per-dataset <d>_k10.json"""
    if not _NONPRIVATE:
        for p in BASELINES_JSON:
            if os.path.exists(p):
                _NONPRIVATE.update(json.load(open(p)))
                _NONPRIVATE['_path'] = p
                break
        else:
            _NONPRIVATE['_path'] = None
    if d in _NONPRIVATE:
        return _NONPRIVATE[d]
    js, _ = rn.baselines_json(d)   # older per-dataset layout: lower-case keys
    if not js:
        return None
    ren = {'bm25': 'BM25', 'dense': 'Dense', 'hybrid': 'Hybrid', 'hybrid_rrf': 'HybridRRF'}
    return {ren.get(k, k): v for k, v in js.items()}


def doc_count(d):
    js = nonprivate(d)
    if js and js.get('NumDocs'):
        return js['NumDocs'], 'baselines.json'
    meta, _ = docfetch_meta(d)
    if meta and meta.get('NumDocs'):
        return int(meta['NumDocs']), 'docfetch'
    return DOC_COUNTS_FALLBACK.get(d), 'CONFIG fallback'


def scale_picks(R, target):
    return {(d, m): pick(R.get(m, d), target) for d in SCALE_DATASETS for m in SCALE_METHODS}


def table3(R, ctx):
    cols = [('Corpus', 'dataset', f_str), ('Method', 'method', f_str)] + RUN_COLS + \
           [('Speedup vs PACMANN', 'speedup', f_x), ('Source', 'source', f_str)]
    t = Table(3, f'Scalability on MS MARCO (v1 8.8M, v2 subsets), fixed MRR@10 = {T_SCALE:.2f}', cols,
              [FLAG_NOTE.format(tol=TOL)])
    picks = scale_picks(R, T_SCALE)
    for d in SCALE_DATASETS:
        pac = picks[(d, 'pacmann')][0]
        for m in SCALE_METHODS:
            run, flag = picks[(d, m)]
            note_pick(3, f'{D(d)} T {T_SCALE:.2f}', m, flag, T_SCALE, R, d)
            sp = pac['wan_time'] / run['wan_time'] if run and pac and run['wan_time'] > 0 else None
            t.add(**run_row(run, flag, dataset=D(d), method=L(m), speedup=sp))
    best_rows = []
    for d in SCALE_DATASETS:
        for m in SCALE_METHODS:
            b = best_mrr(R.get(m, d))
            best_rows.append(run_row(b, '', dataset=D(d), method=L(m)))
    t.blocks.append(('Best MRR@10 config per size', cols[:-2] + [('Source', 'source', f_str)], best_rows))
    ratio_rows = []
    base = SCALE_DATASETS[0]
    for m in SCALE_METHODS:
        r0, f0 = picks[(base, m)]
        for d in SCALE_DATASETS[1:]:
            r1, f1 = picks[(d, m)]
            row = {'method': L(m), 'step': f'{D(base)} → {D(d)}'}
            for key in ('client_storage_mb', 'wan_time'):
                ok = r0 and r1 and pu.is_number(r0.get(key)) and pu.is_number(r1.get(key)) and r0[key] > 0
                row[key] = (r1[key] / r0[key]) if ok else None
            row['flags'] = ' '.join(x for x in (f0, f1) if x and x != 'pending') or ''
            if f1 == 'pending' or f0 == 'pending':
                row['flags'] = 'pending'
            ratio_rows.append(row)
    t.blocks.append((f'Growth of the fixed-MRR@10 pick (ratio vs {D(base)})',
                     [('Method', 'method', f_str), ('Step', 'step', f_str), ('Client storage ×', 'client_storage_mb', f_x),
                      ('WAN ×', 'wan_time', f_x), ('Flags (* / ~ = a side misses T)', 'flags', f_str)], ratio_rows))
    return t


def table4(R, ctx):
    t = Table(4, f'Client storage vs corpus size (MS MARCO, fixed MRR@10 = {T4_TARGET:.2f})',
              [('Corpus', 'dataset', f_str), ('Docs N', 'n', f_int), ('|V|', 'v', f_int),
               ('PILLAR-Bin storage (MB)', 'bs', f_int), ('Bin MRR@10', 'bm', f_str),
               ('PACMANN storage (MB)', 'ps', f_int), ('PACMANN MRR@10', 'pm', f_str),
               ('PACMANN / Bin', 'ratio', f_x), ('|V| / N', 'vn', lambda v: 'n/a' if v is None else f'{v:.3f}'),
               ('Bin bytes / term', 'bpt', f_int)],
              ['|V| = VocabSize in PILLAR-Bin\'s metadata.json (its own tokenisation / stopword removal). '
               f'Same selection rule as Table 1 (lowest {PICK_COST} at MRR@10 >= T). ' + FLAG_NOTE.format(tol=TOL)])
    nsrc = set()
    for d in SCALE_DATASETS:
        vs = Counter(r.get('vocab_size') for r in R.get('bins', d) if pu.is_number(r.get('vocab_size')))
        v = vs.most_common(1)[0][0] if vs else None
        if v is None:
            MISSING.append(f'Table 4, {D(d)}: |V| missing (no PILLAR-Bin run with VocabSize)')
        b, bf = pick(R.get('bins', d), T4_TARGET)
        p, pf = pick(R.get('pacmann', d), T4_TARGET)
        note_pick(4, D(d), 'bins', bf, T4_TARGET, R, d)
        note_pick(4, D(d), 'pacmann', pf, T4_TARGET, R, d)
        mr = lambda run, f: 'pending' if run is None else fv.r2(run['mrr']) + (f' {f}' if f else '')
        n, src = doc_count(d)
        nsrc.add(src)
        t.add(dataset=D(d), n=n, v=v, bs=b and b['client_storage_mb'], bm=mr(b, bf),
              ps=p and p['client_storage_mb'], pm=mr(p, pf),
              ratio=(p['client_storage_mb'] / b['client_storage_mb']) if b and p else None,
              vn=(v / n) if v and n else None, bpt=(b['client_storage_mb'] * 2 ** 20 / v) if b and v else None)
    t.notes.append(f'Docs N from {", ".join(sorted(nsrc))}. PILLAR-Bin keeps one DB entry per term of its vocabulary '
                   '(the corpus is tokenised, stopwords removed, each term maps to its posting bins), so its client '
                   'storage follows |V|, not N.')
    return t


def table5(R, ctx):
    t = Table(5, 'Query-term ablation (MS MARCO, PILLAR-Tree)',
              [('q*', 'q', f_str), ('Term choice', 'order', f_str), ('Config', 'config', f_str),
               ('Recall@10', 'recall', f_score), ('MRR@10', 'mrr', f_score), ('Source', 'source', f_str)])
    rs = R.get('tree', 'msmarco')
    fam = defaultdict(dict)
    idf = defaultdict(dict)
    for r in rs:
        if r.get('q') is None:
            continue
        key = (r['b'], r['r'], r['s'], r['L'], r.get('k_candidates'))
        (idf if IDF_FOLDER_MARK in r['folder'] else fam)[key][r['q']] = r
    if Q_FAMILY:
        key = tuple(Q_FAMILY)
    else:
        key = max(fam, key=lambda k: (sum(q in fam[k] for q in Q_VALUES),
                                      max(x['recall'] for x in fam[k].values())), default=None)
    if key is None:
        MISSING.append('Table 5: no PILLAR-Tree MS MARCO runs with a q* setting')
        return t
    t.intro.append(f'Family b={key[0]} r={key[1]} s={key[2]} L={key[3]} c={key[4]} (everything but q* fixed).')
    for q in Q_VALUES:
        for order, src in (('first q*', fam), ('highest-IDF q*', idf))[:2 if SHOW_IDF_ROWS else 1]:
            r = src.get(key, {}).get(q)
            if r is None:
                t.add(q=q, order=order, config='pending')
                MISSING.append(f'Table 5, q*={q}, {order}: pending (' +
                               ('the tree only keeps the first q* terms; IDF selection needs a code change + runs, '
                                'or SHOW_IDF_ROWS = False)' if src is idf else 'no run)'))
            else:
                t.add(**run_row(r, '', q=q, order=order))
    qs, p = rn.analyzed_queries()
    if qs:
        n = [len(q['terms']) for q in qs]
        t.notes.append(f'Query length (Lucene analyser, stopwords removed; {len(n)} queries): mean {statistics.mean(n):.2f} '
                       f'terms, {sum(x > 8 for x in n) / len(n):.1%} have more than 8 ({p})')
    else:
        MISSING.append('Table 5: query length stats (no analysed query file)')
    t.notes.append('"first q*" = the tree keeps the first q* analysed terms (current code, normalizeQueryTerms).')
    return t


def table6(R, ctx, buckets=(0.0, 0.25, 0.5, 0.75, 1.0001)):
    t = Table(6, 'Win/loss by query-passage term overlap (MS MARCO, PILLAR-Tree vs PACMANN, reciprocal rank)',
              [('Overlap', 'bucket', f_str), ('% queries', 'share', f_pct), ('Tree wins', 'tree', f_pct),
               ('PACMANN wins', 'pacmann', f_pct), ('Ties', 'tie', f_pct)])
    if WINLOSS_PICK == 'tree_target':
        pk = table1_picks(R, ctx).get(('msmarco', 'tree'))
        tr, pr = (pk[1]['tree'][0], pk[1]['pacmann'][0]) if pk else (None, None)
    else:
        tr, pr = rn.best(R.get('tree', 'msmarco')), rn.best(R.get('pacmann', 'msmarco'))
    if not (tr and pr):
        MISSING.append('Table 6: pending (no Tree or PACMANN MS MARCO runs)')
        return t
    t.intro.append(f'Tree `{tr["config"]}` vs PACMANN `{pr["config"]}` ({WINLOSS_PICK}). Overlap = share of the query\'s '
                   'analysed terms in its best-matching relevant passage; buckets are quarters of [0, 1].')
    tf, pf = rn.run_file(tr, 'results_reRank.json'), rn.run_file(pr, 'results_reRank.json')
    qs, _ = rn.analyzed_queries()
    corpus = os.path.join(g.DATASETS_ROOT, 'msmarco', 'corpus.jsonl')
    miss = [n for n, ok in (('Tree results_reRank.json', tf), ('PACMANN results_reRank.json', pf),
                            ('analysed queries', qs), (corpus, os.path.exists(corpus))) if not ok]
    if miss:
        MISSING.append('Table 6: pending, missing ' + ', '.join(miss))
        for i in range(len(buckets) - 1):
            t.add(bucket=f'{buckets[i]:.2f}–{min(buckets[i + 1], 1):.2f}')
        return t
    rel = rn.qrels('msmarco')
    tres, pres = rn.load_json(tf), rn.load_json(pf)
    terms = {q['id']: set(q['terms']) for q in qs}
    qids = [q for q in terms if q in rel and q in tres and q in pres and terms[q]]
    need = set().union(*(rel[q] for q in qids))
    analyze, note = rn.make_analyzer()
    ptext = {}
    with open(corpus) as fh:
        for line in fh:
            dd = json.loads(line)
            if str(dd['_id']) in need:
                ptext[str(dd['_id'])] = set(analyze((dd.get('title', '') + ' ' + dd.get('text', '')).strip()))
    stats = defaultdict(Counter)
    for q in qids:
        ov = max((len(terms[q] & ptext.get(x, set())) / len(terms[q]) for x in rel[q]), default=0.0)
        bi = next(i for i in range(len(buckets) - 1) if ov < buckets[i + 1])
        a, b = rn.rr_at_k(tres[q], rel[q]), rn.rr_at_k(pres[q], rel[q])
        res = 'tree' if a > b else 'pacmann' if b > a else 'tie'
        stats[bi][res] += 1
        stats['all'][res] += 1
    tot = sum(stats['all'].values())
    for i in list(range(len(buckets) - 1)) + ['all']:
        c, n = stats[i], sum(stats[i].values())
        label = 'overall' if i == 'all' else f'{buckets[i]:.2f}–{min(buckets[i + 1], 1):.2f}'
        t.add(bucket=label, share=n / tot if tot else None,
              **{k: (c[k] / n if n else None) for k in ('tree', 'pacmann', 'tie')})
    t.notes.append(f'{tot} queries; term analyser: {note}. Ties are mostly queries both miss.')
    return t


_RAGAS = None


def ragas_runs():
    """[(method, dataset, run or None, csv path)] for every ragas_output_<method>_k<K>.csv one level under RAGAS_DIRS"""
    global _RAGAS
    if _RAGAS is not None:
        return _RAGAS
    _RAGAS, seen = [], set()
    pat = re.compile(rf'^ragas_output_(?P<m>[a-z]+)_k{K}\.csv$')
    for root in dict.fromkeys(RAGAS_DIRS):
        if not os.path.isdir(root):
            continue
        for e in os.scandir(root):
            if not e.is_dir() or e.name.split('_', 1)[0] not in RAGAS_METHODS or e.name in seen:
                continue
            try:
                files = [f for f in os.listdir(e.path) if f.startswith('ragas_output') and RAGAS_SKIP not in f]
            except OSError:
                continue
            for f in files:
                mm = pat.match(f)
                if not mm:
                    continue
                seen.add(e.name)
                status, method, run = load_results._load_folder(e.name, [(root, 'real')])
                ds = run['dataset'] if run else next((d for d in sorted(DATASET_LABELS, key=len, reverse=True)
                                                      if f'_{d}_' in e.name), None)
                _RAGAS.append((mm['m'], ds, run, os.path.join(e.path, f)))
    log.info("RAGAS: %d per-query CSVs", len(_RAGAS))
    return _RAGAS


def unpaired_bootstrap(a, b, n=None, seed=None):
    """mean(a) - mean(b) and a 95% CI, resampling a and b independently (for runs that scored different queries)"""
    import random
    n, rnd = n or BOOTSTRAP_N, random.Random(BOOTSTRAP_SEED if seed is None else seed)
    ds = sorted(sum(rnd.choice(a) for _ in a) / len(a) - sum(rnd.choice(b) for _ in b) / len(b) for _ in range(n))
    return statistics.mean(a) - statistics.mean(b), ds[int(0.025 * n)], ds[int(0.975 * n)]


def _per_q(path):
    return {row['qid']: float(row['answer_relevancy']) for row in csv.DictReader(open(path))
            if row.get('answer_relevancy') not in (None, '', 'nan')}


def table7(R, ctx):
    t = Table(7, 'RAGAS answer relevancy (PILLAR vs PACMANN)',
              [('Dataset', 'dataset', f_str), ('PILLAR method', 'method', f_str), ('Config', 'config', f_str),
               ('PACMANN config', 'pconfig', f_str), ('N queries', 'n', lambda v: v if isinstance(v, str) else f_int(v)), ('PILLAR', 'a', f_score),
               ('PACMANN', 'b', f_score), ('PILLAR − PACMANN', 'diff', lambda v: 'n/a' if v is None else f'{v:+.3f}'),
               ('95% CI', 'ci', f_str), ('Pairing', 'pairing', f_str)],
              [f'Paired bootstrap over the queries both CSVs score ({BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}). '
               f'Each method\'s config with a RAGAS CSV and the highest MRR@10 ({RAGAS_PAIR}). The RAGAS runs are the '
               'older REAL runs (one per select_configs pick), not the simulated sweep; ragas_output_local_* (15-query '
               'test) left out.'])
    allr = ragas_runs()
    per = defaultdict(list)
    for m, d, run, path in allr:
        per[(m, d)].append((run, path))
    choose = lambda xs: max(xs, key=lambda x: (x[0]['mrr'] if x[0] and pu.is_number(x[0].get('mrr')) else -1, x[1]))
    for d in RAGAS_DATASETS:
        for m in [x for x in RAGAS_METHODS if x != 'pacmann']:
            a, b = per.get((m, d)), per.get(('pacmann', d))
            if not (a and b):
                t.add(dataset=D(d), method=L(m), ci='pending')
                MISSING.append(f'Table 7, {D(d)}, {L(m)} vs PACMANN: pending (no RAGAS CSV for '
                               + ', '.join(L(x) for x, have in ((m, a), ('pacmann', b)) if not have) + ')')
                continue
            (ra, pa), (rb, pb) = choose(a), choose(b)
            qa, qb = _per_q(pa), _per_q(pb)
            common = sorted(set(qa) & set(qb))
            cfg = lambda r, p: r['config'] if r else os.path.basename(os.path.dirname(p))
            if not common:   # different query sets: each method's own mean, unpaired bootstrap (flagged)
                ex = lambda q: ', '.join(sorted(q, key=lambda x: (len(x), x))[:3])
                mean, lo, hi = unpaired_bootstrap(list(qa.values()), list(qb.values()))
                t.add(dataset=D(d), method=L(m), config=cfg(ra, pa), pconfig=cfg(rb, pb), n=f'{len(qa)} / {len(qb)}',
                      a=statistics.mean(qa.values()), b=statistics.mean(qb.values()), diff=mean,
                      ci=f'[{lo:+.3f}, {hi:+.3f}]', pairing='UNPAIRED: different queries')
                t.notes.append(f'{D(d)}, {L(m)} vs PACMANN: UNPAIRED. The two RAGAS runs scored different queries '
                               f'({L(m)} qids {ex(qa)}, … vs PACMANN {ex(qb)}, …), so this compares two means over '
                               f'different query samples (independent bootstrap of each side), not a per-query difference.')
                continue
            mean, lo, hi = rn.bootstrap([qa[q] - qb[q] for q in common], n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            t.add(dataset=D(d), method=L(m), config=cfg(ra, pa), pconfig=cfg(rb, pb), n=len(common),
                  a=statistics.mean(qa[q] for q in common), b=statistics.mean(qb[q] for q in common),
                  diff=mean, ci=f'[{lo:+.3f}, {hi:+.3f}]',
                  pairing='paired' + (f' ({len(common)} shared of {len(qa)} / {len(qb)})'
                                      if len(common) < max(len(qa), len(qb)) else ''))
    rows = []
    for m, d, run, path in sorted(allr, key=lambda x: (x[1] or '', x[0], -(x[2] or {}).get('mrr', 0))):
        q = _per_q(path)
        rows.append({'dataset': D(d), 'method': L(m), 'config': run['config'] if run else os.path.basename(os.path.dirname(path)),
                     'mrr': run and run.get('mrr'), 'recall': run and run.get('recall'), 'n': len(q),
                     'rel': statistics.mean(q.values()) if q else None})
    t.blocks.append(('Every RAGAS CSV found', [('Dataset', 'dataset', f_str), ('Method', 'method', f_str),
                     ('Config', 'config', f_str), ('MRR@10', 'mrr', f_score), ('Recall@10', 'recall', f_score),
                     ('N', 'n', f_int), ('Answer relevancy', 'rel', f_score)], rows))
    return t


def docfetch_meta(d):
    for root in DOCFETCH_DIRS:
        p = os.path.join(root, f'docfetch_{d}_k{K}_flate', 'metadata.json')
        if os.path.exists(p):
            return rn.load_json(p), p
    return None, None


def table8(R, ctx):
    t = Table(8, 'End-to-end latency at fixed MRR@10 (top-10 IDs + document text)',
              [('Dataset', 'dataset', f_str), ('Target', 'target', f_str), ('Method', 'method', f_str),
               ('Config', 'config', f_str), ('MRR@10', 'mrr', f_score),
               ('WAN to top-10 IDs (s)', 'wan_time', f_s), ('Doc text PIR fetch (s)', 'fetch', f_s),
               ('Total WAN (s)', 'total', f_s), ('Round trips', 'rounds', f_str), ('Source', 'source', f_str)],
              ['Same picks as Table 1 (same targets and rule). Doc text fetch = the measured docfetch run of that '
               'dataset (one batched PianoPIR round for the 10 passages), added to every method. Round trips = PIR '
               'rounds to the top-10 IDs + 1 for the fetch. ' + FLAG_NOTE.format(tol=TOL)])
    picks = table1_picks(R, ctx)
    for d in E2E_DATASETS:
        meta, path = docfetch_meta(d)
        fetch = float(meta['WANTimePerQuery']) if meta else None
        if meta is None:
            MISSING.append(f'Table 8, {D(d)}: doc-text fetch time (no docfetch_{d}_k{K}_flate/metadata.json)')
        else:
            t.notes.append(f'{D(d)} doc fetch: WAN {fetch:.3f} s, LAN {float(meta["LANTimePerQuery"]):.3f} s, '
                           f'{float(meta["BytesPerQuery"]) / 1024:,.0f} KB/query, client storage '
                           f'{float(meta["ClientStorageMB"]):,.0f} MB ({meta.get("Compression")} rows; {path})')
        for proto in PILLAR:
            if (d, proto) not in picks:
                continue
            tv, pk = picks[(d, proto)]
            for m in [PILLAR[proto]] + BASELINES:
                run, flag = pk[m]
                row = run_row(run, flag, dataset=D(d), target=f'{proto.capitalize()} {tv:.2f}', method=L(m))
                if run:
                    row['fetch'] = fetch
                    row['total'] = run['wan_time'] + fetch if fetch is not None else None
                    pr = run.get('pir_rounds')
                    if m == 'hybrid':
                        pr = (run['tiptoe'].get('pir_rounds') or 0) + (run['coeus'].get('pir_rounds') or 0)
                    row['rounds'] = f'{pr:.0f} + 1' if pu.is_number(pr) else 'n/a'
                t.add(**row)
    t.notes.append('Tiptoe\'s 2 rounds are ranking + URL; Coeus\'s 2 are scoring + metadata. Neither returns the passage '
                   'text itself, so both get the same + 1 fetch.')
    return t


FILTER_FREE = {5, 6, 7}   # tables the client-storage filter doesn't change
TABLES = {0: table0, 1: table1, 2: table2, 3: table3, 4: table4, 5: table5, 6: table6, 7: table7, 8: table8}


def missing_section():
    out = ['## MISSING / PENDING', '']
    out += [f'- {m}' for m in dict.fromkeys(MISSING)] or ['- nothing']
    needed = defaultdict(set)
    for src in (rh.NEEDED, rn.NEEDED):
        for rd, folders in src.items():
            needed[rd] |= folders
    if needed:
        out += ['', 'Per-query files that are not local (fetch, then run again):', '']
        for rd, folders in needed.items():
            server = '/scratch/dblackle/results_simulated/' if 'simulated' in rd else '/scratch/dblackle/results/'
            inc = ' '.join(f"--include='{fo}/' --include='{fo}/results*.json'" for fo in sorted(folders))
            out += ['```', f"rsync -am {inc} --exclude='*' hopper:{server} {rd}/", '```']
    return '\n'.join(out) + '\n'


def main():
    global MATCH_MODE, TOL, T_SCALE, T4_TARGET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table', nargs='+', type=int, choices=list(TABLES), help='only these tables')
    ap.add_argument('--md', help='write the markdown here')
    ap.add_argument('--csv', help='write each table as CSV into this folder')
    ap.add_argument('--mrr-bin', type=float, help='one Bin target for every Table 1 dataset')
    ap.add_argument('--mrr-tree', type=float, help='one Tree target for every Table 1 dataset')
    ap.add_argument('--t1-datasets', nargs='+', default=T1_DATASETS)
    ap.add_argument('--match', choices=['atleast', 'closest_above', 'window'], default=MATCH_MODE)
    ap.add_argument('--tol', type=float, default=TOL)
    ap.add_argument('--t-scale', type=float, default=T_SCALE)
    ap.add_argument('--storage-filter', choices=['off', 'on', 'both'], default=STORAGE_FILTER,
                    help="client-storage filter: off (every run), on (the paper's cut), both (tables twice)")
    ap.add_argument('--storage-exempt', nargs='*', default=STORAGE_EXEMPT,
                    help='with the filter on: datasets that keep their runs anyway (e.g. scifact)')
    args = ap.parse_args()
    MATCH_MODE, TOL, T_SCALE = args.match, args.tol, args.t_scale
    g.setup_logging()
    modes = ['off', 'on'] if args.storage_filter == 'both' else [args.storage_filter]
    data = load(modes, args.storage_exempt)
    head = [f'# Rebuttal tables (k={K})', '',
            f'Generated by plotting/working_rebuttal.py. Match mode {MATCH_MODE}, TOL {TOL:.2f}, MRR floor {MIN_MRR:.2f}. '
            f'Client-storage filter: {args.storage_filter}' +
            (f' (exempt: {", ".join(args.storage_exempt)})' if args.storage_exempt and 'on' in modes else '') + '. '
            'Every PILLAR / PACMANN number is from the simulated runs (PIR simulated: same answers, bytes, LAN/WAN as a '
            'real run). Tiptoe / Coeus are simulated too, their per-query lists are the real ranking outputs.', '']
    tables = []
    for i, mode in enumerate(modes):
        R = Runs(data[mode])
        ctx = {'t1_datasets': args.t1_datasets, 'global_targets': {'bin': args.mrr_bin, 'tree': args.mrr_tree}}
        start = len(MISSING)
        for n in args.table or list(TABLES):
            if i and n in FILTER_FREE:
                continue   # doesn't depend on the storage filter: once is enough
            try:
                t = TABLES[n](R, ctx)
                if len(modes) > 1 and n not in FILTER_FREE:
                    t.title += f' [client-storage filter {mode}]'
                    t.suffix = f'_filter_{mode}'
                tables.append(t)
            except Exception as err:   # one broken table must not lose the others
                log.exception('table %d failed', n)
                MISSING.append(f'Table {n}: FAILED ({err!r})')
        if len(modes) > 1:
            MISSING[start:] = [f'[filter {mode}] {m}' for m in MISSING[start:]]
    text = '\n'.join(head) + '\n' + '\n'.join(t.markdown() for t in tables) + '\n' + missing_section()
    print(text)
    if args.md:
        open(args.md, 'w').write(text)
        print(f'saved {args.md}', file=sys.stderr)
    if args.csv:
        for t in tables:
            t.write_csv(args.csv)
        print(f'saved CSVs to {args.csv}', file=sys.stderr)


if __name__ == '__main__':
    main()
