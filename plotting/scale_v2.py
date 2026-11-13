"""
Scale table (rebuttal): MS MARCO v1 (8.8M) and the MS MARCO v2 subsets (25M, 50M,
full 138M), PILLAR-Bin vs PILLAR-Tree vs PACMANN, at k=10.

    python scale_v2.py                 # T = 0.05 (T_DEFAULT), Bin bs picked automatically
    python scale_v2.py --mrr auto      # T computed (see below)
    python scale_v2.py --mrr 0.06 --bs 10000000
    FILTER_CLIENT_STORAGE=0 python scale_v2.py
    python scale_v2.py --fixed         # fixed-config table instead (configs chosen at MRR@10 >= 0.10 on v1)
    python scale_v2.py --fixed 0.12 --use bins=vec0_bs10000000_dpb100_vb10_vr30_ib24

--fixed: one config per method (PILLAR-Bin, PILLAR-Tree, PACMANN), the smallest client
storage reaching MRR@10 >= 0.10 (FIXED_MRR) on MS MARCO v1 among the configs run at every
size the method has runs at, kept unchanged at every size. Writes figures/scale_v2_fixed_k<K>.md
and .json.

First a compact paper table at MRR@10 >= T: N, |V|, PILLAR-Bin client storage,
PACMANN client storage, ratio PACMANN / PILLAR-Bin.

Then one markdown table, columns: corpus size N, vocabulary |V| (VocabSize from the
Bin runs: the same Lucene analyser + stopword removal PILLAR-Bin indexes with),
method, config, MRR@10, Recall@10, WAN s/query, comm KB/query, client MB. Rows:
  (a) best    each method's best-MRR@10 config per size (ties: lowest WAN)
  (b) at T    each method's smallest-client-storage config with MRR@10 >= T (ties:
              lowest WAN). PILLAR-Bin is held at one bs across all sizes: the bs at
              which every size with Bin runs reaches T, with the lowest total storage
              summed over those sizes (or --bs). T = T_DEFAULT unless --mrr; with
              --mrr auto, T = the largest 2-dp value every method reaches at every
              size it has runs at = floor(min over (method, size) of the best MRR@10)
Then PILLAR-Bin's PIR databases per size for the configs in the table: rows x
average row bytes (= DB bytes / rows; the BM25 DB has RealBinSize rows, the Vec DB
of a vec0 run one row per stored doc) and the hint (= client storage), and the
ratios 8.8M -> 25M / 50M / 138M of client storage and WAN per method, for (a) and (b).

Runs come from load_results (same filters as everything else: client storage
filter on unless FILTER_CLIENT_STORAGE=0). A method with no runs at a size is
listed as missing and left out of T.

Writes figures/scale_v2_k<K>.md and .json.
"""

import argparse
import json
import math
import os

import globals as g
import load_results
import plot_utils as pu

K = g.K_MAIN
SIZES = [('msmarco', 8_841_823), ('msmarco-v2-25m', 25_000_000), ('msmarco-v2-50m', 50_000_000),
         ('msmarco-v2', 138_364_198)]
METHODS = ['bins', 'tree', 'pacmann']
DP = 2
T_DEFAULT = 0.05   # MRR@10 target for the (b) rows; None = computed
FIXED_MRR = 0.10   # --fixed: configs are chosen on MS MARCO v1 at MRR@10 >= this


def L(m):
    return g.METHOD_LABELS.get(m, m)


def floor_dp(x, dp=DP):
    return math.floor(x * 10 ** dp + 1e-9) / 10 ** dp


def fmt(v, f='.4f'):
    return 'n/a' if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, f)


def nsize(n):
    return f'{n / 1e6:.1f}M'


def runs(nd, m, d):
    return [r for r in nd.get(m, {}).get(d, []) if r['k'] == K and pu.is_number(r.get('mrr'))]


def best(rs):
    return max(rs, key=lambda r: (r['mrr'], -r['wan_time'])) if rs else None


def cheapest_at(rs, t):
    """smallest client storage with MRR@10 >= t (ties: lowest WAN, then highest MRR)"""
    ok = [r for r in rs if r['mrr'] >= t and pu.is_number(r.get('client_storage_mb'))]
    return min(ok, key=lambda r: (r['client_storage_mb'], r['wan_time'], -r['mrr'])) if ok else None


def pick_bs(by_d, t):
    """the Bin bs at which every size with Bin runs reaches t, lowest total storage; None if none does"""
    sizes = [rs for rs in by_d.values() if rs]
    tot = {}
    for bs in {r['bs'] for rs in sizes for r in rs}:
        picks = [cheapest_at([r for r in rs if r['bs'] == bs], t) for rs in sizes]
        if all(picks):
            tot[bs] = sum(p['client_storage_mb'] for p in picks)
    return min(tot, key=tot.get) if tot else None


def vocab(nd, d):
    vs = {int(r['vocab_size']) for r in nd.get('bins', {}).get(d, []) if pu.is_number(r.get('vocab_size'))}
    return max(vs) if vs else None   # one value per corpus; max guards against a run with a partial vocab


def meta(r):
    try:
        with open(os.path.join(r['results_dir'], r['folder'], 'metadata.json')) as fh:
            return json.load(fh)
    except OSError:
        return {}


def bin_dbs(r):
    """[(db name, rows, avg row bytes, DB MB, hint MB)] of one PILLAR-Bin run"""
    def row(name, rows, db_mb, hint_mb):
        rb = db_mb * 2 ** 20 / rows if pu.is_number(rows) and rows and pu.is_number(db_mb) else None
        return name, rows, rb, db_mb, hint_mb
    if r['vec'] == 0:
        m = meta(r)
        vrows = float(m['VecDocsStored']) if 'VecDocsStored' in m else None
        return [row('BM25 bins', r['real_bin_size'], r.get('db_size_mb_bm25'), r.get('client_storage_mb_bm25')),
                row('Vec (embeddings)', vrows, r.get('db_size_mb_vec'), r.get('client_storage_mb_vec'))]
    return [row('bins (one DB)' if r['vec'] == 1 else 'bins (text)', r['real_bin_size'], r['db_size_mb'],
                r['client_storage_mb'])]


def scale_v2(nd, target=T_DEFAULT, bs=None, write=True):
    by = {m: {d: runs(nd, m, d) for d, _ in SIZES} for m in METHODS}
    bests = {m: {d: best(by[m][d]) for d, _ in SIZES} for m in METHODS}
    have = [(m, d) for m in METHODS for d, _ in SIZES if bests[m][d]]
    missing = [(m, d) for m in METHODS for d, _ in SIZES if not bests[m][d]]
    limit = min(have, key=lambda md: bests[md[0]][md[1]]['mrr'])
    t_auto = floor_dp(bests[limit[0]][limit[1]]['mrr'])
    t = target if target is not None else t_auto
    bs_auto = pick_bs(by['bins'], t)
    bs = bs if bs is not None else bs_auto
    at = {m: {d: cheapest_at(by[m][d], t) for d, _ in SIZES} for m in METHODS}
    at['bins'] = {d: cheapest_at([r for r in by['bins'][d] if r['bs'] == bs], t) for d, _ in SIZES}
    V = {d: vocab(nd, d) for d, _ in SIZES}

    out = [f'# Scale: MS MARCO v1 -> v2 subsets (k={K})', '',
           f'T = **{t:.2f}** ' + (f'(given; computed would be {t_auto:.2f})' if target is not None else '(computed)') +
           f'. Computed T = the largest 2-dp MRR@10 every method reaches at every size it has runs at; the limit is '
           f'{L(limit[0])} at {g.DATASET_LABELS.get(limit[1], limit[1])} (best MRR@10 '
           f'{bests[limit[0]][limit[1]]["mrr"]:.4f}).',
           f'MRR>=T rows: the smallest client storage reaching T. PILLAR-Bin held at bs = **{fmt(bs, ",")}** '
           + (f'(given; automatic pick would be {fmt(bs_auto, ",")})' if bs != bs_auto else
              '(automatic: every size reaches T, lowest total storage)') + '.',
           f'Client storage filter: {"on" if g.FILTER_CLIENT_STORAGE else "off"}. '
           f'Runs per method/size: ' + '; '.join(
               f'{L(m)} ' + ', '.join(f'{nsize(n)} {len(by[m][d])}' for d, n in SIZES) for m in METHODS) + '.']
    if missing:
        out.append('Missing (no k=10 runs, left out of T): ' +
                   ', '.join(f'{L(m)} at {g.DATASET_LABELS.get(d, d)}' for m, d in missing) + '.')
    out += ['', f'## Client storage at MRR@10 >= {t:.2f}', '',
            f'PILLAR-Bin at bs = {fmt(bs, ",")}; ratio = PACMANN / PILLAR-Bin.', '',
            '| Corpus size | Vocabulary size | PILLAR-Bin storage (MB) | PACMANN storage (MB) | Ratio |',
            '|---|---|---|---|---|']
    for d, n in SIZES:
        b, p = at['bins'][d], at['pacmann'][d]
        bv = b['client_storage_mb'] if b else None
        pv = p['client_storage_mb'] if p else None
        ratio = pv / bv if bv and pv else None
        out.append(f'| {nsize(n)} ({"v1" if d == "msmarco" else "v2"}) | {fmt(V[d], ",")} | {fmt(bv, ",.1f")} | '
                   f'{fmt(pv, ",.1f")} | {fmt(ratio, ".2f") + ("x" if ratio else "")} |')
    out += ['', '| N | \\|V\\| | Method | Row | Config | MRR@10 | Recall@10 | WAN (s/query) | Comm (KB/query) | '
                'Client (MB) | Source |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    rows_json = []
    for d, n in SIZES:
        for m in METHODS:
            for label, r in (('best', bests[m][d]), (f'MRR>={t:.2f}', at[m][d])):
                if r is None:
                    why = 'no runs' if not by[m][d] else f'does not reach {t:.2f}'
                    out.append(f'| {nsize(n)} | {fmt(V[d], ",")} | {L(m)} | {label} | {why} | | | | | | |')
                    continue
                out.append(f'| {nsize(n)} | {fmt(V[d], ",")} | {L(m)} | {label} | `{r["config"]}` | {fmt(r["mrr"])} | '
                           f'{fmt(r["recall"])} | {fmt(r["wan_time"], ".3f")} | {fmt(r["comm_kb"], ",.1f")} | '
                           f'{fmt(r["client_storage_mb"], ",.1f")} | {r.get("source", "")} |')
                rows_json.append(dict(dataset=d, n=n, vocab=V[d], method=m, row=label, config=r['config'],
                                      mrr=r['mrr'], recall=r['recall'], wan_time=r['wan_time'],
                                      comm_kb=r['comm_kb'], client_storage_mb=r['client_storage_mb'],
                                      source=r.get('source')))

    out += ['', '## PILLAR-Bin PIR databases', '',
            'Rows x average row bytes (DB bytes / rows; DBSizeInBytesMB is whole MB, so small DBs are approximate). '
            'Hint = client storage of that DB. BM25 rows = RealBinSize (the bs setting).', '',
            '| N | \\|V\\| | Row | Config | DB | Rows | Avg row bytes | DB (MB) | Hint (MB) |',
            '|---|---|---|---|---|---|---|---|---|']
    for d, n in SIZES:
        for label, r in (('best', bests['bins'][d]), (f'MRR>={t:.2f}', at['bins'][d])):
            if r is None:
                continue
            for name, nrows, rb, db_mb, hint in bin_dbs(r):
                out.append(f'| {nsize(n)} | {fmt(V[d], ",")} | {label} | `{r["config"]}` | {name} | {fmt(nrows, ",.0f")} | '
                           f'{fmt(rb, ",.1f")} | {fmt(db_mb, ",.0f")} | {fmt(hint, ",.1f")} |')

    out += ['', '## Ratios to MS MARCO v1 (8.8M)', '',
            '| Method | Row | Metric | ' + ' | '.join(f'8.8M -> {nsize(n)}' for _, n in SIZES[1:]) + ' |',
            '|---|---|---|' + '---|' * (len(SIZES) - 1)]
    base_d = SIZES[0][0]
    ratios = {}
    for m in METHODS:
        for label, sel in (('best', bests), (f'MRR>={t:.2f}', at)):
            for key, name in (('client_storage_mb', 'client storage'), ('wan_time', 'WAN')):
                b = sel[m][base_d]
                vals = []
                for d, _ in SIZES[1:]:
                    r = sel[m][d]
                    v = r[key] / b[key] if b and r and b[key] else None
                    vals.append(v)
                    ratios[f'{m}|{label}|{key}|{d}'] = v
                out.append(f'| {L(m)} | {label} | {name} | ' + ' | '.join(fmt(v, '.2f') + ('x' if v else '')
                                                                        for v in vals) + ' |')
    out += ['', f'|V| ratio: ' + ', '.join(f'8.8M -> {nsize(n)} {V[d] / V[base_d]:.2f}x' for d, n in SIZES[1:]
                                           if V[d] and V[base_d]) +
            f'; N ratio: ' + ', '.join(f'{n / SIZES[0][1]:.2f}x' for _, n in SIZES[1:]) + '.']

    text = '\n'.join(out) + '\n'
    print(text)
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'scale_v2_k{K}')
        open(base + '.md', 'w').write(text)
        json.dump(dict(target=t, target_computed=t_auto, bins_bs=bs, bins_bs_auto=bs_auto, vocab=V, rows=rows_json, ratios=ratios,
                       filter_client_storage=g.FILTER_CLIENT_STORAGE), open(base + '.json', 'w'), indent=1)
        print(f'saved {base}.md / .json')
    return text


def pick_fixed(by_d, f):
    """the config with MRR@10 >= f on the base size (MS MARCO v1) and the smallest client storage there (ties: lowest
    WAN), among the configs that also have runs at every other size this method has runs at"""
    base = SIZES[0][0]
    others = [rs for d, rs in by_d.items() if d != base and rs]
    ok = [r for r in by_d[base] if r['mrr'] >= f and pu.is_number(r.get('client_storage_mb'))
          and all(any(o['config'] == r['config'] for o in rs) for rs in others)]
    r = min(ok, key=lambda r: (r['client_storage_mb'], r['wan_time'])) if ok else None
    return r['config'] if r else None


def fixed_configs(nd, f=None, use=None, write=True):
    """One config per method, chosen on MS MARCO v1 at MRR@10 >= f, followed unchanged to the larger corpora: how the
    cost of a fixed config grows with N, and what happens to its MRR@10."""
    f = FIXED_MRR if f is None else f
    use = use or {}
    by = {m: {d: runs(nd, m, d) for d, _ in SIZES} for m in METHODS}
    cfg = {m: use.get(m) or pick_fixed(by[m], f) for m in METHODS}
    sel = {m: {d: next((r for r in by[m][d] if r['config'] == cfg[m]), None) for d, _ in SIZES} for m in METHODS}
    V = {d: vocab(nd, d) for d, _ in SIZES}
    base_d, base_n = SIZES[0]

    def ratio(m, d, key):
        b, r = sel[m][base_d], sel[m][d]
        return r[key] / b[key] if b and r and b[key] else None

    def x(v):
        return fmt(v, '.2f') + ('x' if v else '')

    out = [f'# Scale at a fixed config (k={K})', '',
           f'One config per method, chosen on MS MARCO v1 ({nsize(base_n)}) as the smallest client storage with '
           f'MRR@10 >= {f:.2f} (among configs run at every size the method has runs at), then kept unchanged at '
           f'every size: MRR@10 may drop as N grows. Client storage filter: '
           f'{"on" if g.FILTER_CLIENT_STORAGE else "off"}.', '']
    for m in METHODS:
        out.append(f'- {L(m)}: ' + (f'`{cfg[m]}`' + (' (given)' if m in use else '') if cfg[m] else
                                    f'no config reaches {f:.2f} on MS MARCO v1 and is run at every size'))
    out += ['', '## Client storage', '',
            'Ratio = PACMANN / PILLAR-Bin. In brackets: MRR@10.', '',
            '| Corpus size | Vocabulary size | PILLAR-Bin storage (MB) | PACMANN storage (MB) | Ratio |',
            '|---|---|---|---|---|']
    for d, n in SIZES:
        b, p = sel['bins'][d], sel['pacmann'][d]
        bv, pv = (b['client_storage_mb'] if b else None), (p['client_storage_mb'] if p else None)
        cell = lambda r, v: fmt(v, ',.1f') + (f' ({r["mrr"]:.3f})' if r else '')
        out.append(f'| {nsize(n)} ({"v1" if d == base_d else "v2"}) | {fmt(V[d], ",")} | {cell(b, bv)} | '
                   f'{cell(p, pv)} | {x(pv / bv if bv and pv else None)} |')

    out += ['', '## All methods', '',
            '| N | \\|V\\| | Method | MRR@10 | Recall@10 | WAN (s/query) | Comm (KB/query) | Client (MB) | '
            'Client vs 8.8M | WAN vs 8.8M |',
            '|---|---|---|---|---|---|---|---|---|---|']
    rows_json = []
    for d, n in SIZES:
        for m in METHODS:
            r = sel[m][d]
            if r is None:
                out.append(f'| {nsize(n)} | {fmt(V[d], ",")} | {L(m)} | ' +
                           ('no run of this config' if cfg[m] else 'no config') + ' | | | | | | |')
                continue
            out.append(f'| {nsize(n)} | {fmt(V[d], ",")} | {L(m)} | {fmt(r["mrr"])} | {fmt(r["recall"])} | '
                       f'{fmt(r["wan_time"], ".3f")} | {fmt(r["comm_kb"], ",.1f")} | '
                       f'{fmt(r["client_storage_mb"], ",.1f")} | {x(ratio(m, d, "client_storage_mb"))} | '
                       f'{x(ratio(m, d, "wan_time"))} |')
            rows_json.append(dict(dataset=d, n=n, vocab=V[d], method=m, config=cfg[m], mrr=r['mrr'],
                                  recall=r['recall'], wan_time=r['wan_time'], comm_kb=r['comm_kb'],
                                  client_storage_mb=r['client_storage_mb'],
                                  client_ratio=ratio(m, d, 'client_storage_mb'), wan_ratio=ratio(m, d, 'wan_time')))

    if sel['bins'][base_d]:
        out += ['', '## PILLAR-Bin PIR databases at this config', '',
                '| N | DB | Rows | Avg row bytes | DB (MB) | Hint (MB) |', '|---|---|---|---|---|---|']
        for d, n in SIZES:
            if sel['bins'][d]:
                for name, nrows, rb, db_mb, hint in bin_dbs(sel['bins'][d]):
                    out.append(f'| {nsize(n)} | {name} | {fmt(nrows, ",.0f")} | {fmt(rb, ",.1f")} | '
                               f'{fmt(db_mb, ",.0f")} | {fmt(hint, ",.1f")} |')
    out += ['', 'N vs 8.8M: ' + ', '.join(f'{nsize(n)} {n / base_n:.2f}x' for _, n in SIZES[1:]) +
            '; |V| vs 8.8M: ' + ', '.join(f'{nsize(n)} {V[d] / V[base_d]:.2f}x' for d, n in SIZES[1:]
                                          if V[d] and V[base_d]) + '.']

    text = '\n'.join(out) + '\n'
    print(text)
    if write:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        base = os.path.join(g.FIGURE_DIR, f'scale_v2_fixed_k{K}')
        open(base + '.md', 'w').write(text)
        json.dump(dict(fixed_mrr=f, configs=cfg, given=sorted(use), vocab=V, rows=rows_json,
                       filter_client_storage=g.FILTER_CLIENT_STORAGE), open(base + '.json', 'w'), indent=1)
        print(f'saved {base}.md / .json')
    return text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mrr', default=None, help=f'MRR@10 target T (default {T_DEFAULT}); "auto" = computed')
    ap.add_argument('--bs', type=int, help='hold PILLAR-Bin at this bs instead of the automatic pick')
    ap.add_argument('--fixed', nargs='?', type=float, const=FIXED_MRR, default=None, metavar='MRR',
                    help=f'only the fixed-config table: one config per method, chosen on MS MARCO v1 at MRR@10 >= MRR '
                         f'(default {FIXED_MRR}), followed unchanged across sizes')
    ap.add_argument('--use', action='append', default=[], metavar='METHOD=CONFIG',
                    help='with --fixed: use this config for a method instead of the automatic pick '
                         '(e.g. --use bins=vec0_bs10000000_dpb100_vb10_vr30_ib24); repeatable')
    args = ap.parse_args()
    target = T_DEFAULT if args.mrr is None else None if args.mrr == 'auto' else float(args.mrr)
    g.setup_logging()
    nd = load_results.load_results()
    if args.fixed is not None:
        fixed_configs(nd, args.fixed, dict(u.split('=', 1) for u in args.use))
    else:
        scale_v2(nd, target, args.bs)
