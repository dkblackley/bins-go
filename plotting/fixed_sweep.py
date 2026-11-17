"""
The drawing code behind plot_fixed_mrr.py and plot_fixed_storage.py. Both sweep one fixed value (an MRR@10, a
client storage) along x and, at each value, pick one config per method from the configs that match it:

    x       the fixed values, cfg.TARGETS[dataset] (or cfg.auto_targets(...) for a dataset not listed)
    y       one panel row per key in cfg.ROWS ('wan_time', 'mrr', ...)
    points  each method's pick at each x, joined by a line in the method's colour. At each x the method with
            the best y (WAN lower, MRR@10 higher, compared at WIN_DP decimals; a tie goes to the lower WAN,
            WIN_TIE_BREAK) is drawn bigger with a black outline and on top, the others plain. POINTS = 'fade' also draws every other matching
            config faintly.

Each plot file is the config: this module reads its CONFIG block (cfg.<NAME>) and calls its
candidates(runs, target, dataset) / pick(candidates, y_key) / auto_targets(per_method) / x_tick(value) / run_x(run).
Y axes use plot_dataset_costs's axis code, so PAD / NICE_ENDS / TICK_SIG there apply here too.
"""

import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import frozen_values as fv
import globals as g
import plot_dataset_costs as pdc
import plot_utils as pu


# ======================================================================
# RUNS AND PICKS
# ======================================================================

def usable_runs(nested_data, cfg, dataset, rows):
    """{method: runs} with every key the plot needs, above the MRR@10 floor; methods without runs left out."""
    need = set(rows) | set(cfg.NEED)
    out = {}
    for m in cfg.METHODS or g.METHOD_ORDER:
        runs = pu.get_runs(nested_data, m, dataset, k=cfg.K, **(cfg.BINS_FILTER if m == 'bins' else {}))
        runs = [r for r in runs if all(pu.is_number(r.get(x)) for x in need)
                and (not cfg.MIN_MRR or fv.rkey(r['mrr']) >= fv.rkey(cfg.MIN_MRR))]
        if runs:
            out[m] = runs
    return out


def sweep(cfg, per_method, targets, dataset, rows):
    """{y_key: {method: [(target, matching runs, pick or None), ...]}}: cfg.pick(cands, y_key) may pick a different
    config for each panel row (plot_fixed_storage.PICK_BY_ROW)."""
    out = {}
    for y_key in rows:
        out[y_key] = {}
        for m, runs in per_method.items():
            out[y_key][m] = []
            for t in targets:
                cands = cfg.candidates(runs, t, dataset)
                out[y_key][m].append((t, cands, cfg.pick(cands, y_key) if cands else None))
    return out


# ======================================================================
# AXES AND STYLE
# ======================================================================

def axis_spec(cfg, key, dataset):
    spec = {**pdc.AXIS_FALLBACK, **cfg.AXES.get(key, {}), **cfg.AXES.get((dataset, key), {}), 'key': key}
    if spec['label'] is None:
        text = g.label(key, cfg.K)
        spec['label'] = re.sub(r'\s*\([^)]*\)$', '', text) if spec['fmt'] in ('time', 'bytes') else text
    return spec


def offsets(cfg, methods, targets):
    """x shift per method (DODGE x the smallest gap between targets, methods spread round 0), and that gap."""
    gaps = [b - a for a, b in zip(targets, targets[1:]) if b > a]
    step = min(gaps) if gaps else (abs(targets[0]) or 1.0)
    n = len(methods)
    return {m: (i - (n - 1) / 2) * cfg.DODGE * step for i, m in enumerate(methods)}, step


def method_z(cfg, method, layer):
    order = list(cfg.STACK_ORDER or g.METHOD_ORDER)
    rank = order.index(method) if method in order else len(order)
    return layer + 0.9 * (len(order) - rank) / (len(order) + 1)


def style(cfg, method, won):
    """Marker style of a pick: the winner at its x (bigger, outlined, on top) or the rest (plain, no outline)."""
    color = g.METHOD_COLORS[method]
    return dict(linestyle='none',
                marker=g.METHOD_MARKERS[method] if cfg.PER_METHOD_MARKERS else cfg.MARKER,
                markersize=cfg.MARKER_SIZE * (cfg.WIN_SIZE if won else cfg.LOSE_SIZE),
                markerfacecolor=color,
                markeredgecolor='#000000' if won and cfg.WIN_EDGE else color,
                markeredgewidth=cfg.WIN_EDGE if won else 0,
                alpha=cfg.WIN_ALPHA if won else cfg.LOSE_ALPHA,
                zorder=method_z(cfg, method, cfg.Z_WIN if won else cfg.Z_LOSE))


def winners(cfg, sw, y_key):
    """{target: {the one winning method}}: best y rounded down to WIN_DP decimals; a tie there goes to the better
    WIN_TIE_BREAK (lower WAN by default), then to the better raw y. Targets where fewer than two methods have a
    pick get no winner."""
    sign, tb = pdc.better_sign(y_key), cfg.WIN_TIE_BREAK
    tb_sign = pdc.better_sign(tb) if tb else 1

    def rank(p):
        tie = tb_sign * p[tb] if tb and pu.is_number(p.get(tb)) else -math.inf
        return sign * fv.rkey(p[y_key], cfg.WIN_DP), tie, sign * p[y_key]

    out = {}
    targets = [row[0] for row in next(iter(sw.values()))]   # every method's rows share the same targets
    for i, t in enumerate(targets):
        picks = {m: rows[i][2] for m, rows in sw.items() if rows[i][2] is not None}
        out[t] = {max(picks, key=lambda m: rank(picks[m]))} if len(picks) > 1 else set()
    return out


# ======================================================================
# DRAWING
# ======================================================================

def draw_panel(ax, cfg, sw, y_key, off, pos):
    """One panel; returns the y values drawn (for the limits). pos(t) = where target t sits on the x axis."""
    ys = []
    won = winners(cfg, sw, y_key)
    for m, rows in sw.items():
        dx = off[m]
        if cfg.POINTS == 'fade':   # every other matching config, faint (off by default)
            fx, fy = [], []
            for t, cands, pick in rows:
                for r in cands:
                    if r is not pick:
                        fx.append(pos(cfg.run_x(r)) if cfg.CANDIDATE_X == 'actual' else pos(t) + dx)
                        fy.append(r[y_key])
            if fx:
                ax.plot(fx, fy, linestyle='none', marker=cfg.MARKER, markersize=cfg.MARKER_SIZE * cfg.FADE_SIZE,
                        color=g.METHOD_COLORS[m], markeredgewidth=0, alpha=cfg.FADE_ALPHA,
                        zorder=method_z(cfg, m, cfg.Z_FADED))
                ys += fy
        picked = [(t, p) for t, _, p in rows if p is not None]
        if cfg.LINES and len(picked) > 1:
            if cfg.LINE_ACROSS_GAPS:
                lx, ly = [pos(t) + dx for t, _ in picked], [p[y_key] for _, p in picked]
            else:   # NaN breaks the line where a method has no config at a target
                lx = [pos(t) + dx for t, _, _ in rows]
                ly = [p[y_key] if p is not None else math.nan for _, _, p in rows]
            ax.plot(lx, ly, color=g.METHOD_COLORS[m], linewidth=cfg.LINE_WIDTH, linestyle=cfg.LINE_STYLE,
                    alpha=cfg.LINE_ALPHA, zorder=method_z(cfg, m, cfg.Z_LINES))
        for flag in (False, True):
            sel = [(t, p) for t, p in picked if (m in won[t]) == flag]
            if sel:
                ax.plot([pos(t) + dx for t, _ in sel], [p[y_key] for _, p in sel], **style(cfg, m, flag))
        ys += [p[y_key] for _, p in picked]
        if cfg.LOG_POINTS:
            for t, cands, p in rows:
                if p is not None:
                    pu.log.info("%-8s t=%-6g %s=%-10.4g (%d matching) %s", m, t, y_key, p[y_key], len(cands),
                                p['folder'])
    return ys


def add_legend(fig, cfg, present):
    rows = cfg.LEGEND_ROWS or [list(r) for r in g.METHOD_LEGEND_ROWS]
    style_kw = {**g.LEGEND_STYLE, **g.METHOD_LEGEND_STYLE, **cfg.LEGEND_STYLE}
    gap = g.METHOD_LEGEND_LINE_GAP / 72 / fig.get_figheight()
    fig.draw_without_rendering()
    top = fig.get_tightbbox().y0 / fig.get_figheight()
    for row in rows:
        handles = []
        for m in row:
            if m not in present:
                continue
            color = g.METHOD_COLORS[m]
            handles.append(Line2D([], [], color=color, linewidth=cfg.LINE_WIDTH,
                                  linestyle=cfg.LINE_STYLE if cfg.LINES else 'none',
                                  marker=g.METHOD_MARKERS[m] if cfg.PER_METHOD_MARKERS else cfg.MARKER,
                                  markersize=cfg.LEGEND_MARKER_SIZE, markerfacecolor=color, markeredgecolor=color,
                                  label=g.METHOD_LEGEND_LABELS.get(m, g.METHOD_LABELS.get(m, m))))
        if not handles:
            continue
        leg = fig.legend(handles, [h.get_label() for h in handles], loc='upper center',
                         bbox_to_anchor=(0.5, top - gap), ncol=len(handles), **style_kw)
        fig.draw_without_rendering()
        top = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0


# ======================================================================
# FIGURE
# ======================================================================

def figure(nested_data, cfg, datasets, rows, name):
    """One figure: a block of len(rows) panels per dataset, N_COLS blocks per line."""
    blocks = []
    for d in datasets:
        per_method = usable_runs(nested_data, cfg, d, rows)
        if not per_method:
            pu.warn_once("%s: no usable runs on %s, skipped", cfg.__name__, d)
            continue
        targets = sorted(cfg.TARGETS.get(d) or cfg.auto_targets(per_method))
        if not targets:
            pu.warn_once("%s: no targets on %s, skipped", cfg.__name__, d)
            continue
        blocks.append((d, targets, sweep(cfg, per_method, targets, d, rows)))
    if not blocks:
        return None

    n_keys = len(rows)
    n_cols = min(cfg.N_COLS or len(blocks), len(blocks))
    n_lines = math.ceil(len(blocks) / n_cols)
    fig = plt.figure(figsize=(cfg.PANEL_SIZE[0] * n_cols, cfg.PANEL_SIZE[1] * n_lines * n_keys), layout='constrained')
    fig.get_layout_engine().set(wspace=cfg.WSPACE, hspace=cfg.HSPACE)
    axes = fig.subplots(n_lines * n_keys, n_cols, squeeze=False)

    cell, ys = {}, {}   # (row, col) -> (dataset, y_key), y values
    for b, (d, targets, sw) in enumerate(blocks):
        line, j = divmod(b, n_cols)
        if cfg.X_CATEGORICAL:   # targets evenly spaced, in order; anything between two targets interpolated
            idx = list(range(len(targets)))
            pos = lambda v, _t=targets, _i=idx: float(np.interp(v, _t, _i))
            off, step = offsets(cfg, list(sw[rows[0]]), idx)
            ticks = idx
        else:
            pos = lambda v: v
            off, step = offsets(cfg, list(sw[rows[0]]), targets)
            ticks = targets
        for kk, y_key in enumerate(rows):
            i = line * n_keys + kk
            ax = axes[i][j]
            cell[i, j] = (d, y_key)
            ax.grid(cfg.GRID, which='major')
            ax.tick_params(labelsize=cfg.TICK_SIZE)
            ys[i, j] = draw_panel(ax, cfg, sw[y_key], y_key, off, pos)
            ax.set_xlim(ticks[0] - step * cfg.X_PAD, ticks[-1] + step * cfg.X_PAD)
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.xaxis.set_major_formatter(ticker.FixedFormatter([cfg.x_tick(t) for t in targets]))
            if cfg.X_TICK_ROTATION:
                ax.tick_params(axis='x', labelrotation=cfg.X_TICK_ROTATION)
            if kk == 0 and cfg.PANEL_TITLE:
                ax.set_title(cfg.PANEL_TITLE.format(dataset=g.DATASET_LABELS.get(d, d)), fontsize=cfg.TITLE_SIZE)

    bottom = {j: max(i for i, c in cell if c == j) for _, j in cell}
    for (i, j), (d, y_key) in cell.items():
        ax = axes[i][j]
        spec = axis_spec(cfg, y_key, d)
        vals = [v for c in range(n_cols) for v in ys.get((i, c), [])] if cfg.SHARE_Y_ROW else ys[i, j]
        y_log = pdc.apply_axis(ax, 'y', spec, pdc.fit_limits(spec, vals))
        if cfg.X_LABEL and (i == bottom[j] or (cfg.X_LABEL_EVERY_ROW and i % n_keys == n_keys - 1)):
            ax.set_xlabel(cfg.X_LABEL, fontsize=cfg.X_LABEL_SIZE)
        if cfg.Y_LABEL_EVERY_COL or j == 0:
            text = ax.set_ylabel(spec['label'] + (cfg.LOG_SUFFIX if y_log else ''), fontsize=cfg.Y_LABEL_SIZE)
            if cfg.SHOW_ARROWS:
                g.add_arrow(text, y_key, 'y')
    for i in range(n_lines * n_keys):
        for j in range(n_cols):
            if (i, j) not in cell:
                axes[i][j].set_visible(False)

    if cfg.ALIGN_LABELS:
        for j in range(n_cols):
            fig.align_ylabels(axes[:, j])
        for i in range(n_lines * n_keys):
            fig.align_xlabels(axes[i, :])
    if cfg.SUPTITLE:
        ds = blocks[0][0] if len(blocks) == 1 else ''
        fig.suptitle(cfg.SUPTITLE.format(dataset=g.DATASET_LABELS.get(ds, ds)), fontsize=cfg.SUPTITLE_SIZE)
    if cfg.LEGEND:
        add_legend(fig, cfg, {m for _, _, sw in blocks for m in sw[rows[0]]})

    path = pu.save_figure(fig, name)
    if cfg.PRINT_TABLE:
        print_table(cfg, blocks, rows, name)
    return path


# ======================================================================
# TABLES
# ======================================================================

def fmt_value(key, v):
    if key == 'client_storage_mb' or key == 'db_size_mb':
        return f'{v / 1024:.2f} GB'
    if key.startswith('comm'):
        return f'{v:,.1f} KB'
    if key in ('mrr', 'recall', 'mrr_pre_rerank', 'recall_pre_rerank') or key.endswith('_time'):
        return fv.r2(v)   # rounded down to 2 dp, like frozen_values.py
    return f'{v:.4g}'


def print_table(cfg, blocks, rows, name):
    """A table per dataset (a row per method, a column per target and y key), a dashed line, then the picks."""
    clean, details = [], []
    for d, targets, sw in blocks:
        ds = g.DATASET_LABELS.get(d, d)
        arrow = {y: '↓' if pdc.better_sign(y) < 0 else '↑' for y in rows}
        heads = [f"{cfg.X_TABLE.format(t=t)} {axis_spec(cfg, y, d)['label']} {arrow[y]}" for t in targets for y in rows]
        title = cfg.TABLE_TITLE() if callable(cfg.TABLE_TITLE) else cfg.TABLE_TITLE
        clean += [f'### {ds}: {title}', '', '| Method | ' + ' | '.join(heads) + ' |',
                  '|---|' + '---|' * len(heads)]
        for m in sw[rows[0]]:
            cells = [fmt_value(y, sw[y][m][k][2][y]) if sw[y][m][k][2] is not None else 'n/a'
                     for k in range(len(targets)) for y in rows]
            clean.append(f"| {g.METHOD_LABELS.get(m, m)} | " + ' | '.join(cells) + ' |')
        # the configs behind the cells: once if every row uses the same picks, else once per row
        same = all(sw[y][m][k][2] is sw[rows[0]][m][k][2] for y in rows for m in sw[y] for k in range(len(targets)))
        for y in rows[:1] if same else rows:
            tag = '' if same else f" [{axis_spec(cfg, y, d)['label']} row]"
            for m, picks in sw[y].items():
                for t, cands, p in picks:
                    details.append(f"{ds}{tag} {m:<8} {cfg.X_TABLE.format(t=t):<18} {len(cands):>3} matching  "
                                   + (f"{p['config']}  ({p['folder']})" if p is not None else 'none'))
        clean.append('')
    text = '\n'.join(clean + ['n/a = no config of that method matches the target.', '-' * 72] + details)
    print('\n' + text)
    if cfg.WRITE_FILES:
        os.makedirs(g.FIGURE_DIR, exist_ok=True)
        path = os.path.join(g.FIGURE_DIR, f'{name}.md')
        with open(path, 'w') as fh:
            fh.write(text + '\n')
        pu.log.info('saved %s', path)


def make_figures(nested_data, cfg, datasets=None, rows=None, name=None):
    """cfg.FIGURE_GROUPS: one figure per group of datasets (<name>_<d1>_<d2>.pdf; with `datasets` given, they are
    split into groups of N_COLS). cfg.PER_DATASET: one figure per dataset. Else one figure with every dataset."""
    rows, name = list(rows or cfg.ROWS), name or cfg.FIG_NAME
    base = name.replace('_{dataset}', '').replace('{dataset}', '')
    if cfg.PER_DATASET:
        if '{dataset}' not in name:
            name += '_{dataset}'
        return [figure(nested_data, cfg, [d], rows, name.format(dataset=d)) for d in (datasets or cfg.DATASETS)]
    if datasets:
        n = cfg.N_COLS or len(datasets)
        groups = [list(datasets[i:i + n]) for i in range(0, len(datasets), n)] if cfg.FIGURE_GROUPS else [list(datasets)]
    else:
        groups = [list(gr) for gr in cfg.FIGURE_GROUPS] if cfg.FIGURE_GROUPS else [list(cfg.DATASETS)]
    if len(groups) == 1:
        return [figure(nested_data, cfg, groups[0], rows, base)]
    return [figure(nested_data, cfg, gr, rows, base + '_' + '_'.join(gr)) for gr in groups]
