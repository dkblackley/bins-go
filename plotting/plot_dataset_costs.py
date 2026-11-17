"""
One figure per dataset: quality (rows) against cost (columns), every config of
every method as a point, the Pareto frontier over all methods as a black line.

    figures/dataset_costs_msmarco.pdf
    figures/dataset_costs_scifact.pdf
    ...                                  (FIG_NAME, one per entry of DATASETS)

By default that is MRR@10 / Recall@10 down the rows and WAN latency, LAN
latency and client storage across the columns, i.e. a 2x3 grid per dataset and
six figures for g.REBUTTAL_DATASETS.

Everything you'd want to change is in the CONFIG block below:
    ROWS / COLS        which run keys go on y (one per row) and x (one per column)
    DATASETS           which datasets get a figure
    AXES               per run key (or per (dataset, run key)) axis settings:
                       label, lim, log, fmt, ticks
    the rest           points, frontier, layout, legend, title, file name

    python plot_dataset_costs.py
or from main.py:
    plot_dataset_costs.make_plots(nested_data)
    plot_dataset_costs.make_plots(nested_data, cols=['wan_time', 'comm_kb'], name='dataset_comm_{dataset}')
"""

import math
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

# ======================================================================
# CONFIG: WHAT IS PLOTTED
# ======================================================================

K = g.K_MAIN

DATASETS = g.REBUTTAL_DATASETS   # one figure each; g.DATASETS follows the DATASETS=... env var instead

ROWS = ['mrr', 'recall']         # y key of each row (any run key: 'mrr_pre_rerank', 'faithfulness', ...)
COLS = ['wan_time', 'lan_time', 'client_storage_mb']   # x key of each column. Others: 'total_time',
                                 # 'comm_kb', 'maintenance_time', 'rerank_time', 'db_size_mb', 'pir_rounds'
                                 # (swap the two lists to put quality on x and cost on y)

METHODS = None                   # None = g.METHOD_ORDER; or e.g. ['bins', 'tree', 'pacmann'] to drop some
BEST_N = None                    # configs per method via select_configs.py; None = every config
BINS_FILTER = {}                 # e.g. {'vec': 1}: only single-DB bins runs

# ======================================================================
# CONFIG: THE AXES
# ======================================================================

# Run key -> axis settings, and (dataset, run key) -> the same, which wins for
# that dataset only. Anything not set falls back to AXIS_FALLBACK.
#
#   label   axis name; None = g.label(key) with any '(unit)' dropped when fmt carries the unit
#   lim     (lo, hi) in the unit the key is STORED in (seconds, MB for client_storage_mb, KB for
#           comm_kb), or None to fit the data. Either end may be None: (0, None) = start at 0
#   log     log axis (an axis with a value <= 0 falls back to plain)
#   fmt     tick labels: 'time' (seconds -> '12ms', '1.5s'), 'bytes' (with 'base' = stored unit,
#           -> '340MB', '1.2GB'), 'plain' ('0.25'), a '%.2f' string, or a function(value, pos)
#   base    stored unit for fmt='bytes': 'KB' (comm_kb) or 'MB' (client_storage_mb, db_size_mb)
#   ticks   number of ticks, first and last on the ends of the axis; None = matplotlib's own
AXES = {
    'wan_time':          dict(label='WAN Latency', fmt='time', log=True),
    'lan_time':          dict(label='LAN Latency', fmt='time', log=True),
    'total_time':        dict(label='Computation', fmt='time', log=True),
    'maintenance_time':  dict(label='Preprocessing', fmt='time', log=True),
    'comm_kb':           dict(label='Data Sent', fmt='bytes', base='KB', log=True),
    'client_storage_mb': dict(label='Client Storage', fmt='bytes', base='MB', log=False, lim=(0, None)),
    'db_size_mb':        dict(fmt='bytes', base='MB', log=True),
    'mrr':               dict(fmt='plain', lim=(0, None)),
    'recall':            dict(fmt='plain', lim=(0, None)),

    # per-dataset pins, e.g.
    # ('msmarco', 'client_storage_mb'): dict(lim=(0, 6500)),
    # ('scifact', 'wan_time'):          dict(lim=(0.01, 1)),
    # ('msmarco', 'mrr'):               dict(lim=(0, 0.35)),
    # On a log axis a lower end of None (or 0) is fitted to the data; ticks still split the range in 4.

    ('fever', 'mrr'):               dict(lim=(0, 0.85)),
    ('fever', 'recall'):            dict(lim=(0, 0.9)),
    ('fever', 'wan_time'):          dict(lim=(0, 2)),        # seconds
    ('fever', 'lan_time'):          dict(lim=(0, 0.5)),      # seconds
    ('fever', 'client_storage_mb'): dict(lim=(0, 4 * 1024)),    # MB: 3 GB

    # Set lim=None on any line to go back to fitting the data.
    ('msmarco', 'mrr'):               dict(lim=(0, 0.35)),
    ('msmarco', 'recall'):            dict(lim=(0, 0.62)),
    ('msmarco', 'wan_time'):          dict(lim=(0, 2.0)),        # seconds
    ('msmarco', 'lan_time'):          dict(lim=(0, 0.75)),      # seconds
    ('msmarco', 'client_storage_mb'): dict(lim=(0, 6 * 1024)),    # MB

    ('scifact', 'mrr'):               dict(lim=(0, 0.75)),
    ('scifact', 'recall'):            dict(lim=(0, 0.9)),
    ('scifact', 'wan_time'):          dict(lim=(0, 0.4)),        # seconds
    ('scifact', 'lan_time'):          dict(lim=(0, 0.05)),      # seconds
    ('scifact', 'client_storage_mb'): dict(lim=(0, 0.11 * 1024)),    # MB

    ('nq', 'mrr'):                    dict(lim=(0, 0.55)),
    ('nq', 'recall'):                 dict(lim=(0, 0.78)),
    ('nq', 'wan_time'):               dict(lim=(0, 2.5)),        # seconds
    ('nq', 'lan_time'):               dict(lim=(0, 0.275)),      # seconds
    ('nq', 'client_storage_mb'):      dict(lim=(0, 1.0 * 1024)),    # MB

    ('msmarco-v2-25m', 'mrr'):               dict(lim=(0, 0.3)),
    ('msmarco-v2-25m', 'recall'):            dict(lim=(0, 0.45)),
    ('msmarco-v2-25m', 'wan_time'):          dict(lim=(0, 2.5)),        # seconds
    ('msmarco-v2-25m', 'lan_time'):          dict(lim=(0, 1.5)),      # seconds
    ('msmarco-v2-25m', 'client_storage_mb'): dict(lim=(0, 10 * 1024)),    # MB

    ('msmarco-v2-50m', 'mrr'):               dict(lim=(0, 0.21)),
    ('msmarco-v2-50m', 'recall'):            dict(lim=(0, 0.35)),
    ('msmarco-v2-50m', 'wan_time'):          dict(lim=(0, 5)),        # seconds
    ('msmarco-v2-50m', 'lan_time'):          dict(lim=(0, 2.5)),      # seconds
    ('msmarco-v2-50m', 'client_storage_mb'): dict(lim=(0, 14 * 1024)),    # MB
}

AXIS_FALLBACK = dict(label=None, lim=None, log=False, fmt='plain', base=None, ticks=4)

PAD = 0.04            # margin round an auto-fitted end, as a fraction of the range (log range on log axes)
NICE_ENDS = True      # round auto-fitted ends outward to TICK_SIG figures, so the end ticks read cleanly
TICK_SIG = 2          # significant figures in the tick labels

SHARE_Y_ROW = True    # every panel of a row gets the same y limits (they already share the y key)
SHARE_X_COL = True    # every panel of a column gets the same x limits

# ======================================================================
# CONFIG: POINTS AND LINES
# ======================================================================

POINTS = 'fade'       # 'all' every config solid, 'frontier' only frontier points, 'fade' frontier solid + rest faint
FADE_ALPHA = 0.35
FADE_SIZE = 0.6       # faded marker size as a fraction of MARKER_SIZE

MARKER = 'o'
PER_METHOD_MARKERS = False   # True: g.METHOD_MARKERS shape per method
MARKER_SIZE = 5
FRONTIER_EDGE = 0.6   # black outline on points on the global frontier (0 = none)

FRONTIER_LINE = True
FRONTIER_COLOR = '#000000'
FRONTIER_WIDTH = g.LINE_WIDTH + 0.3
FRONTIER_STYLE = '-'
FRONTIER_STEPS = False   # staircase instead of straight segments
FRONTIER_LABEL = 'Pareto Frontier'
# Drawing order (higher = on top): every other config, then the frontier line, then the points on it.
Z_POINTS = 2             # configs off the frontier, faded or not, and METHOD_LINES
Z_FRONTIER_LINE = 4      # the black line, above every config
Z_FRONTIER_POINTS = 5    # the outlined points on the frontier, above the line
# Within each layer the methods stack in g.METHOD_ORDER: the first (bins) on top, then tree, pacmann, ...
# Each gets a fraction under 1 added, so a method never jumps to another layer. Reverse it with
# STACK_ORDER = list(reversed(g.METHOD_ORDER)), or name your own order (first = on top).
STACK_ORDER = None       # None = g.METHOD_ORDER

METHOD_LINES = False  # also draw each method's own frontier in its colour

# ======================================================================
# CONFIG: LAYOUT, TITLES, LEGEND, OUTPUT
# ======================================================================

PANEL_SIZE = (1.6, 1.15)   # inches per panel (width, height). Fonts are fixed in points, so shrinking this
                          # makes the panels smaller and the text relatively bigger, e.g. (1.8, 1.35)
WSPACE = 0.01             # constrained-layout gaps (figure fractions)
HSPACE = 0.02

SUPTITLE = 'PPRAG {dataset} Comparison'    # '{dataset}' -> g.DATASET_LABELS name; None for no title
SUPTITLE_SIZE = 16

X_LABEL_SIZE = g.FONT_SIZE
Y_LABEL_SIZE = g.FONT_SIZE
TICK_SIZE = 8
X_LABEL_EVERY_ROW = False # False: only the bottom row gets x labels (the column shares its x key)
Y_LABEL_EVERY_COL = False # False: only the left column gets y labels
HIDE_INNER_TICKLABELS = False   # with shared limits, also drop the inner panels' tick numbers
ALIGN_LABELS = True       # line up the y labels down each column (and x labels along each row), so
                          # 'MRR@10' and 'Recall@10' sit at the same x even when their ticks differ in width
SHOW_ARROWS = True        # better-direction badge after each axis label
LOG_SUFFIX = ' (log)'     # appended to the label of any axis drawn on a log scale; '' for none
GRID = True

LEGEND = True
LEGEND_ROWS = None        # None = g.METHOD_LEGEND_ROWS with the frontier added to the last row;
                          # otherwise e.g. [['bins', 'tree', 'pacmann', 'frontier']]
LEGEND_MARKER_SIZE = 6
LEGEND_STYLE = dict(handlelength=0.8, fontsize=12)

FIG_NAME = 'dataset_costs_{dataset}'   # figures/<this>.pdf

PRINT_FRONTIER = True     # print each panel's frontier points to the terminal
WRITE_FILES = g.WRITE_TEXT_FILES   # also write them to figures/<name>_frontier.md (default off, see globals.py)
LOG_POINTS = False        # log every point drawn with its folder (python load_results.py <folder>)


# ======================================================================
# FRONTIER
# ======================================================================

def better_sign(key):
    return -1 if g.METRICS.get(key, {}).get('better') == 'lower' else 1


def dominates_fn(x_key, y_key):
    sx, sy = better_sign(x_key), better_sign(y_key)

    def dominates(a, b):
        ax_, ay, bx, by = sx * a[x_key], sy * a[y_key], sx * b[x_key], sy * b[y_key]
        return ax_ >= bx and ay >= by and (ax_ > bx or ay > by)
    return dominates


def frontier_ids(runs, x_key, y_key):
    dom = dominates_fn(x_key, y_key)
    return {id(r) for r in runs if not any(dom(o, r) for o in runs if o is not r)}


def frontier(runs, x_key, y_key):
    """Non-dominated runs sorted by x, exact duplicates kept once."""
    ids = frontier_ids(runs, x_key, y_key)
    front, seen = [], set()
    for r in sorted((r for r in runs if id(r) in ids), key=lambda r: (r[x_key], r[y_key])):
        point = (r[x_key], r[y_key])
        if point not in seen:
            seen.add(point)
            front.append(r)
    return front


# ======================================================================
# RUNS
# ======================================================================

_cache = {}


def method_runs(nested_data, dataset):
    key = (dataset, BEST_N, tuple(sorted(BINS_FILTER.items())))
    if key not in _cache:
        out = {}
        for method in METHODS or g.METHOD_ORDER:
            runs = pu.get_runs(nested_data, method, dataset, k=K,
                               **(BINS_FILTER if method == 'bins' else {}))
            if BEST_N:
                runs = sc.select_configs(runs, n=BEST_N, name=f'{method}/{dataset}')
            out[method] = runs
        _cache[key] = out
    return _cache[key]


# ======================================================================
# AXES
# ======================================================================

UNITS = {
    'bytes':   [('B', 1), ('KB', 1024), ('MB', 1024 ** 2), ('GB', 1024 ** 3), ('TB', 1024 ** 4)],
    'seconds': [('ns', 1e-9), ('µs', 1e-6), ('ms', 1e-3), ('s', 1), ('m', 60), ('h', 3600)],
}


def sig(value):
    return f'{float(f"%.{TICK_SIG}g" % value):g}'


def unit_label(value, kind, base):
    if not value:
        return '0'
    table = dict(UNITS[kind])
    raw = abs(value) * table[base]
    name, size = UNITS[kind][0]
    for cand, cand_size in UNITS[kind]:
        if raw >= cand_size * 0.9999:
            name, size = cand, cand_size
    return f'{sig(value * table[base] / size)}{name}'


def formatter(spec):
    fmt = spec['fmt']
    if fmt == 'time':
        return ticker.FuncFormatter(lambda v, _p=None: unit_label(v, 'seconds', 's'))
    if fmt == 'bytes':
        base = spec['base'] or 'KB'
        return ticker.FuncFormatter(lambda v, _p=None: unit_label(v, 'bytes', base))
    if fmt == 'plain':
        return ticker.FuncFormatter(lambda v, _p=None: '0' if not v else sig(v))
    if callable(fmt):
        return ticker.FuncFormatter(fmt)
    if isinstance(fmt, str):
        return ticker.FormatStrFormatter(fmt)
    return None


def axis_spec(key, dataset):
    spec = {**AXIS_FALLBACK, **AXES.get(key, {}), **AXES.get((dataset, key), {})}
    spec['key'] = key
    if spec['label'] is None:
        text = g.label(key, K)
        spec['label'] = re.sub(r'\s*\([^)]*\)$', '', text) if spec['fmt'] in ('time', 'bytes') else text
    return spec


def round_sig(value, up):
    """Round to TICK_SIG significant figures, upward when `up`, downward otherwise."""
    if not value:
        return 0.0
    step = 10.0 ** (math.floor(math.log10(abs(value))) - (TICK_SIG - 1))
    return (math.ceil if up else math.floor)(value / step) * step


def fit_limits(spec, values):
    """(lo, hi, log) for one axis from its spec and every value drawn on it."""
    values = [v for v in values if pu.is_number(v)]
    log = bool(spec['log']) and (not values or min(values) > 0)
    if spec['log'] and not log:
        pu.warn_once("dataset_costs: %s has a value <= 0, drawn on a plain axis", spec['key'])
    lim = spec['lim'] or (None, None)
    lo, hi = lim
    if log and lo is not None and lo <= 0:
        lo = None   # a 0 lower limit can't go on a log axis: fit it instead
    if values:
        vmin, vmax = min(values), max(values)
        if log:
            span = math.log10(vmax / vmin) if vmax > vmin else 1.0
            lo = vmin / 10 ** (span * PAD) if lo is None else lo
            hi = vmax * 10 ** (span * PAD) if hi is None else hi
        else:
            span = (vmax - vmin) or abs(vmax) or 1.0
            if lo is None:
                lo = vmin - span * PAD
                lo = round_sig(lo, up=False) if NICE_ENDS else lo
            if hi is None:
                hi = vmax + span * PAD
                hi = round_sig(hi, up=True) if NICE_ENDS else hi
        if log and NICE_ENDS:
            lo = round_sig(lo, up=False) if lim[0] is None or lim[0] <= 0 else lo
            hi = round_sig(hi, up=True) if lim[1] is None else hi
    if lo is None or hi is None or hi <= lo:
        return None
    return lo, hi, log


def apply_axis(ax, which, spec, limits):
    """Set scale, limits and ticks; returns True when the axis ended up on a log scale."""
    axis = ax.xaxis if which == 'x' else ax.yaxis
    if limits is None:
        return False
    lo, hi, log = limits
    (ax.set_xscale if which == 'x' else ax.set_yscale)('log' if log else 'linear')
    (ax.set_xlim if which == 'x' else ax.set_ylim)(lo, hi)
    axis.set_minor_locator(ticker.NullLocator())
    n = spec['ticks']
    if n and n >= 2:
        if log:
            r = (hi / lo) ** (1 / (n - 1))
            ticks = [lo * r ** i for i in range(n)]
        else:
            ticks = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        axis.set_major_locator(ticker.FixedLocator(ticks))
    fmt = formatter(spec)
    if fmt:
        axis.set_major_formatter(fmt)
    return log


# ======================================================================
# DRAWING
# ======================================================================

def method_z(method, layer):
    """`layer` plus a fraction under 1 that is bigger the earlier `method` is in STACK_ORDER."""
    order = list(STACK_ORDER or g.METHOD_ORDER)
    rank = order.index(method) if method in order else len(order)
    return layer + 0.9 * (len(order) - rank) / (len(order) + 1)


def style(method, size=MARKER_SIZE, outlined=False):
    color = g.METHOD_COLORS[method]
    return dict(linestyle='none',
                marker=g.METHOD_MARKERS[method] if PER_METHOD_MARKERS else MARKER,
                markersize=size, markerfacecolor=color,
                markeredgecolor='#000000' if outlined and FRONTIER_EDGE else color,
                markeredgewidth=FRONTIER_EDGE if outlined else 0,
                zorder=method_z(method, Z_FRONTIER_POINTS if outlined else Z_POINTS))


def draw_panel(ax, per_method, dataset, x_key, y_key):
    """Draw one panel; returns (x values, y values, frontier runs) for limits and printing."""
    def usable(runs):
        return [r for r in runs if pu.is_number(r.get(x_key)) and pu.is_number(r.get(y_key))]

    good = {m: usable(runs) for m, runs in per_method.items()}
    pooled = [r for runs in good.values() for r in runs]
    on_global = frontier_ids(pooled, x_key, y_key)
    front = frontier(pooled, x_key, y_key)

    def plot(runs, **kw):
        ax.plot([r[x_key] for r in runs], [r[y_key] for r in runs], **kw)

    if FRONTIER_LINE and len(front) > 1:
        plot(front, color=FRONTIER_COLOR, linewidth=FRONTIER_WIDTH, linestyle=FRONTIER_STYLE,
             drawstyle=('steps-post' if better_sign(x_key) < 0 else 'steps-pre') if FRONTIER_STEPS else 'default',
             zorder=Z_FRONTIER_LINE)

    drawn = []
    for method, runs in good.items():
        if not runs:
            pu.warn_once("dataset_costs: %s/%s has no run with both %s and %s", method, dataset, x_key, y_key)
            continue
        solid = runs if POINTS == 'all' else [r for r in runs if id(r) in on_global]
        faded = [r for r in runs if id(r) not in on_global] if POINTS == 'fade' else []
        if faded:
            plot(faded, alpha=FADE_ALPHA, **style(method, MARKER_SIZE * FADE_SIZE))
        if METHOD_LINES:
            own = frontier(runs, x_key, y_key)
            if len(own) > 1:
                plot(own, color=g.METHOD_COLORS[method], linewidth=g.LINE_WIDTH, zorder=method_z(method, Z_POINTS))
        plain = [r for r in solid if id(r) not in on_global]
        edged = [r for r in solid if id(r) in on_global]
        if plain:
            plot(plain, **style(method))
        if edged:
            plot(edged, **style(method, outlined=True))
        drawn += solid + faded
        if LOG_POINTS:
            for r in solid + faded:
                pu.log.info("%-8s %-10s %s=%-10.4g %s=%-10.4g %s", method, dataset,
                            x_key, r[x_key], y_key, r[y_key], r['folder'])

    return [r[x_key] for r in drawn], [r[y_key] for r in drawn], front


def legend_rows():
    if LEGEND_ROWS is not None:
        return LEGEND_ROWS
    rows = [list(r) for r in g.METHOD_LEGEND_ROWS]
    rows[-1].append('frontier')
    return rows


def legend_handle(entry, present):
    if entry == 'frontier':
        if not FRONTIER_LINE:
            return None
        return Line2D([], [], color=FRONTIER_COLOR, linewidth=FRONTIER_WIDTH,
                      linestyle=FRONTIER_STYLE, label=FRONTIER_LABEL)
    if entry not in present:
        return None
    color = g.METHOD_COLORS[entry]
    return Line2D([], [], linestyle='none', color=color,
                  marker=g.METHOD_MARKERS[entry] if PER_METHOD_MARKERS else MARKER,
                  markersize=LEGEND_MARKER_SIZE, markerfacecolor=color, markeredgecolor=color,
                  label=g.METHOD_LEGEND_LABELS.get(entry, g.METHOD_LABELS.get(entry, entry)))


def add_legend(fig, present):
    style_kw = {**g.LEGEND_STYLE, **g.METHOD_LEGEND_STYLE, **LEGEND_STYLE}
    gap = g.METHOD_LEGEND_LINE_GAP / 72 / fig.get_figheight()
    fig.draw_without_rendering()
    top = fig.get_tightbbox().y0 / fig.get_figheight()
    for row in legend_rows():
        handles = [h for h in (legend_handle(e, present) for e in row) if h is not None]
        if not handles:
            continue
        leg = fig.legend(handles, [h.get_label() for h in handles], loc='upper center',
                         bbox_to_anchor=(0.5, top - gap), ncol=len(handles), **style_kw)
        fig.draw_without_rendering()
        top = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0


# ======================================================================
# FIGURE
# ======================================================================

def plot_dataset(nested_data, dataset, rows, cols, name):
    """One rows x cols figure for one dataset."""
    n_rows, n_cols = len(rows), len(cols)

    def pair(i, j):   # (x_key, y_key) of panel row i, column j
        return cols[j], rows[i]

    per_method = method_runs(nested_data, dataset)
    fig = plt.figure(figsize=(PANEL_SIZE[0] * n_cols, PANEL_SIZE[1] * n_rows), layout='constrained')
    fig.get_layout_engine().set(wspace=WSPACE, hspace=HSPACE)
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    xs, ys, fronts = {}, {}, []
    for i in range(n_rows):
        for j in range(n_cols):
            x_key, y_key = pair(i, j)
            ax = axes[i][j]
            ax.grid(GRID, which='major')
            ax.tick_params(labelsize=TICK_SIZE)
            xs[i, j], ys[i, j], front = draw_panel(ax, per_method, dataset, x_key, y_key)
            fronts.append((x_key, y_key, front))

    for i in range(n_rows):
        for j in range(n_cols):
            x_key, y_key = pair(i, j)
            ax = axes[i][j]
            xspec, yspec = axis_spec(x_key, dataset), axis_spec(y_key, dataset)
            x_vals = [v for r in range(n_rows) for v in xs[r, j]] if SHARE_X_COL else xs[i, j]
            y_vals = [v for c in range(n_cols) for v in ys[i, c]] if SHARE_Y_ROW else ys[i, j]
            x_log = apply_axis(ax, 'x', xspec, fit_limits(xspec, x_vals))
            y_log = apply_axis(ax, 'y', yspec, fit_limits(yspec, y_vals))

            if X_LABEL_EVERY_ROW or i == n_rows - 1:
                text = ax.set_xlabel(xspec['label'] + (LOG_SUFFIX if x_log else ''), fontsize=X_LABEL_SIZE)
                if SHOW_ARROWS:
                    g.add_arrow(text, x_key, 'x')
            elif HIDE_INNER_TICKLABELS and SHARE_X_COL:
                ax.tick_params(labelbottom=False)
            if Y_LABEL_EVERY_COL or j == 0:
                text = ax.set_ylabel(yspec['label'] + (LOG_SUFFIX if y_log else ''), fontsize=Y_LABEL_SIZE)
                if SHOW_ARROWS:
                    g.add_arrow(text, y_key, 'y')
            elif HIDE_INNER_TICKLABELS and SHARE_Y_ROW:
                ax.tick_params(labelleft=False)

    if ALIGN_LABELS:   # same x for every y label in a column, same y for every x label in a row,
        for j in range(n_cols):   # whatever the width / height of each panel's tick numbers
            fig.align_ylabels(axes[:, j])
        for i in range(n_rows):
            fig.align_xlabels(axes[i, :])

    if SUPTITLE:
        fig.suptitle(SUPTITLE.format(dataset=g.DATASET_LABELS.get(dataset, dataset)), fontsize=SUPTITLE_SIZE)
    if LEGEND:
        add_legend(fig, {m for m, runs in per_method.items() if runs})

    fig_name = name.format(dataset=dataset)
    path = pu.save_figure(fig, fig_name)
    if PRINT_FRONTIER:
        print_frontier(dataset, fronts, fig_name)
    return path


def fmt_value(key, v):
    if key in ('mrr', 'recall', 'mrr_pre_rerank', 'recall_pre_rerank'):
        return f'{v:.4f}'
    if key == 'client_storage_mb' or key == 'db_size_mb':
        return f'{v:,.1f} MB'
    if key.startswith('comm'):
        return f'{v:,.1f} KB'
    if key.endswith('_time'):
        return f'{v:.3f} s'
    return f'{v:.4g}'


def print_frontier(dataset, fronts, fig_name):
    """Clean tables first, then a dashed line, then the run folders behind each point."""
    clean, details = [], []
    ds = g.DATASET_LABELS.get(dataset, dataset)
    for x_key, y_key, front in fronts:
        xl, yl = axis_spec(x_key, dataset)['label'], g.label(y_key, K)
        clean += [f'### {ds}: {yl} vs {xl} (Pareto frontier)', '',
                  f'| Method | {yl} | {xl} |', '|---|---|---|']
        clean += [f"| {g.METHOD_LABELS.get(r['method'], r['method'])} | {fmt_value(y_key, r[y_key])} | "
                  f"{fmt_value(x_key, r[x_key])} |" for r in front]
        clean.append('')
        details += [f'{ds} {y_key} vs {x_key}:'] + [f"  {r['method']:<10} {r['folder']}" for r in front]
    text = '\n'.join(clean + ['-' * 72] + details)
    print('\n' + text)
    if not WRITE_FILES:
        return
    os.makedirs(g.FIGURE_DIR, exist_ok=True)
    path = os.path.join(g.FIGURE_DIR, f'{fig_name}_frontier.md')
    with open(path, 'w') as fh:
        fh.write(text + '\n')
    pu.log.info('saved %s', path)


def make_plots(nested_data, datasets=None, rows=None, cols=None, name=None):
    """One figure per dataset. Every argument defaults to the CONFIG value of the same name."""
    _cache.clear()
    return [plot_dataset(nested_data, d, list(rows or ROWS), list(cols or COLS), name or FIG_NAME)
            for d in (datasets or DATASETS)]


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results(g.RESULTS_DIR))
