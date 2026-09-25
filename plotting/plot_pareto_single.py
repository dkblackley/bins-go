"""
plot_pareto_frontier.py cut down to the one panel it is made of:

    figures/pareto_msmarco_faithfulness_vs_answer_relevancy.pdf   (FIG_NAME)

One dataset (DATASET), one pair of metrics (X_KEY / Y_KEY), and everything the
grid hides inside a PANELS entry or behind a per-dataset override spelled out
as a global of its own: X_LIM / Y_LIM, X_STEP / Y_STEP, X_LOG / Y_LOG, and so
on, two globals per axis. Nothing else changes: the same runs, the same picks
(BEST_N, select_configs.py), the same frontier, the same legend.

Each method brings its BEST_N configs and every one of them gets a marker in
its method's colour (g.METHOD_COLORS), so the whole spread is on the axes
rather than one hand-picked run. A point is on a frontier when nothing else in
the same pool is at least as good on BOTH axes and strictly better on one,
'better' following g.METRICS (highest MRR, lowest latency, ...), so the axes
need no direction of their own. The only line drawn by default is the
GLOBAL_FRONTIER over all methods pooled: the outer envelope of the panel, made
of whichever method wins in each region. Turn METHOD_LINES on to also join each
method's own frontier, or set POINTS to 'frontier' to drop the configs that
frontier rejects. Every point drawn is logged with its folder, so

    python load_results.py <folder>

shows the full run behind it.
"""

import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

# ==========================================
# WHAT IS PLOTTED
# ==========================================

DATASET = 'msmarco'             # one of g.DATASETS: the only dataset in the panel
X_KEY = 'wan_time'      # run keys, see load_results.py for the full list
Y_KEY = 'answer_relevancy'
K = g.K_MAIN

BEST_N = 5              # configs per method, picked by select_configs.py exactly as
                        # plot_metric_vs_latency.py picks them (None = every config in the sweep)
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs

FIG_NAME = 'pareto_{dataset}_{y}_vs_{x}'   # figures/<this>.pdf, so one dataset or one pair of
                                           # metrics at a time doesn't overwrite the last figure

# ==========================================
# THE AXES (two globals per axis, x then y)
# ==========================================

AUTO = 'auto'           # a format left at AUTO is trim_tick on a plain axis and
                        # matplotlib's own on a log or rescaled-unit axis

X_LIM = (0.0, 2.0)      # (lo, hi) in the units shown on the axis, or None to fit the data. Both ends
Y_LIM = (0.54, 0.78)     # are whole multiples of the step below, so each one lands on a labelled tick
                        # rather than sitting between two of them

X_STEP = 0.5            # gap between ticks, or None for about N_TICKS auto-placed ticks
Y_STEP = 0.06

X_FMT = AUTO            # tick label format: AUTO, a format string ('%.1f'), a function
Y_FMT = AUTO            # (value, pos) -> str, or None for matplotlib's default

X_LOG = False           # True for a log axis (values spanning several orders of magnitude)
Y_LOG = False

X_UNITS = None          # 'bytes' or 'seconds': rescale to the most readable unit in UNITS
Y_UNITS = None          # (e.g. KB -> MB -> GB) and put it in the axis label
X_BASE = None           # the unit the run key is stored in, e.g. 'KB' for comm_kb, 's' for wan_time
Y_BASE = None

X_LABEL = None          # axis label instead of g.label(key)
Y_LABEL = None

# ready-made pairs, swap any of the above for these:
#   X_KEY='wan_time',  X_UNITS='seconds', X_BASE='s', X_LOG=True,  Y_KEY='mrr'
#   X_KEY='comm_kb',   X_UNITS='bytes',   X_BASE='KB', X_LOG=True, Y_KEY='mrr'
#   X_KEY='wan_time',  X_UNITS='seconds', X_BASE='s', Y_KEY='pir_rounds', Y_STEP=2, Y_FMT='%.0f'

# ==========================================
# THE POINTS AND THE LINES
# ==========================================

POINTS = 'fade'          # which of a method's runs get a marker:
                        #   'all'       every run in the pool, i.e. all BEST_N picks
                        #   'frontier'  only the runs on the frontier (POINTS_AGAINST)
                        #   'fade'      the frontier solid, the rest faint
POINTS_AGAINST = 'global'   # which frontier 'frontier'/'fade' judge a run against:
                        #   'global'  the pooled frontier, i.e. the black line that is actually
                        #             drawn, so a run is solid exactly when it sits on that line
                        #   'method'  that method's own frontier, which also keeps runs the line
                        #             skips because another method dominates them
DOMINATED_ALPHA = 0.45  # 'fade' only
DOMINATED_SIZE = 0.75    # 'fade' only: marker size of those points, as a fraction of MARKER_SIZE

GLOBAL_FRONTIER = True  # the frontier over ALL methods pooled together, i.e. the best anyone
                        # achieves at each trade-off. Every point on it is already drawn in its
                        # own method's colour, so the line only says which of them survive the
                        # comparison across methods
GLOBAL_COLOR = '#000000'
GLOBAL_LINESTYLE = '-'
GLOBAL_WIDTH = g.LINE_WIDTH + 1.5
GLOBAL_LABEL = 'Pareto Frontier'

METHOD_LINES = False    # also join each method's own frontier, one line per method. False leaves
                        # the global line as the only line, with every config a scatter point
STEP = False            # True draws a line as a staircase (what is actually attainable between
                        # two configs) instead of joining the points straight
LINE_WIDTH = g.LINE_WIDTH

MARKER = 'o'            # every point's shape: the panel is one dataset and one metric pair, so
                        # colour alone carries the method. Set it per method with g.METHOD_MARKERS
                        # if you would rather the shapes told them apart too (see marker_for)
PER_METHOD_MARKERS = False   # True takes each method's shape from g.METHOD_MARKERS instead
MARKER_SIZE = 8         # in points: big enough to read the shape, not just the colour
MARKER_EDGE = 0         # white ring around a marker, in points (0 = none), so two methods
                        # landing on the same spot stay apart
SAME_POINT_TOL = 1e-6   # two frontier points closer than this (relative to the spread of the
                        # data) are the same point: one marker, and no line drawn between them

# ==========================================
# THE FIGURE
# ==========================================

FIG_SIZE = (1.6 * g.FIG_SIZE[0], 1.3 * g.FIG_SIZE[1])   # (width, height) in inches, the size one
                                                      # panel of the grid ends up at

TITLE = '{dataset}'     # '{dataset}' becomes g.DATASET_LABELS[DATASET]. None for no title, e.g.
                        # when the LaTeX caption already says which dataset it is
TITLE_SIZE = 20

X_LABEL_SIZE = 18       # metric name under the panel
Y_LABEL_SIZE = 18       # metric name to the left of it
LABEL_PAD = 2           # gap between an axis name and its tick labels, in points
TICK_SIZE = 16          # tick labels (the numbers)

SHOW_ARROWS = True      # add the better-direction arrow to each axis label, as in the grids

LEGEND = True
LEGEND_ROWS = [['bins', 'tree'], ['pacmann', 'global']]
# one centred legend line per row, under the figure, in place of pu.method_legend's
# g.METHOD_LEGEND_ROWS. An entry is a method (labelled from g.METHOD_LEGEND_LABELS) or
# 'global' (the GLOBAL_FRONTIER line). An entry nothing was drawn for is skipped and an
# empty line disappears, so 'global' costs nothing when GLOBAL_FRONTIER is off
LEGEND_MARKER_SIZE = 6          # markers in the legend, where they need less room
LEGEND_STYLE = dict(handlelength=0.6)   # on top of g.LEGEND_STYLE / g.METHOD_LEGEND_STYLE:
                                        # room for a marker plus a dash of line

TICK_DP = 2             # decimal places on an axis left at AUTO
N_TICKS = 4             # roughly how many ticks an auto-placed (step None) axis gets
LOG_TICK_SUBS = (1.0, 2.0, 5.0)   # label these points in each decade: ..., 2, 5, 10, 20, ...
PAD = 0.05              # margin around an auto-fitted axis, as a fraction of its range
                        # (of the log range on a log axis)

UNITS = {               # smallest to largest, each as a multiple of the first
    'bytes':   {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4},
    'seconds': {'µs': 1e-6, 'ms': 1e-3, 's': 1, 'min': 60},
}


# ==========================================
# THE FRONTIER
# ==========================================

def better_sign(key):
    """+1 when higher is better for this metric, -1 when lower is."""
    return -1 if g.METRICS.get(key, {}).get('better') == 'lower' else 1


def dominance(x_key, y_key):
    """The 'a is better than b' test for one pair of axes, each in its own better direction."""
    sx, sy = better_sign(x_key), better_sign(y_key)

    def dominates(a, b):
        return (sx * a[x_key] >= sx * b[x_key] and sy * a[y_key] >= sy * b[y_key]
                and (sx * a[x_key] > sx * b[x_key] or sy * a[y_key] > sy * b[y_key]))

    return dominates


def on_frontier(runs, pool, x_key, y_key):
    """
    The ids of the runs nothing in `pool` beats, i.e. the ones sitting on `pool`'s
    frontier. `pool` is what the run is judged against (its own method's runs, or
    every method pooled), which is the part frontier() can't answer: that returns
    one method's line, while a point is only on the drawn envelope if no other
    method dominates it. Runs landing on the same point are all kept here, unlike
    in frontier(), where the duplicate would only redraw a marker already there.
    """
    dominates = dominance(x_key, y_key)
    return {id(r) for r in runs if not any(dominates(o, r) for o in pool if o is not r)}


def frontier(runs, x_key=None, y_key=None, name=''):
    """
    The non-dominated runs, sorted by x. A run is dominated when another is at
    least as good on both axes (each in its own better direction) and strictly
    better on one. Runs landing on the same point (within SAME_POINT_TOL of the
    spread of the data) are kept once, so they draw one marker and no line is
    drawn from a point to itself.
    """
    x_key, y_key = x_key or X_KEY, y_key or Y_KEY
    good = [r for r in runs if pu.is_number(r.get(x_key)) and pu.is_number(r.get(y_key))]
    if runs and len(good) < len(runs):
        pu.warn_once("%s: %d of %d runs are missing %s or %s (NaN), dropped",
                     name, len(runs) - len(good), len(runs), x_key, y_key)
    dominates = dominance(x_key, y_key)

    front = [r for r in good if not any(dominates(o, r) for o in good if o is not r)]
    front.sort(key=lambda r: r[x_key])

    def spread(key):
        values = [r[key] for r in good]
        return (max(values) - min(values)) or abs(max(values)) or 1.0

    tol_x, tol_y = spread(x_key) * SAME_POINT_TOL, spread(y_key) * SAME_POINT_TOL
    kept = []   # the frontier is sorted, so a repeat of a point sits next to it
    for run in front:
        if kept and (abs(run[x_key] - kept[-1][x_key]) <= tol_x
                     and abs(run[y_key] - kept[-1][y_key]) <= tol_y):
            pu.log.debug("%s: %s is the same point as %s, drawn once",
                         name, run['folder'], kept[-1]['folder'])
            continue
        kept.append(run)
    pu.log.debug("%s: %d of %d runs on the %s/%s frontier", name, len(kept), len(good), x_key, y_key)
    return kept


def step_style(x_key):
    """
    Which staircase joins two frontier points. When lower x is better a point
    covers everything to its right ('steps-post'), when higher x is better it
    covers everything to its left ('steps-pre').
    """
    return 'steps-post' if better_sign(x_key) < 0 else 'steps-pre'


def method_runs(nested_data, dataset=DATASET):
    """{method: runs} for the one dataset, cut down to the BEST_N picks per method."""
    per_method = {}
    for method in g.METHOD_ORDER:
        fixed = BINS_FILTER if method == 'bins' else {}
        runs = pu.get_runs(nested_data, method, dataset, k=K, **fixed)
        if BEST_N:
            runs = sc.select_configs(runs, n=BEST_N, name=f'{method}/{dataset}')
        per_method[method] = runs
    return per_method


# ==========================================
# AXES
# ==========================================

def trim_tick(value, _pos=None):
    """Tick label rounded to TICK_DP decimals with trailing zeros dropped, so a
    0.05 step prints 0.6, 0.65, 0.7 rather than 0.60, 0.65, 0.70 (or 0.6, 0.7, 0.7)."""
    text = f'{value:.{TICK_DP}f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return '0' if text in ('', '-', '-0') else text


def axis_spec(key, lim, step, fmt, log, units, base, label):
    """
    One axis' globals as the dict the drawing code reads, with an AUTO format
    resolved: trimmed decimals on a plain axis (right for MRR / Recall / RAGAS
    scores), matplotlib's own on a log or rescaled-unit axis, where there is no
    natural number of decimals.
    """
    if fmt == AUTO:
        fmt = None if (log or units) else trim_tick
    return dict(key=key, lim=lim, step=step, fmt=fmt, log=log,
                units=units, base=base, label=label)


def x_spec():
    return axis_spec(X_KEY, X_LIM, X_STEP, X_FMT, X_LOG, X_UNITS, X_BASE, X_LABEL)


def y_spec():
    return axis_spec(Y_KEY, Y_LIM, Y_STEP, Y_FMT, Y_LOG, Y_UNITS, Y_BASE, Y_LABEL)


def pick_unit(values, units, base):
    """
    (unit name, multiplier) that shows `values` (stored in `base`) most readably:
    the largest unit the biggest value still reaches at least 1 of, so
    2,500,000 KB comes out as 2.4 GB rather than 2441 MB.
    """
    table = UNITS[units]
    top = max((abs(v) * table[base] for v in values if pu.is_number(v)), default=0)
    name = next(iter(table))
    for candidate, size in table.items():
        if top >= size:
            name = candidate
    return name, table[base] / table[name]


def scale_for(spec, values):
    """(multiplier, axis label) for one axis, rescaling to a readable unit when asked."""
    label = spec['label'] or g.label(spec['key'], K)
    if not spec['units']:
        return 1.0, label
    unit, scale = pick_unit(values, spec['units'], spec['base'])
    label = re.sub(r'\s*\([^)]*\)$', '', label)   # drop the stored unit, '(s)'
    return scale, f'{label} ({unit})'


def decimal_tick(value, _pos=None):
    """Tick label as a plain decimal ('0.02'), not scientific notation ('2 x 10^-2')."""
    return f'{value:g}'


def setup_axis(ax, which, spec, values):
    """Scale, limits, ticks and tick labels for one axis; `values` is everything drawn on it."""
    axis = ax.xaxis if which == 'x' else ax.yaxis
    set_scale = ax.set_xscale if which == 'x' else ax.set_yscale
    set_lim = ax.set_xlim if which == 'x' else ax.set_ylim

    if spec['log']:
        set_scale('log')
        axis.set_minor_locator(ticker.NullLocator())   # set_scale brings the minor ticks back

    if spec['lim']:
        set_lim(*spec['lim'])
    elif values:
        lo, hi = min(values), max(values)
        if spec['log']:
            if lo > 0:   # a zero or negative value has no place on a log axis; leaving
                         # the limits alone lets matplotlib drop it and autoscale
                pad = (hi / lo) ** PAD if hi > lo else 1.5
                set_lim(lo / pad, hi * pad)
        else:
            pad = (hi - lo) * PAD or (abs(hi) or 1) * PAD
            set_lim(lo - pad, hi + pad)

    if spec['step']:
        axis.set_major_locator(ticker.MultipleLocator(spec['step']))
    elif spec['log']:
        axis.set_major_locator(ticker.LogLocator(base=10, subs=LOG_TICK_SUBS))
    else:
        axis.set_major_locator(ticker.MaxNLocator(nbins=N_TICKS))

    if callable(spec['fmt']):
        axis.set_major_formatter(ticker.FuncFormatter(spec['fmt']))
    elif spec['fmt']:
        axis.set_major_formatter(ticker.FormatStrFormatter(spec['fmt']))
    elif spec['log']:
        axis.set_major_formatter(ticker.FuncFormatter(decimal_tick))


# ==========================================
# DRAWING
# ==========================================

def marker_for(method):
    """The shape of one method's points: MARKER for everyone, unless PER_METHOD_MARKERS."""
    return g.METHOD_MARKERS[method] if PER_METHOD_MARKERS else MARKER


def marker_style(method, size=MARKER_SIZE):
    """Marker keywords for a point in one method's colour, with a white ring."""
    color = g.METHOD_COLORS[method]
    return dict(marker=marker_for(method), markersize=size, markerfacecolor=color,
                markeredgecolor='white' if MARKER_EDGE else color,
                markeredgewidth=MARKER_EDGE, zorder=2)


def draw_panel(ax, per_method, dataset=DATASET):
    """
    The one panel: a marker per config in its method's colour (POINTS), the
    frontier over all methods pooled as one black line (GLOBAL_FRONTIER) and,
    if METHOD_LINES is on, each method's own frontier as a line of its own. The
    legend is set by the caller.
    """
    xspec, yspec = x_spec(), y_spec()
    x_key, y_key = xspec['key'], yspec['key']

    def usable(runs):
        return [r for r in runs if pu.is_number(r.get(x_key)) and pu.is_number(r.get(y_key))]

    pooled = [r for runs in per_method.values() for r in usable(runs)]

    shown, faded, fronts = {}, {}, {}
    for method in g.METHOD_ORDER:
        name = f'{method}/{dataset}'
        runs = usable(per_method.get(method, []))
        if not runs:
            pu.warn_once("pareto_single: %s has no run with both %s and %s", name, x_key, y_key)
            continue
        fronts[method] = frontier(runs, x_key, y_key, name=name)   # the METHOD_LINES line
        on_front = on_frontier(runs, pooled if POINTS_AGAINST == 'global' else runs, x_key, y_key)
        shown[method] = runs if POINTS == 'all' else [r for r in runs if id(r) in on_front]
        faded[method] = [r for r in runs if id(r) not in on_front] if POINTS == 'fade' else []
        for run in shown[method]:
            pu.log.info("%-9s %-8s %-8s %s=%-10.4g %s=%-10.4g %s",
                        'frontier' if id(run) in on_front else 'point', method,
                        dataset, x_key, run[x_key], y_key, run[y_key], run['folder'])

    every = [r for runs in shown.values() for r in runs]
    every += [r for runs in faded.values() for r in runs]

    def all_runs(key):
        """Every value of `key` in the figure, so the unit is picked over the whole panel."""
        return [r[key] for runs in per_method.values() for r in runs if pu.is_number(r.get(key))]

    xscale, xlabel = scale_for(xspec, all_runs(x_key))
    yscale, ylabel = scale_for(yspec, all_runs(y_key))

    def draw(runs, **kwargs):
        return ax.plot([r[x_key] * xscale for r in runs], [r[y_key] * yscale for r in runs],
                       **kwargs)

    if GLOBAL_FRONTIER:
        front = frontier(pooled, x_key, y_key, name=f'all/{dataset}')
        for run in front:
            pu.log.info("%-12s %-8s %-8s %s=%-10.4g %s=%-10.4g %s", 'global', 'all',
                        dataset, x_key, run[x_key], y_key, run[y_key], run['folder'])
        if len(front) > 1:   # a single point is already drawn in its method's colour
            draw(front, color=GLOBAL_COLOR, linewidth=GLOBAL_WIDTH, linestyle=GLOBAL_LINESTYLE,
                 marker='', drawstyle=step_style(x_key) if STEP else 'default',
                 zorder=1)    # under the method markers, which are its own points

    for method, runs in shown.items():
        color = g.METHOD_COLORS[method]
        if faded[method]:
            draw(faded[method], linestyle='none', alpha=DOMINATED_ALPHA,
                 **marker_style(method, MARKER_SIZE * DOMINATED_SIZE))
        # the line and the markers are drawn separately: the line follows the
        # method's frontier, the markers cover every config POINTS asks for, and
        # a frontier down to one point would only draw a stub of a line
        if METHOD_LINES and len(fronts[method]) > 1:
            draw(fronts[method], color=color, linewidth=LINE_WIDTH, linestyle='-', marker='',
                 drawstyle=step_style(x_key) if STEP else 'default', zorder=2)
        draw(runs, linestyle='none', **marker_style(method))

    setup_axis(ax, 'x', xspec, [r[x_key] * xscale for r in every])
    setup_axis(ax, 'y', yspec, [r[y_key] * yscale for r in every])
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, which='major')

    x_text = ax.set_xlabel(xlabel, fontsize=X_LABEL_SIZE, labelpad=LABEL_PAD)
    y_text = ax.set_ylabel(ylabel, fontsize=Y_LABEL_SIZE, labelpad=LABEL_PAD, y=0.34)
    if SHOW_ARROWS:
        g.add_arrow(x_text, x_key, 'x')
        g.add_arrow(y_text, y_key, 'y')


def legend_handle(entry):
    """
    The legend line for one LEGEND_ROWS entry, or None when that entry was not
    drawn. A method is its colour as a line when METHOD_LINES draws lines, and
    as a plain marker otherwise.
    """
    if entry in g.METHOD_ORDER:
        color = g.METHOD_COLORS[entry]
        shape = dict(marker='') if METHOD_LINES else dict(
            marker=marker_for(entry), markersize=LEGEND_MARKER_SIZE,
            markerfacecolor=color, markeredgecolor=color, linestyle='none')
        return Line2D([], [], color=color, linewidth=LINE_WIDTH,
                      label=g.METHOD_LEGEND_LABELS[entry], **shape)
    if entry == 'global' and GLOBAL_FRONTIER:
        return Line2D([], [], color=GLOBAL_COLOR, linewidth=GLOBAL_WIDTH,
                      linestyle=GLOBAL_LINESTYLE, marker='', label=GLOBAL_LABEL)
    pu.warn_once("pareto_single: legend entry %r was not drawn, left out", entry)
    return None


def combined_legend(fig):
    """
    LEGEND_ROWS centred under the figure, one line per row ('Bin  Tree', then
    'PACMANN  Pareto Frontier'). Same layout as pu.method_legend, which can't
    be used here because the frontier entry has to share a line with a method
    one: each line is its own legend, placed under the one before, since one
    legend can't centre a lone entry.
    """
    style = {**g.LEGEND_STYLE, **g.METHOD_LEGEND_STYLE, **LEGEND_STYLE}
    style['fontsize'] = 13
    gap = g.METHOD_LEGEND_LINE_GAP / 72 / fig.get_figheight()   # points -> figure fraction

    fig.draw_without_rendering()   # lay out the figure so each line knows where to go
    top = fig.get_tightbbox().y0 / fig.get_figheight()   # bottom of the axes, labels, ...
    legends = []
    for row in LEGEND_ROWS:
        handles = [h for h in (legend_handle(entry) for entry in row) if h is not None]
        if not handles:
            continue
        legend = fig.legend(handles, [h.get_label() for h in handles],
                            loc='upper center', bbox_to_anchor=(0.5, top - gap),
                            ncol=len(handles), **style)
        legends.append(legend)
        fig.draw_without_rendering()
        top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
    return legends


def plot_pareto_single(nested_data, dataset=DATASET):
    fig, ax = plt.subplots(figsize=FIG_SIZE, layout='constrained')

    draw_panel(ax, method_runs(nested_data, dataset), dataset)
    if TITLE:
        ax.set_title("MS MARCO Pareto Frontier", fontsize=TITLE_SIZE, y=1.16, x=0.365)
    if LEGEND:
        combined_legend(fig)   # the methods and the global frontier (LEGEND_ROWS)

    return pu.save_figure(fig, FIG_NAME.format(dataset=dataset, x=X_KEY, y=Y_KEY))


def make_plots(nested_data):
    plot_pareto_single(nested_data)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())