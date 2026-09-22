"""
The same 'best-case' comparison as plot_best_histograms.py, but as Pareto
frontiers instead of bars, all in one column PDF:

    figures/pareto_grid.pdf

The grid is one row per entry in PANELS and one column per entry in
PANEL_DATASETS, so the default is four panels: quality against quality (MRR vs
Relevancy) along the top row and cost against cost (LAN vs WAN latency) along
the bottom, MS MARCO on the left and SciFact on the right. A panel with one
dataset takes that dataset's ranges out of the PANELS entry, the way
plot_best_histograms.py keeps one range per metric.

A histogram bar hides everything except the one run behind it. Here each method
brings its BEST_N configs, picked by select_configs.py exactly as
plot_metric_vs_latency.py picks them, and every one of them gets a marker in
its method's colour. So "PILLAR-Bin is up and to the left" is readable straight
off the axes, and it is the whole spread that says it, not one hand-picked run.

Colour (g.METHOD_COLORS) is the only channel that means anything: one dataset
per panel, named by the column heading, so every point is the same MARKER. Only
a panel given several datasets falls back to a shape per dataset
(DATASET_MARKERS, DATASET_FILLED, DATASET_LINESTYLES).

A point is on a frontier when nothing else in the same pool is at least as good
on BOTH axes and strictly better on one, 'better' following g.METRICS (highest
MRR, lowest latency, ...), so the panels need no per-axis direction of their
own. The only line drawn by default is the grey GLOBAL_FRONTIER: that test run
over all methods of a dataset pooled together, i.e. the outer envelope of the
whole figure, made of whichever method wins in each region. Turn METHOD_LINES
on to also join each method's own frontier (six more lines, one per method and
dataset), or set POINTS to 'frontier' to drop the configs each method's own
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

# One dict per ROW of the grid, top to bottom. 'x' and 'y' are the run keys;
# every other field is optional, named by the axis it belongs to ('xlim',
# 'ystep', ...), and falls back to AXIS_DEFAULTS (or UNIT_DEFAULTS /
# LOG_DEFAULTS when 'units' or 'log' is set for that axis):
#   xlim/ylim    (lo, hi) in the units shown on the axis, or None to fit the data
#   xstep/ystep  gap between ticks, or None for about N_TICKS auto-placed ticks
#   xfmt/yfmt    tick label format ('%.1f'), a function (value, pos) -> str, or
#                None for matplotlib's default (plain decimals on a log axis)
#   xlog/ylog    True for a log axis (for values spanning several orders of magnitude)
#   xunits/yunits  'bytes' or 'seconds': rescale to the most readable unit in
#                UNITS (e.g. KB -> MB -> GB) and put it in the axis label
#   xbase/ybase  the unit the run key is stored in, e.g. 'KB' for comm_kb
#   xlabel/ylabel  axis label instead of g.label(key)
# A dataset name ('msmarco', 'scifact') holds the same fields again, used only
# in that dataset's column: ranges that fit one dataset but not the other, as
# in plot_metric_vs_latency.py's Y_LIMS. A panel showing both datasets at once
# ignores them and uses the panel's own fields.
PANELS = [
    # top row: retrieval / answer quality-
    dict(x='answer_relevancy', y='mrr',                    # used if the two share a panel
         msmarco=dict(xlim=(0.54, 0.8),  ylim=(0.08, 0.4), ystep=0.08),       # MRR lands much lower on MS MARCO
         scifact=dict(xlim=(0.56, 0.72), ylim=(0.5, 0.7))),       # than on SciFact, so each gets its own

    # dict(x='wan_time', y='pir_rounds', xunits='seconds', xbase='s',
    #      msmarco=dict(xlim=(0.0, 2.0),  ylim=(0.0, 40), ystep=10),
    #      scifact=dict(xlim=(0.0, 1.5), ylim=(0.0, 20))),

    dict(x='wan_time', y='pir_rounds', xunits='seconds', xbase='s',
         msmarco=dict(xlim=(0.0, 450),  ylim=(0.0, 10)),
         scifact=dict(xlim=(0.0, 300), ylim=(0.0, 10))),

# # top row: retrieval / answer quality-
#     dict(x='wan_time', y='mrr', xunits='seconds', xbase='s',                    # used if the two share a panel
#          msmarco=dict(xlim=(0.0, 2.0),  ylim=(0.08, 0.4), ystep=0.08),       # MRR lands much lower on MS MARCO
#          scifact=dict(xlim=(0.0, 1.5), ylim=(0.5, 0.7))),       # than on SciFact, so each gets its own
#
#     dict(x='wan_time', y='answer_relevancy', xunits='seconds', xbase='s',
#          msmarco=dict(xlim=(0.0, 2.0),  ylim=(0.54, 0.8)),
#          scifact=dict(xlim=(0.0, 1.5), ylim=(0.56, 0.72))),

    # # top row: retrieval / answer quality-
    # dict(x='wan_time', xunits='seconds', xbase='s',  y='mrr',                    # used if the two share a panel
    #      msmarco=dict(ylim=(0.08, 0.4)),        # MRR lands much lower on MS MARCO
    #      scifact=dict(ylim=(0.5, 0.7))),       # than on SciFact, so each gets its own

    # # bottom row: cost against cost, the same latency over two link speeds
    # dict(x='pir_rounds', y='wan_time', xunits='seconds', xbase='s', xlog=True,
    #      yunits='seconds', ybase='s', ylog=True),
    # per-dataset ranges here too, if the two end up on different scales, e.g.
    #    msmarco=dict(xlim=(10, 400)), scifact=dict(xlim=(1, 40)),
    # (a lim on a unit axis is in the unit the axis ended up in, see xunits)

    # other ready-made rows, swap any of the above for these:
    # dict(x='wan_time', y='pir_rounds', xunits='seconds', xbase='s', xlog=True,
    #      ylim=(0, 8), ystep=2, yfmt='%.0f'),   # the PIR rounds panel
    # dict(x='answer_relevancy', y='recall', xlim=(0.4, 0.8), xstep=0.1),
    # dict(x='faithfulness', y='mrr', xlim=(0.35, 1.0), xstep=0.1),
    # dict(x='wan_time', y='mrr', xunits='seconds', xbase='s', xlog=True),
    # dict(x='comm_kb', y='mrr', xunits='bytes', xbase='KB', xlog=True),

]

PANEL_DATASETS = [['msmarco'], ['scifact']]
# one COLUMN per entry: [['msmarco'], ['scifact']] is the four-panel grid, one
# dataset per panel. An entry listing several datasets draws them together in
# one panel, so [g.DATASETS] gives the earlier one-column figure back
K = g.K_MAIN

BEST_N = 1              # configs per method, picked by select_configs.py exactly as
                        # plot_metric_vs_latency.py picks them (None = every config in the sweep)
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs

POINTS = 'all'          # which of a method's runs get a marker:
                        #   'all'       every run in the pool, i.e. all BEST_N picks
                        #   'frontier'  only the runs on that method's own frontier
                        #   'fade'      the frontier solid, the rest faint
DOMINATED_ALPHA = 0.25  # 'fade' only
DOMINATED_SIZE = 0.6    # 'fade' only: marker size of those points, as a fraction of MARKER_SIZE

GLOBAL_FRONTIER = True  # one line per dataset: the frontier over ALL methods pooled together,
                        # i.e. the best anyone achieves at each trade-off. Every point on it is
                        # already drawn in its own method's colour, so the line only says which
                        # of them survive the comparison across methods
GLOBAL_COLOR = '#000000'
GLOBAL_LINESTYLE = "-"
GLOBAL_WIDTH = g.LINE_WIDTH
GLOBAL_LABEL = 'Pareto Frontier'

METHOD_LINES = False    # also join each method's own frontier, one line per (method, dataset).
                        # False leaves the global line as the only line, with every config a
                        # scatter point: the figure then says 'here is everything each method
                        # can do, and here is the envelope' without six curves crossing
STEP = False            # True draws a line as a staircase (what is actually attainable between
                        # two configs) instead of joining the points straight
LINE_WIDTH = g.LINE_WIDTH

MARKER_SIZE = 5        # in points: big enough to read the shape, not just the colour
MARKER_EDGE = 0       # white ring around a filled marker, in points (0 = none), so two
                        # methods landing on the same spot stay apart
HOLLOW_EDGE = 1.8       # line width of a hollow marker, in points
SAME_POINT_TOL = 1e-6   # two frontier points closer than this (relative to the axis range)
                        # are the same point: one marker, and no line drawn between them

MARKER = 'o'            # every point's shape. A panel is one dataset (PANEL_DATASETS) and the
                        # column heading names it, so the shape has nothing left to say and
                        # colour alone carries the method

# Only used by a panel that shows more than one dataset, where the shape has to tell
# them apart again: marker, fill and line style per dataset. A hollow marker can sit on
# top of a filled one and both stay visible, which a pair of filled shapes cannot.
DATASET_MARKERS = {'msmarco': 'D', 'scifact': '^'}
DATASET_FILLED = {'msmarco': True, 'scifact': False}
DATASET_LINESTYLES = {'msmarco': '-', 'scifact': '--'}

PANEL_SIZE = (1.1 * g.FIG_SIZE[0], 1 * g.FIG_SIZE[1])   # (width, height) of ONE panel, in inches
GRID_FIG_SIZE = (PANEL_SIZE[0] * len(PANEL_DATASETS), PANEL_SIZE[1] * len(PANELS))
# two columns of these land at 5.6in, so the figure still fits a page width at its natural size

TICK_DP = 2             # decimal places on an axis with no fmt of its own


def trim_tick(value, _pos=None):
    """Tick label rounded to TICK_DP decimals with trailing zeros dropped, so a
    0.05 step prints 0.6, 0.65, 0.7 rather than 0.60, 0.65, 0.70 (or 0.6, 0.7, 0.7)."""
    text = f'{value:.{TICK_DP}f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return '0' if text in ('', '-', '-0') else text


AXIS_DEFAULTS = dict(lim=None, step=None, fmt=trim_tick, log=False,
                     units=None, base=None, label=None)
# trimmed to 2 dp, right for MRR / Recall / RAGAS scores
UNIT_DEFAULTS = dict(lim=None, step=None, fmt=None)
LOG_DEFAULTS = dict(step=None, fmt=None)
# costs and log axes have no natural range, so they fit the data unless told otherwise
N_TICKS = 4             # roughly how many ticks an auto-placed ('step': None) axis gets
LOG_TICK_SUBS = (1.0, 2.0, 5.0)   # label these points in each decade: ..., 2, 5, 10, 20, ...
PAD = 0.05              # margin around an auto-fitted axis, as a fraction of its range
                        # (of the log range on a log axis)

UNITS = {               # smallest to largest, each as a multiple of the first
    'bytes':   {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4},
    'seconds': {'µs': 1e-6, 'ms': 1e-3, 's': 1, 'min': 60},
}

X_LABEL_SIZE = 15       # metric name under each panel
Y_LABEL_SIZE = 15       # metric name on the left of each panel
LABEL_PAD = 2           # gap between an axis name and its tick labels, in points
TICK_SIZE = 12          # tick labels (the numbers)

SHOW_ARROWS = True      # add the better-direction arrow to each axis label, as in the other grids

TITLES = True           # dataset name over each column, so the panels say which is which
TITLE_SIZE = 15

LEGEND_ROWS = [['bins', 'tree'], ['pacmann', 'global']]
# one centred legend line per row, under the figure, in place of pu.method_legend's
# g.METHOD_LEGEND_ROWS. An entry is a method (labelled from g.METHOD_LEGEND_LABELS),
# 'global' (the GLOBAL_FRONTIER line) or a dataset (a black marker, labelled from
# g.DATASET_LABELS) — that last one only earns its place in a figure where a panel
# shows more than one dataset, since the column headings name them otherwise. An
# entry nothing was drawn for is skipped and an empty line disappears, so 'global'
# costs nothing when GLOBAL_FRONTIER is off
LEGEND_MARKER_SIZE = 8          # markers in the legend, where they need less room
METHOD_LEGEND_MARKER = MARKER   # the method's swatch when METHOD_LINES draws no lines
LEGEND_STYLE = dict(handlelength=1.6)   # on top of g.LEGEND_STYLE / g.METHOD_LEGEND_STYLE:
                                        # room for a marker plus a dash of line

COL_SPACE = 0.06        # extra gap between columns, as a fraction of the figure width
ROW_SPACE = 0.08        # extra gap between rows, as a fraction of the figure height


# ==========================================
# THE FRONTIER
# ==========================================

def better_sign(key):
    """+1 when higher is better for this metric, -1 when lower is."""
    return -1 if g.METRICS.get(key, {}).get('better') == 'lower' else 1


def frontier(runs, x_key, y_key, name=''):
    """
    The non-dominated runs, sorted by x. A run is dominated when another is at
    least as good on both axes (each in its own better direction) and strictly
    better on one. Runs landing on the same point (within SAME_POINT_TOL of the
    spread of the data) are kept once, so they draw one marker and no line is
    drawn from a point to itself.
    """
    good = [r for r in runs if pu.is_number(r.get(x_key)) and pu.is_number(r.get(y_key))]
    if runs and len(good) < len(runs):
        pu.warn_once("%s: %d of %d runs are missing %s or %s (NaN), dropped",
                     name, len(runs) - len(good), len(runs), x_key, y_key)
    sx, sy = better_sign(x_key), better_sign(y_key)

    def dominates(a, b):
        return (sx * a[x_key] >= sx * b[x_key] and sy * a[y_key] >= sy * b[y_key]
                and (sx * a[x_key] > sx * b[x_key] or sy * a[y_key] > sy * b[y_key]))

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


def group_runs(nested_data):
    """{(method, dataset): runs}, computed once for every panel."""
    groups = {}
    for method in g.METHOD_ORDER:
        fixed = BINS_FILTER if method == 'bins' else {}
        for dataset in g.DATASETS:
            runs = pu.get_runs(nested_data, method, dataset, k=K, **fixed)
            if BEST_N:
                runs = sc.select_configs(runs, n=BEST_N, name=f'{method}/{dataset}')
            groups[method, dataset] = runs
    return groups


# ==========================================
# AXES
# ==========================================

def panel_for(panel, datasets):
    """
    A PANELS entry with the per-dataset overrides applied, for the dataset(s)
    one panel shows. Only a panel showing a single dataset takes them: with
    both in one panel there is one pair of axes for the two, so the panel's own
    fields have to cover both.
    """
    plain = {key: value for key, value in panel.items() if key not in g.DATASETS}
    if len(datasets) == 1:
        return {**plain, **panel.get(datasets[0], {})}
    return plain


def axis_spec(panel, prefix):
    """
    One axis of a PANELS entry with every unset field filled in from the
    defaults: axis_spec(panel, 'x') turns {'x': 'wan_time', 'xlog': True} into
    {'key': 'wan_time', 'log': True, 'lim': None, ...}.
    """
    fields = {key[1:]: value for key, value in panel.items()
              if key.startswith(prefix) and key[1:]}
    spec = dict(AXIS_DEFAULTS)
    if fields.get('units'):
        spec.update(UNIT_DEFAULTS)
    if fields.get('log'):
        spec.update(LOG_DEFAULTS)
    spec.update(fields, key=panel[prefix])
    return spec


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

def marker_style(color, dataset=None, size=MARKER_SIZE, distinct=False):
    """
    Marker keywords for a point in one method's colour: MARKER, filled, with a
    white ring. With `distinct` (a panel holding more than one dataset) the
    shape and fill come from the dataset instead, and the hollow one goes on
    top, so a pair sitting on the same point shows both outlines.
    """
    if not distinct:
        return dict(marker=MARKER, markersize=size, markerfacecolor=color,
                    markeredgecolor='white' if MARKER_EDGE else color,
                    markeredgewidth=MARKER_EDGE, zorder=2)
    if DATASET_FILLED[dataset]:
        return dict(marker=DATASET_MARKERS[dataset], markersize=size, markerfacecolor=color,
                    markeredgecolor='white' if MARKER_EDGE else color,
                    markeredgewidth=MARKER_EDGE, zorder=2)
    return dict(marker=DATASET_MARKERS[dataset], markersize=size, markerfacecolor='none',
                markeredgecolor=color, markeredgewidth=HOLLOW_EDGE, zorder=3)


def draw_panel(ax, groups, panel, datasets):
    """
    One pair of metrics for one dataset (or several, when a PANEL_DATASETS
    entry lists more than one): a marker per config in its method's colour
    (POINTS), the frontier over all methods pooled per dataset as one grey line
    (GLOBAL_FRONTIER) and, if METHOD_LINES is on, each method's own frontier as
    a line of its own. The legend is set by the caller.
    """
    spec = panel_for(panel, datasets)
    xspec, yspec = axis_spec(spec, 'x'), axis_spec(spec, 'y')
    x_key, y_key = xspec['key'], yspec['key']

    def usable(runs):
        return [r for r in runs if pu.is_number(r.get(x_key)) and pu.is_number(r.get(y_key))]

    shown, faded, fronts = {}, {}, {}
    for method in g.METHOD_ORDER:
        for dataset in datasets:
            name = f'{method}/{dataset}'
            runs = usable(groups[method, dataset])
            if not runs:
                pu.warn_once("pareto_grid: %s has no run with both %s and %s",
                             name, x_key, y_key)
                continue
            front = frontier(runs, x_key, y_key, name=name)
            on_front = {id(r) for r in front}
            fronts[method, dataset] = front
            shown[method, dataset] = runs if POINTS == 'all' else front
            faded[method, dataset] = ([r for r in runs if id(r) not in on_front]
                                      if POINTS == 'fade' else [])
            for run in shown[method, dataset]:
                pu.log.info("%-9s %-8s %-8s %s=%-10.4g %s=%-10.4g %s",
                            'frontier' if id(run) in on_front else 'point', method,
                            dataset, x_key, run[x_key], y_key, run[y_key], run['folder'])

    every = [r for runs in shown.values() for r in runs]
    every += [r for runs in faded.values() for r in runs]

    def all_runs(key):
        """Every value of `key` in the figure, so one metric keeps one unit in every panel."""
        return [r[key] for runs in groups.values() for r in runs if pu.is_number(r.get(key))]

    xscale, xlabel = scale_for(xspec, all_runs(x_key))
    yscale, ylabel = scale_for(yspec, all_runs(y_key))

    if GLOBAL_FRONTIER:
        for dataset in datasets:
            # pooled per dataset: MRR on SciFact and on MS MARCO are different
            # scales, so a frontier over the two together would mean nothing
            pooled = [r for method in g.METHOD_ORDER for r in groups[method, dataset]]
            front = frontier(pooled, x_key, y_key, name=f'all/{dataset}')
            for run in front:
                pu.log.info("%-12s %-8s %-8s %s=%-10.4g %s=%-10.4g %s", 'global', 'all',
                            dataset, x_key, run[x_key], y_key, run[y_key], run['folder'])
            if len(front) > 1:   # a single point is already drawn in its method's colour
                ax.plot([r[x_key] * xscale for r in front], [r[y_key] * yscale for r in front],
                        color=GLOBAL_COLOR, linewidth=GLOBAL_WIDTH, linestyle=GLOBAL_LINESTYLE,
                        marker='', drawstyle=step_style(x_key) if STEP else 'default',
                        zorder=1)   # under the method markers, which are its own points

    def draw(runs, **kwargs):
        return ax.plot([r[x_key] * xscale for r in runs], [r[y_key] * yscale for r in runs],
                       **kwargs)

    # one dataset in the panel: the heading names it, so every point is MARKER
    # and only the colour means anything. Several: they need telling apart again
    distinct = len(datasets) > 1
    for (method, dataset), runs in shown.items():
        color = g.METHOD_COLORS[method]
        if faded[method, dataset]:
            draw(faded[method, dataset], linestyle='none', alpha=DOMINATED_ALPHA,
                 **marker_style(color, dataset, MARKER_SIZE * DOMINATED_SIZE, distinct))
        # the line and the markers are drawn separately: the line follows the
        # method's frontier, the markers cover every config POINTS asks for, and
        # a frontier down to one point would only draw a stub of a line
        if METHOD_LINES and len(fronts[method, dataset]) > 1:
            draw(fronts[method, dataset], color=color, linewidth=LINE_WIDTH,
                 linestyle=DATASET_LINESTYLES[dataset] if distinct else '-', marker='',
                 drawstyle=step_style(x_key) if STEP else 'default', zorder=2)
        draw(runs, linestyle='none', **marker_style(color, dataset, distinct=distinct))

    setup_axis(ax, 'x', xspec, [r[x_key] * xscale for r in every])
    setup_axis(ax, 'y', yspec, [r[y_key] * yscale for r in every])
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, which='major')

    x_text = ax.set_xlabel(xlabel, fontsize=X_LABEL_SIZE, labelpad=LABEL_PAD)
    y_text = ax.set_ylabel(ylabel, fontsize=Y_LABEL_SIZE, labelpad=LABEL_PAD)
    if SHOW_ARROWS:
        g.add_arrow(x_text, x_key, 'x')
        g.add_arrow(y_text, y_key, 'y')


def legend_handle(entry):
    """
    The legend line for one LEGEND_ROWS entry, or None when that entry was not
    drawn. A method is its colour as a line when METHOD_LINES draws lines, and
    as a plain dot otherwise. A dataset is drawn in black, since its colour is
    whichever method it belongs to, in the shape a shared panel gives it.
    """
    if entry in g.METHOD_ORDER:
        color = g.METHOD_COLORS[entry]
        shape = dict(marker='') if METHOD_LINES else dict(
            marker=METHOD_LEGEND_MARKER, markersize=LEGEND_MARKER_SIZE,
            markerfacecolor=color, markeredgecolor=color, linestyle='none')
        return Line2D([], [], color=color, linewidth=LINE_WIDTH,
                      label=g.METHOD_LEGEND_LABELS[entry], **shape)
    if entry in g.DATASETS:
        return Line2D([], [], color='black', linewidth=LINE_WIDTH,
                      linestyle=DATASET_LINESTYLES[entry] if METHOD_LINES else 'none',
                      label=g.DATASET_LABELS[entry],
                      **marker_style('black', entry, LEGEND_MARKER_SIZE, distinct=True))
    if entry == 'global' and GLOBAL_FRONTIER:
        return Line2D([], [], color=GLOBAL_COLOR, linewidth=GLOBAL_WIDTH,
                      linestyle=GLOBAL_LINESTYLE, marker='', label=GLOBAL_LABEL)
    pu.warn_once("pareto_grid: legend entry %r was not drawn, left out", entry)
    return None


def combined_legend(fig):
    """
    LEGEND_ROWS centred under the figure, one line per row ('Bin  Tree', then
    'PACMANN  MS MARCO  SciFact'). Same layout as pu.method_legend, which can't
    be used here because a dataset entry has to share a line with a method one:
    each line is its own legend, placed under the one before, since one legend
    can't centre a lone entry.
    """
    style = {**g.LEGEND_STYLE, **g.METHOD_LEGEND_STYLE, **LEGEND_STYLE}
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


def plot_pareto_frontier(nested_data):
    fig, axes = plt.subplots(len(PANELS), len(PANEL_DATASETS), figsize=GRID_FIG_SIZE,
                             squeeze=False, layout='constrained')
    fig.get_layout_engine().set(wspace=COL_SPACE, hspace=ROW_SPACE)

    groups = group_runs(nested_data)
    for row, panel in enumerate(PANELS):
        for col, datasets in enumerate(PANEL_DATASETS):
            draw_panel(axes[row][col], groups, panel, list(datasets))
    if TITLES:
        for ax, datasets in zip(axes[0], PANEL_DATASETS):   # one heading per column
            ax.set_title(' / '.join(g.DATASET_LABELS[d] for d in datasets), fontsize=TITLE_SIZE)

    combined_legend(fig)   # methods, datasets and the global frontier (LEGEND_ROWS)

    return pu.save_figure(fig, 'pareto_grid')


def make_plots(nested_data):
    plot_pareto_frontier(nested_data)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())