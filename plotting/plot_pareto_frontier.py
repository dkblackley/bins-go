"""
The same 'best-case' comparison as plot_best_histograms.py, but as Pareto
frontiers instead of bars, all in one column PDF:

    figures/pareto_grid.pdf

One panel per entry in PANELS, stacked top to bottom (N_COLS = 1). By default
the top panel is quality against quality (MRR vs Relevancy) and the bottom one
is cost against cost (PIR rounds vs Latency).

A histogram bar hides everything except the one run behind it; a frontier shows
the whole trade-off: every non-dominated config of a method, so "PILLAR-Bin is
up and to the left" is readable straight off the axes without first arguing
about which single config is the fair one to show.

Colour is the method (g.METHOD_COLORS); marker and line style are the dataset
(g.DATASET_MARKERS, DATASET_LINESTYLES here), so both datasets share a panel
and the same method can be followed across them. That means one frontier per
(method, dataset), six lines per panel.

A point is on the frontier when no other run of the same (method, dataset) is
at least as good on BOTH axes and strictly better on one, 'better' following
g.METRICS (highest MRR, lowest latency, ...), so the panels need no per-axis
direction of their own. Every point drawn is logged with its folder, so

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

# One dict per panel, top to bottom. 'x' and 'y' are the run keys; every other
# field is optional, named by the axis it belongs to ('xlim', 'ystep', ...),
# and falls back to AXIS_DEFAULTS (or UNIT_DEFAULTS / LOG_DEFAULTS when 'units'
# or 'log' is set for that axis):
#   xlim/ylim    (lo, hi) in the units shown on the axis, or None to fit the data
#   xstep/ystep  gap between ticks, or None for about N_TICKS auto-placed ticks
#   xfmt/yfmt    tick label format ('%.1f'), a function (value, pos) -> str, or
#                None for matplotlib's default (plain decimals on a log axis)
#   xlog/ylog    True for a log axis (for values spanning several orders of magnitude)
#   xunits/yunits  'bytes' or 'seconds': rescale to the most readable unit in
#                UNITS (e.g. KB -> MB -> GB) and put it in the axis label
#   xbase/ybase  the unit the run key is stored in, e.g. 'KB' for comm_kb
#   xlabel/ylabel  axis label instead of g.label(key)
PANELS = [
    # top: retrieval / answer quality, both datasets together, so the ranges
    # cover SciFact and MS MARCO at once rather than either one on its own
    dict(x='answer_relevancy', y='mrr',
         xlim=(0.4, 0.8), xstep=0.1,
         ylim=(0.0, 0.8), ystep=0.2),

    # bottom: cost against cost
    dict(x='wan_time', y='pir_rounds', xunits='seconds', xbase='s', xlog=True,
         ylim=(0, 8), ystep=2, yfmt='%.0f'),

    # other ready-made panels, swap any of the above for these:
    # dict(x='answer_relevancy', y='recall', xlim=(0.4, 0.8), xstep=0.1),
    # dict(x='faithfulness', y='mrr', xlim=(0.35, 1.0), xstep=0.1),
    # dict(x='wan_time', y='mrr', xunits='seconds', xbase='s', xlog=True),
    # dict(x='comm_kb', y='mrr', xunits='bytes', xbase='KB', xlog=True),
    # dict(x='maintenance_time', y='pir_rounds', xunits='seconds', xbase='s', xlog=True),
]
N_COLS = 1              # 1 = one panel under the other, as asked for; 2 puts them side by side
K = g.K_MAIN

BEST_N = None           # None uses every config of a method (the frontier is the filter);
                        # set to e.g. 5 to draw the frontier of the select_configs.py picks only
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs

SHOW_DOMINATED = False  # also scatter the configs the frontier drops, faintly
DOMINATED_ALPHA = 0.25
DOMINATED_SIZE = 0.6    # marker size of those points, as a fraction of g.MARKER_SIZE

STEP = False            # True draws the frontier as a staircase (what is actually attainable
                        # between two configs) instead of joining the points with a straight line
LINE_WIDTH = g.LINE_WIDTH
MARKER_SIZE = g.MARKER_SIZE + 1
MARKER_EDGE = 0.0       # white outline around each marker, in points (0 = none); helps where
                        # two methods overlap

DATASET_LINESTYLES = {'msmarco': '-', 'scifact': '--'}
# second channel for the dataset, on top of g.DATASET_MARKERS ('o' / 's'), so the
# two frontiers of one method stay apart in greyscale. Set both to '-' to drop it.

GRID_FIG_SIZE = (2.5 * g.FIG_SIZE[0], 1.55 * g.FIG_SIZE[1] * -(-len(PANELS) // N_COLS))
# as wide as the histogram grid, one panel's worth of height per row

AXIS_DEFAULTS = dict(lim=None, step=None, fmt='%.1f', log=False,
                     units=None, base=None, label=None)
# labels rounded to 1 dp, right for MRR / Recall / RAGAS scores
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

DATASET_LEGEND_PANEL = 0        # which panel carries the dataset legend (None for no legend)
DATASET_LEGEND_LOC = 'lower left'   # where in that panel, e.g. 'upper right', 'best'

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
    better on one. Runs tied on both values are kept once, so two configs that
    landed on the same point draw one marker.
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

    kept, seen = [], set()
    for run in front:
        point = (run[x_key], run[y_key])
        if point in seen:
            continue
        seen.add(point)
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
        if spec['log'] and lo > 0:
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

def draw_panel(ax, groups, panel):
    """One pair of metrics: a frontier per (method, dataset). The legends are set by the caller."""
    xspec, yspec = axis_spec(panel, 'x'), axis_spec(panel, 'y')
    x_key, y_key = xspec['key'], yspec['key']

    fronts, dropped = {}, {}
    for method in g.METHOD_ORDER:
        for dataset in g.DATASETS:
            name = f'{method}/{dataset}'
            runs = groups[method, dataset]
            front = frontier(runs, x_key, y_key, name=name)
            if not front:
                pu.warn_once("pareto_grid: %s has no run with both %s and %s",
                             name, x_key, y_key)
                continue
            fronts[method, dataset] = front
            on_front = {id(r) for r in front}
            dropped[method, dataset] = [r for r in runs if id(r) not in on_front
                                        and pu.is_number(r.get(x_key))
                                        and pu.is_number(r.get(y_key))]
            for run in front:
                pu.log.info("%-12s %-8s %-8s %s=%-10.4g %s=%-10.4g %s", 'frontier', method,
                            dataset, x_key, run[x_key], y_key, run[y_key], run['folder'])

    every = [r for runs in fronts.values() for r in runs]
    if SHOW_DOMINATED:
        every += [r for runs in dropped.values() for r in runs]
    xscale, xlabel = scale_for(xspec, [r[x_key] for r in every])
    yscale, ylabel = scale_for(yspec, [r[y_key] for r in every])

    for (method, dataset), front in fronts.items():
        style = dict(color=g.METHOD_COLORS[method], marker=g.DATASET_MARKERS[dataset],
                     markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE,
                     markeredgecolor='white' if MARKER_EDGE else 'none')
        if SHOW_DOMINATED and dropped[method, dataset]:
            ax.plot([r[x_key] * xscale for r in dropped[method, dataset]],
                    [r[y_key] * yscale for r in dropped[method, dataset]],
                    linestyle='none', alpha=DOMINATED_ALPHA,
                    **{**style, 'markersize': MARKER_SIZE * DOMINATED_SIZE, 'markeredgewidth': 0})
        ax.plot([r[x_key] * xscale for r in front], [r[y_key] * yscale for r in front],
                linestyle=DATASET_LINESTYLES[dataset], linewidth=LINE_WIDTH,
                drawstyle=step_style(x_key) if STEP else 'default', **style)

    setup_axis(ax, 'x', xspec, [r[x_key] * xscale for r in every])
    setup_axis(ax, 'y', yspec, [r[y_key] * yscale for r in every])
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, which='major')

    x_text = ax.set_xlabel(xlabel, fontsize=X_LABEL_SIZE, labelpad=LABEL_PAD)
    y_text = ax.set_ylabel(ylabel, fontsize=Y_LABEL_SIZE, labelpad=LABEL_PAD)
    if SHOW_ARROWS:
        g.add_arrow(x_text, x_key, 'x')
        g.add_arrow(y_text, y_key, 'y')


def dataset_legend(ax):
    """One black entry per dataset: colour is the method, marker and line style the dataset."""
    handles = [Line2D([], [], color='black', marker=g.DATASET_MARKERS[d],
                      linestyle=DATASET_LINESTYLES[d], markersize=MARKER_SIZE,
                      label=g.DATASET_LABELS[d]) for d in g.DATASETS]
    return ax.legend(handles=handles, loc=DATASET_LEGEND_LOC, **g.LEGEND_STYLE)


def method_proxies(ax):
    """
    Colour-only handles for pu.method_legend. The drawn lines carry a marker
    each, which would put one dataset's symbol in the method legend and read as
    if it meant something, so the legend gets its own markerless lines instead.
    """
    for method in g.METHOD_ORDER:
        ax.plot([], [], color=g.METHOD_COLORS[method], linewidth=LINE_WIDTH,
                marker='', label=g.METHOD_LABELS[method])


def plot_pareto_frontier(nested_data):
    n_rows = -(-len(PANELS) // N_COLS)   # ceiling division
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=GRID_FIG_SIZE,
                             squeeze=False, layout='constrained')
    fig.get_layout_engine().set(wspace=COL_SPACE, hspace=ROW_SPACE)

    groups = group_runs(nested_data)
    for ax, panel in zip(axes.flat, PANELS):
        draw_panel(ax, groups, panel)
    for ax in axes.flat[len(PANELS):]:
        ax.set_visible(False)   # spare cell when PANELS doesn't fill the grid

    if PANELS:
        method_proxies(axes.flat[0])
    if DATASET_LEGEND_PANEL is not None and DATASET_LEGEND_PANEL < len(PANELS):
        dataset_legend(axes.flat[DATASET_LEGEND_PANEL])

    pu.method_legend(fig)   # under everything; styled and placed in globals (METHOD_LEGEND_*)

    return pu.save_figure(fig, 'pareto_grid')


def make_plots(nested_data):
    plot_pareto_frontier(nested_data)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())