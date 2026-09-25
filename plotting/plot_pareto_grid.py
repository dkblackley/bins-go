"""
One wide figure of up to 8 Pareto panels per dataset, each panel with its own axes:

    figures/pareto_grid_msmarco.pdf
    figures/pareto_grid_scifact.pdf     (FIG_NAME, one per entry of DATASETS)

plot_pareto_vs_latency.py draws one 2x2 grid per dataset, every panel sharing
the same x key (a latency) and taking its y key from Y_KEYS. This file drops
both assumptions: PANELS is a list of panels, one dict each, and a panel says
what its x key, its y key, its dataset, how many configs per method and every
detail of both axes are. So the grid can be MRR and Recall (rows) against WAN
time, LAN time, computation and preprocessing (columns), which is what
PANELS is set to below, but equally two datasets side by side on one metric, or
eight unrelated pairs.

The same panels are drawn once per dataset in DATASETS, so one edit gives both
the MS MARCO figure and the SciFact one.

The axis machinery is plot_pareto_single.py's (limits, steps, formats, log,
unit rescaling), just reachable per panel and with defaults shared by key in
AXIS_DEFAULTS, so 'comm_kb' is set up as bytes once and every panel using it
follows. Each panel's own entry wins over AXIS_DEFAULTS, which wins over the
fallbacks at the bottom of THE AXES. Two things there are new: EVEN_TICKS gives
every axis the same N_TICKS ticks with the first and the last ON the ends of
the axis, and the 'time' and 'bytes' formats put a unit in each tick ('1.3ms',
'1.5m', '0.4MB') instead of one unit in the axis label, which a computation
axis running from microseconds to seconds has no room for.

Everything else matches the other Pareto files: each method brings its BEST_N
configs (BEST_N = None, as here, is every config in the sweep), every one gets
a marker in its method's colour, the runs off the frontier are drawn faint
(POINTS = 'fade'), and the only line is the GLOBAL_FRONTIER over all methods
pooled. 'Better' follows g.METRICS, so an axis where lower is better needs no
other change. Every point drawn is logged with its folder, so

    python load_results.py <folder>

shows the full run behind it.

Metrics not every config has (answer_relevancy, faithfulness) are the reason
this file exists: leave them out of PANELS and BEST_N can be None without the
missing-value warnings that the 2x2 grid produces.

More panels than eight don't fit, so a second set of comparisons means running
this twice: change PANELS (or COST_KEYS) and FIG_NAME together, FIG_NAME being
what stops the second run overwriting the first two figures.
"""

import math
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

K = g.K_MAIN

DATASETS = g.DATASETS   # one figure per dataset, the same panels in each: PANELS says what is
                        # plotted, this says on what. ['msmarco'] for the one figure
DATASET = None          # the dataset of a panel in a figure that doesn't name one; None (as here)
                        # means the figure's own dataset, i.e. one figure per entry of DATASETS.
                        # A panel with a 'dataset' of its own keeps it in both figures
BEST_N = None           # configs per method, picked by select_configs.py (None = every config
                        # in the sweep), for any panel that doesn't name its own
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs

COST_KEYS = ['wan_time', 'lan_time', 'total_time']
QUALITY_KEYS = ['mrr', 'recall']
# the columns and the rows of the default PANELS below, a panel per pair. The
# other three costs asked for are 'comm_kb' (total data sent) and
# 'client_storage_mb' (client storage), which with 'maintenance_time' make the
# second figure:
# COST_KEYS = ['maintenance_time', 'comm_kb', 'client_storage_mb']

FIG_NAME = 'pareto_grid_storage_{dataset}'   # figures/<this>.pdf, '{dataset}' becoming the
                        # dataset of the figure, so the two figures of one run don't overwrite
                        # each other. Change it whenever COST_KEYS or PANELS changes, so the
                        # second set of comparisons doesn't overwrite the first


def cross(quality_keys=QUALITY_KEYS, cost_keys=COST_KEYS, **panel):
    """
    A panel per (quality, cost) pair, quality down the rows and cost across the
    columns, which is the usual shape of this figure. Extra keywords go into
    every panel, e.g. cross(dataset='scifact'). Write PANELS out by hand
    instead when the panels have nothing in common.
    """
    return [dict(y=y_key, x=x_key, **panel) for y_key in quality_keys for x_key in cost_keys]


PANELS = cross()
# One dict per panel, filling the grid left to right, then top to bottom, at
# most GRID_MAX of them. A panel is
#
#   {'x': <run key>, 'y': <run key>,          # the only two it needs
#    'dataset': 'scifact',                    # default: the figure's own dataset
#    'best_n': 5, 'bins': {'vec': 1},         # default BEST_N / BINS_FILTER
#    'title': 'SciFact',                      # over this panel, None for none
#    'xlim': (0, 2), 'xstep': 0.5, 'xlog': True, 'xfmt': '%.1f',
#    'xunits': 'seconds', 'xbase': 's', 'xlabel': 'Latency',
#    'ylim': ..., 'ystep': ..., and so on for y}
#
# Anything an axis doesn't set falls back to AXIS_DEFAULTS[(dataset, key)],
# then AXIS_DEFAULTS[key], then the fallbacks under THE AXES.

# ==========================================
# THE GRID
# ==========================================

COLS = len(COST_KEYS)   # panels across; the rows follow from len(PANELS). None spreads
                        # every panel over a single row
GRID_MAX = 8            # panels past this are dropped with a warning: the point of the
                        # figure is that it still reads at one page width

PANEL_SIZE = (0.85 * g.FIG_SIZE[0], 1.0 * g.FIG_SIZE[1])   # (width, height) in inches of one
                        # panel (2.1 x 1.4 in), before the gaps below. Eight of them make a
                        # figure about 9 in wide, i.e. a full page in landscape or a two-column
                        # spread. Raise both to make the whole figure bigger without changing
                        # how much of it is whitespace
PANEL_GAP = -0.00        # space between two columns, as a fraction of a panel's width
ROW_GAP = -0.2          # space between two rows, as a fraction of a panel's height
                        # Both are gaps ON TOP of the room constrained layout already leaves for
                        # the tick and axis labels, so they only need to be big enough to keep
                        # two panels from reading as one

SUPTITLE = '{dataset} PPRAG Full Comparison'
                        # over the whole figure, '{dataset}' becoming the g.DATASET_LABELS name
                        # of the figure's dataset, or None for no title (when the LaTeX caption
                        # already says it)
SUPTITLE_SIZE = 18
TITLE_SIZE = 14         # a panel's own 'title'

X_LABEL_SIZE = g.FONT_SIZE + 1   # the cost name under each panel: every panel has its own x
Y_LABEL_SIZE = g.FONT_SIZE + 1   # key, so unlike the 2x2 grid there is no label to share
LABEL_PAD = 0                    # gap between an axis name and its tick labels, in points
TICK_SIZE = 9                    # tick labels in this figure only (g.TICK_SIZE is 12 elsewhere)

SHOW_ARROWS = True      # the better-direction badge after each axis label (g.add_arrow)

Y_LABEL_ROW = True      # only the leftmost panel of a row is labelled, when every panel in that
                        # row shares one y key. False labels all of them
Y_LABEL_X = -0.3        # x of a y label, as a fraction of its panel's width (0 = the y axis), so
                        # every row's label sits at the same place whatever its tick labels are.
                        # None restores matplotlib's placement off the widest tick label
Y_LABEL_Y = 0.4         # height of a y label up its panel, 0 = bottom, 1 = top

SHARE_Y_ROW = True      # a row whose panels share a y key gets one set of y limits, fitted to
                        # the row rather than to each panel, so the rows line up

LEGEND = True
LEGEND_ROWS = [['pacmann', 'global'], ['bins', 'tree']]
# one centred legend line per row, under the figure. An entry is a method
# (labelled from g.METHOD_LEGEND_LABELS) or 'global' (the GLOBAL_FRONTIER
# line), and an entry nothing was drawn for is skipped, so 'global' costs
# nothing while GLOBAL_FRONTIER is off. The figure is wide enough for one line;
# [['bins', 'tree'], ['pacmann', 'global']] stacks them as the narrow figures do
LEGEND_MARKER_SIZE = 6
LEGEND_STYLE = dict(handlelength=0.6, fontsize=13)   # on top of g.LEGEND_STYLE /
                                                     # g.METHOD_LEGEND_STYLE

# ==========================================
# THE AXES
# ==========================================

AUTO = 'auto'           # a format left at AUTO is trim_tick on a plain axis and
                        # matplotlib's own on a log or rescaled-unit axis

AXIS_DEFAULTS = {
    # run key (or (dataset, run key)) -> any of the axis options a panel takes,
    # without the x/y prefix: lim, step, fmt, log, units, base, label. Set a
    # cost up once here and every panel that plots it looks the same, and pin
    # it for one dataset with a (dataset, key) entry, which wins over the plain
    # key one: ('msmarco', 'client_storage_mb') is MS MARCO's client storage
    # axis wherever it is plotted, on x or on y, in any panel of that figure.
    # A single panel overrides even that with its own 'xlim' / 'ylim' / 'xstep'.
    #
    # 'lim' and 'step' are in the unit the RUN KEY IS STORED IN, not the unit
    # the ticks happen to show: MB for client_storage_mb and db_size_mb, KB for
    # comm_kb, seconds for every time. So 0 to 1 GB of client storage is
    # (0, 1024) and a 100 MB step is 100, however the ticks end up labelled.
    # (The exception is an axis using whole-axis 'units' rescaling instead of a
    # per-tick format, where the axis really is in the unit it names.)
    #
    # The costs all carry their unit in each tick ('time' / 'bytes', see
    # FORMATTERS) rather than once in the axis label: a computation axis runs
    # from under a millisecond to seconds and a preprocessing one from seconds
    # to hours, so no single unit reads well down the whole axis. The label
    # then drops its own '(s)' / '(KB)', which would contradict the ticks.
    'wan_time':          dict(fmt='time'),
    'lan_time':          dict(fmt='time'),
    'total_time':        dict(fmt='time', label='Computation'),
    'maintenance_time':  dict(fmt='time', label='Preprocessing'),
    'rerank_time':       dict(fmt='time'),
    'comm_kb':           dict(fmt='bytes', label='Data Sent'),
    'client_storage_mb': dict(fmt='bytes', base='MB', label='Client Storage'),
    'db_size_mb':        dict(fmt='bytes', base='MB'),
    # per-dataset overrides, the same options pinned for one dataset only. MS
    # MARCO's client storage runs 0 to 1 GB (1024 MB, the stored unit), which
    # has to be a plain axis: a log one cannot reach 0, so keep log=False and
    # start at lim=(1, 1024) if the small configs matter more than the zero.
    # With EVEN_TICKS a fixed range is simply divided into N_TICKS ticks, so
    # this one is labelled 0, 341MB, 683MB, 1GB (N_TICKS = 5 would make it a
    # round 0, 256MB, 512MB, 768MB, 1GB), and 'step' is another way to say the
    # same thing: step=256 with no lim puts a tick every 256 MB.
    # SciFact gets the same 0 to 1 GB so the two figures' storage axes match.
    ('msmarco', 'client_storage_mb'): dict(lim=(0, 6500), log=False),
    ('scifact', 'client_storage_mb'): dict(lim=(0, 135), log=False),
    ('msmarco', 'wan_time'): dict(lim=(0, 1), log=False),
    ('scifact', 'wan_time'): dict(lim=(0, 0.25), log=False),
    ('msmarco', 'lan_time'): dict(lim=(0, 0.32), log=False),
    ('scifact', 'lan_time'): dict(lim=(0, 0.035), log=False),
    ('msmarco', 'total_time'): dict(lim=(0, 0.9), log=False),
    ('scifact', 'total_time'): dict(lim=(0, 0.025), log=False),
    # data sent (stored in KB) and preprocessing (stored in seconds), per
    # dataset. lim=None fits the data; set a (lo, hi) to pin the range, e.g.
    # lim=(1, 1024 ** 2) for 1 KB to 1 GB sent, or lim=(1, 3600) for 1 s to
    # 1 h of preprocessing. Both are log axes, so lo has to be above 0
    ('msmarco', 'comm_kb'):          dict(lim=(0, 35000), step=None, log=False),
    ('scifact', 'comm_kb'):          dict(lim=(0, 1500), step=None, log=False),
    ('msmarco', 'maintenance_time'): dict(lim=(0, 3000), step=None, log=False),
    ('scifact', 'maintenance_time'): dict(lim=(0, 20), step=None, log=False),
    # the quality metrics keep the plain axis and the trimmed decimals of the
    # fallbacks; pin a range per dataset here when the auto fit is too generous
    ('msmarco', 'mrr'):    dict(lim=(0.0, 0.35)),      # e.g. dict(lim=(0.12, 0.36), step=0.06)
    ('scifact', 'mrr'):    dict(lim=(0.0, 0.725)),
    ('msmarco', 'recall'): dict(),
    ('scifact', 'recall'): dict(),
}

LIM = None              # (lo, hi) in the unit the run key is stored in (see AXIS_DEFAULTS),
                        # or None to fit the data
STEP = None             # gap between ticks, in that same unit, or None for N_TICKS ticks
                        # fitted to the data
FMT = AUTO              # a FORMATTERS name ('time', 'bytes', 'plain', 'trim'), AUTO, a format
                        # string ('%.1f'), a (value, pos) -> str function, or None for
                        # matplotlib's default
LOG = False             # True for a log axis (values spanning several orders of magnitude)
UNIT_KIND = None        # 'bytes' or 'seconds': rescale the whole axis to the most readable unit
UNIT_BASE = None        # in UNITS and put it in the axis label, the alternative to a per-tick
                        # 'time' / 'bytes' format. This is the unit the run key is stored in
                        # ('KB' for comm_kb, 'MB' for client_storage_mb, 's' for any time)
LABEL = None            # axis label instead of g.label(key)

TICK_DP = 2             # decimal places on an axis left at AUTO
TICK_SIG = 2            # significant figures a fitted tick is rounded to, and the ones the
                        # 'time' / 'bytes' / 'plain' labels show, so every label is exact
N_TICKS = 4             # ticks per axis: EVEN_TICKS gives every axis exactly this many
EVEN_TICKS = True       # every axis gets N_TICKS ticks, evenly spaced (in log space on a log
                        # axis), the first and the last of them AT the ends of the axis, so no
                        # panel has a bare stretch past its outermost tick and the panels of a
                        # row line up. False falls back to matplotlib's own tick placing, i.e.
                        # round numbers wherever they happen to land (LOG_TICK_SUBS / N_TICKS)
LOG_TICK_SUBS = (1.0, 2.0, 5.0)   # EVEN_TICKS off: label these points in each decade
PAD = 0.05              # margin around an auto-fitted axis, as a fraction of its range (of the
                        # log range on a log axis). EVEN_TICKS puts the ends on ticks instead,
                        # so the margin is whatever rounding those out to TICK_SIG adds

UNITS = {               # smallest to largest, each as a multiple of the first
    'bytes':   {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4},
    'seconds': {'ns': 1e-9, 'µs': 1e-6, 'ms': 1e-3, 's': 1, 'm': 60, 'h': 3600, 'd': 86400},
}
UNIT_KEY = {'time': 'seconds', 'bytes': 'bytes'}   # which UNITS table a per-tick format reads
UNIT_BASES = {'time': 's', 'bytes': 'KB'}          # the unit those keys are stored in, unless
                                                   # the axis gives a 'base' of its own

# ==========================================
# THE POINTS AND THE LINES
# ==========================================

POINTS = 'fade'         # which of a method's runs get a marker:
                        #   'all'       every run in the pool, i.e. all BEST_N picks
                        #   'frontier'  only the runs on the frontier (POINTS_AGAINST)
                        #   'fade'      the frontier solid, the rest faint
POINTS_AGAINST = 'global'   # which frontier 'frontier'/'fade' judge a run against:
                        #   'global'  the pooled frontier, i.e. the black line that is actually
                        #             drawn, so a run is solid exactly when it sits on that line
                        #   'method'  that method's own frontier, which also keeps runs the line
                        #             skips because another method dominates them
DOMINATED_ALPHA = 0.45  # 'fade' only
DOMINATED_SIZE = 0.7    # 'fade' only: marker size of those points, as a fraction of MARKER_SIZE

GLOBAL_FRONTIER = True  # the frontier over ALL methods pooled together, i.e. the best anyone
                        # achieves at each cost. Every point on it is already drawn in its own
                        # method's colour, so the line only says which of them survive the
                        # comparison across methods
GLOBAL_COLOR = '#000000'
GLOBAL_LINESTYLE = '-'
GLOBAL_WIDTH = g.LINE_WIDTH + 0.5
GLOBAL_LABEL = 'Pareto Frontier'

METHOD_LINES = False    # also join each method's own frontier, one line per method. False leaves
                        # the global line as the only line, with every config a scatter point
STEP_LINE = False       # True draws a line as a staircase (what is actually attainable between
                        # two configs) instead of joining the points straight
LINE_WIDTH = g.LINE_WIDTH

MARKER = 'o'            # every point's shape: colour alone carries the method
PER_METHOD_MARKERS = False   # True takes each method's shape from g.METHOD_MARKERS instead
MARKER_SIZE = 5         # in points: big enough to read the shape, not just the colour
MARKER_EDGE = 0         # white ring around a marker, in points (0 = none), so two methods
                        # landing on the same spot stay apart
SAME_POINT_TOL = 1e-6   # two frontier points closer than this (relative to the spread of the
                        # data) are the same point: one marker, and no line drawn between them

LOG_POINTS = True       # log every point drawn with its folder, as the other Pareto files do.
                        # With BEST_N = None that is the whole sweep, several hundred lines


# ==========================================
# THE FRONTIER (as in plot_pareto_single.py)
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
    every method pooled). Runs landing on the same point are all kept here, unlike
    in frontier(), where the duplicate would only redraw a marker already there.
    """
    dominates = dominance(x_key, y_key)
    return {id(r) for r in runs if not any(dominates(o, r) for o in pool if o is not r)}


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


# ==========================================
# THE PANELS AND THEIR RUNS
# ==========================================

PANEL_OPTIONS = ({'x', 'y', 'dataset', 'best_n', 'bins', 'title'}
                 | {axis + option for axis in 'xy'
                    for option in ('lim', 'step', 'fmt', 'log', 'units', 'base', 'label')})


def panels():
    """PANELS, checked: every panel has both keys, no stray options, at most GRID_MAX of them."""
    kept = []
    for i, panel in enumerate(PANELS):
        stray = set(panel) - PANEL_OPTIONS
        if stray:
            pu.warn_once("pareto_grid: panel %d has unknown option(s) %s, ignored",
                         i, ', '.join(sorted(stray)))
        if not panel.get('x') or not panel.get('y'):
            pu.warn_once("pareto_grid: panel %d has no 'x' or no 'y' key, dropped", i)
            continue
        kept.append(panel)
    if len(kept) > GRID_MAX:
        pu.warn_once("pareto_grid: %d panels is more than GRID_MAX (%d), the last %d are dropped",
                     len(kept), GRID_MAX, len(kept) - GRID_MAX)
        kept = kept[:GRID_MAX]
    return kept


def grid_shape(n_panels):
    """(rows, cols) for n_panels: COLS across, or everything on one row when COLS is None."""
    cols = min(COLS or n_panels, n_panels) or 1
    return -(-n_panels // cols), cols


_runs_cache = {}


def method_runs(nested_data, dataset, best_n, bins_filter):
    """
    {method: runs} for one dataset, cut down to the best_n picks per method when
    best_n is set. Cached, because panels sharing a dataset share their runs and
    select_configs.py would otherwise pick them again for every panel.
    """
    key = (dataset, best_n, tuple(sorted(bins_filter.items())))
    if key not in _runs_cache:
        per_method = {}
        for method in g.METHOD_ORDER:
            fixed = bins_filter if method == 'bins' else {}
            runs = pu.get_runs(nested_data, method, dataset, k=K, **fixed)
            if best_n:
                runs = sc.select_configs(runs, n=best_n, name=f'{method}/{dataset}')
            per_method[method] = runs
        _runs_cache[key] = per_method
    return _runs_cache[key]


# ==========================================
# AXES (as in plot_pareto_single.py, per panel)
# ==========================================

def trim_tick(value, _pos=None):
    """Tick label rounded to TICK_DP decimals with trailing zeros dropped, so a
    0.05 step prints 0.6, 0.65, 0.7 rather than 0.60, 0.65, 0.70 (or 0.6, 0.7, 0.7)."""
    text = f'{value:.{TICK_DP}f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return '0' if text in ('', '-', '-0') else text


def decimal_tick(value, _pos=None):
    """Tick label as a plain decimal ('0.02'), not scientific notation ('2 x 10^-2')."""
    return f'{value:g}'


def short(value):
    """`value` as a plain number of TICK_SIG significant figures: 0.0013, 1.5, 250."""
    return f'{float(f"%.{TICK_SIG}g" % value):g}'


def unit_tick(value, kind, base):
    """
    Tick label carrying its own unit, `value` being stored in `base`: the
    largest unit of UNITS[kind] the value reaches one of, so on one seconds
    axis 0.0013 reads '1.3ms', 0.4 reads '400ms', 1 reads '1s', 90 reads '1.5m'
    and 7200 reads '2h'. A per-tick unit, unlike the one 'units' puts in the label,
    because these axes are log ones spanning several units at once.
    """
    table = UNITS[kind]
    size = table[base]
    if not value:
        return '0'
    name, scale = next(iter(table.items()))
    for candidate, candidate_size in table.items():
        if abs(value) * size >= candidate_size:
            name, scale = candidate, candidate_size
    return f'{short(value * size / scale)}{name}'


def time_tick(value, _pos=None):
    """A value in seconds with its unit: '1.3ms', '0.4s', '1.5m', '2h'."""
    return unit_tick(value, 'seconds', 's')


def bytes_tick(value, _pos=None):
    """A value in KB with its unit: '12KB', '0.4MB', '3GB'."""
    return unit_tick(value, 'bytes', 'KB')


FORMATTERS = {          # a name an axis' 'fmt' can be, instead of a function of its own
    'time':  time_tick,     # a value in seconds, each tick in its own unit: '1.3ms', '1.5m'
    'bytes': bytes_tick,    # a value in KB, each tick in its own unit: '12KB', '0.4MB'
    'plain': decimal_tick,  # '0.02', never '2 x 10^-2'
    'trim':  trim_tick,     # TICK_DP decimals, trailing zeros dropped: '0.6', '0.65'
}
PER_TICK_UNITS = ('time', 'bytes')   # the formats that put the unit in every tick, so the
                                     # axis label has to give its own unit up


def tick_formatter(spec):
    """The FuncFormatter for one axis, or None when matplotlib's own is wanted. A 'time' or
    'bytes' format on an axis stored in another unit ('MB') is rebased to that unit."""
    fmt = spec['fmt']
    if fmt in PER_TICK_UNITS:
        kind = UNIT_KEY[fmt]
        base = spec['base'] or UNIT_BASES[fmt]
        return lambda value, _pos=None: unit_tick(value, kind, base)
    if isinstance(fmt, str) and fmt in FORMATTERS:
        return FORMATTERS[fmt]
    if callable(fmt):
        return fmt
    return None


def round_sig(value, up=False):
    """`value` rounded to TICK_SIG significant figures, away from zero when `up`, so a
    tick label of TICK_SIG figures is exactly the value the tick sits at."""
    if not value:
        return 0.0
    exponent = math.floor(math.log10(abs(value))) - (TICK_SIG - 1)
    step = 10.0 ** exponent
    rounder = (math.ceil if value > 0 else math.floor) if up else round
    return rounder(value / step) * step


def nice_step(size):
    """The smallest 1, 2 or 5 times a power of ten that is at least `size`, i.e. the step of a
    plain axis, so the ticks are round numbers rather than a seventh of the data's range."""
    if size <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(size))
    for multiple in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):   # a fine ladder, because a step that
                                            # has to grow to cover the data (see even_ticks)
                                            # otherwise jumps to twice the range it needs
        if size <= multiple * power * 1.000001:
            return multiple * power
    return 10 * power


def even_ticks(lo, hi, log, fixed_lim, n=None):
    """
    Exactly N_TICKS ticks from lo to hi, the first and the last at the ends of
    the axis. Returns (ticks, (lo, hi)) with the limits moved out to the outer
    ticks, so every panel is the same shape, with a tick in each corner.

    With `fixed_lim` the limits are the axis' own and the ticks just divide
    them. Otherwise the ticks are fitted to the data: a log axis spaces them
    geometrically and rounds each to TICK_SIG figures (a factor-of-1000 axis
    has no round step), a plain axis takes a round step from nice_step, big
    enough that N_TICKS of them still cover the data. `n` is a tick count of
    the caller's own instead of N_TICKS (plot_bins_ablation_grid.py).
    """
    n = max(n or N_TICKS, 2)
    if fixed_lim or hi <= lo:
        if log and lo > 0:
            ratio = (hi / lo) ** (1 / (n - 1))
            return [lo * ratio ** i for i in range(n)], (lo, hi)
        step = (hi - lo) / (n - 1) if hi > lo else 1.0
        return [lo + step * i for i in range(n)], (lo, hi if hi > lo else lo + step * (n - 1))

    if log and lo > 0:
        lo, hi = round_sig(lo), round_sig(hi, up=True)
        ratio = (hi / lo) ** (1 / (n - 1))
        ticks = [lo] + [round_sig(lo * ratio ** i) for i in range(1, n - 1)] + [hi]
        if len(set(ticks)) < n:   # a range too narrow to round apart: leave them where they are
            ticks = [lo * ratio ** i for i in range(n)]
        return ticks, (ticks[0], ticks[-1])

    step = nice_step((hi - lo) / (n - 1))
    start = math.floor(lo / step) * step
    while start + step * (n - 1) < hi:   # rounding the start down can cost the last tick the
        step = nice_step(step * 1.0001)  # top of the data, so widen the step until it doesn't
        start = math.floor(lo / step) * step
    ticks = [start + step * i for i in range(n)]
    return ticks, (ticks[0], ticks[-1])


def axis_spec(panel, which, dataset):
    """
    One axis of one panel as the dict the drawing code reads. An option is taken
    from the panel ('xlog'), else from AXIS_DEFAULTS[(dataset, key)], else from
    AXIS_DEFAULTS[key], else from the fallback global. A format left at AUTO
    resolves to trimmed decimals on a plain axis (right for MRR / Recall) and to
    matplotlib's own on a log or rescaled-unit axis, where there is no natural
    number of decimals.
    """
    key = panel[which]
    shared = {**AXIS_DEFAULTS.get(key, {}), **AXIS_DEFAULTS.get((dataset, key), {})}
    fallback = dict(lim=LIM, step=STEP, fmt=FMT, log=LOG,
                    units=UNIT_KIND, base=UNIT_BASE, label=LABEL)

    def option(name):
        if which + name in panel:
            return panel[which + name]
        return shared[name] if name in shared else fallback[name]

    spec = {name: option(name) for name in fallback}
    if spec['fmt'] == AUTO:
        spec['fmt'] = None if (spec['log'] or spec['units']) else 'trim'
    spec['key'] = key
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
    """
    (multiplier, axis label) for one axis. A 'time' / 'bytes' format leaves the
    values alone and gives every tick its own unit, so the label drops the unit
    it was stored in ('Computation (s)' -> 'Computation'); 'units' instead
    rescales the whole axis and names that one unit in the label.
    """
    label = spec['label'] or g.label(spec['key'], K)
    no_unit = re.sub(r'\s*\([^)]*\)$', '', label)   # drop the stored unit, '(s)'
    if spec['fmt'] in PER_TICK_UNITS:
        return 1.0, no_unit
    if not spec['units']:
        return 1.0, label
    unit, scale = pick_unit(values, spec['units'], spec['base'])
    return scale, f'{no_unit} ({unit})'


def setup_axis(ax, which, spec, values):
    """Scale, limits, ticks and tick labels for one axis; `values` is everything drawn on it."""
    axis = ax.xaxis if which == 'x' else ax.yaxis
    set_scale = ax.set_xscale if which == 'x' else ax.set_yscale
    set_lim = ax.set_xlim if which == 'x' else ax.set_ylim

    log = spec['log'] and (not values or min(values) > 0)   # a zero or negative value has no
    if log:                                                 # place on a log axis, so an axis
        set_scale('log')                                    # holding one stays plain
        axis.set_minor_locator(ticker.NullLocator())   # set_scale brings the minor ticks back
    elif spec['log']:
        pu.warn_once("pareto_grid: %s has a value at or below zero, drawn on a plain axis",
                     spec['key'])

    fitted = None
    if EVEN_TICKS and not spec['step'] and (spec['lim'] or values):
        lo, hi = spec['lim'] or (min(values), max(values))
        fitted, (lo, hi) = even_ticks(lo, hi, log, bool(spec['lim']))
        set_lim(lo, hi)
        axis.set_major_locator(ticker.FixedLocator(fitted))
    elif spec['lim']:
        set_lim(*spec['lim'])
    elif values:
        lo, hi = min(values), max(values)
        if log:
            pad = (hi / lo) ** PAD if hi > lo else 1.5
            set_lim(lo / pad, hi * pad)
        else:
            pad = (hi - lo) * PAD or (abs(hi) or 1) * PAD
            set_lim(lo - pad, hi + pad)

    if fitted is None:
        if spec['step']:
            axis.set_major_locator(ticker.MultipleLocator(spec['step']))
        elif log:
            axis.set_major_locator(ticker.LogLocator(base=10, subs=LOG_TICK_SUBS))
        else:
            axis.set_major_locator(ticker.MaxNLocator(nbins=N_TICKS))

    formatter = tick_formatter(spec)
    if formatter:
        axis.set_major_formatter(ticker.FuncFormatter(formatter))
    elif isinstance(spec['fmt'], str):
        axis.set_major_formatter(ticker.FormatStrFormatter(spec['fmt']))
    elif fitted is not None or log:
        # fitted ticks are rounded to TICK_SIG figures, and matplotlib's own
        # labels would print them as '2 x 10^-2' or with every decimal they have
        axis.set_major_formatter(ticker.FuncFormatter(
            decimal_tick if log else lambda v, _p=None: short(v)))


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


def draw_panel(ax, panel, nested_data, figure_dataset):
    """
    One panel: a marker per config in its method's colour (POINTS), the frontier
    over all methods pooled as one black line (GLOBAL_FRONTIER) and, if
    METHOD_LINES is on, each method's own frontier as a line of its own.

    Returns what the second pass needs to finish the axes: the two specs, the
    two labels and every value drawn on each axis (already scaled to the unit
    the axis shows), because a row's y limits can only be fitted once all of
    its panels have been drawn.
    """
    dataset = panel.get('dataset') or DATASET or figure_dataset
    per_method = method_runs(nested_data, dataset,
                            panel.get('best_n', BEST_N), panel.get('bins', BINS_FILTER))
    xspec, yspec = axis_spec(panel, 'x', dataset), axis_spec(panel, 'y', dataset)
    x_key, y_key = xspec['key'], yspec['key']

    ax.grid(True, which='major')
    ax.tick_params(labelsize=TICK_SIZE)

    def usable(runs):
        return [r for r in runs if pu.is_number(r.get(x_key)) and pu.is_number(r.get(y_key))]

    pooled = [r for runs in per_method.values() for r in usable(runs)]

    def values(key):
        """Every value of `key` in the panel, so the unit is picked over all of it."""
        return [r[key] for r in pooled]

    xscale, xlabel = scale_for(xspec, values(x_key))
    yscale, ylabel = scale_for(yspec, values(y_key))

    def draw(runs, **kwargs):
        return ax.plot([r[x_key] * xscale for r in runs], [r[y_key] * yscale for r in runs],
                       **kwargs)

    def log_point(kind, method, run):
        if LOG_POINTS:
            pu.log.info("%-9s %-8s %-8s %-10s %s=%-10.4g %s=%-10.4g %s", kind, method, dataset,
                        y_key, x_key, run[x_key], y_key, run[y_key], run['folder'])

    if GLOBAL_FRONTIER:
        front = frontier(pooled, x_key, y_key, name=f'all/{dataset}')
        for run in front:
            log_point('global', 'all', run)
        if len(front) > 1:   # a single point is already drawn in its method's colour
            draw(front, color=GLOBAL_COLOR, linewidth=GLOBAL_WIDTH, linestyle=GLOBAL_LINESTYLE,
                 marker='', drawstyle=step_style(x_key) if STEP_LINE else 'default',
                 zorder=1)   # under the method markers, which are its own points

    drawn = []
    for method in g.METHOD_ORDER:
        name = f'{method}/{dataset}'
        good = usable(per_method.get(method, []))
        if not good:
            pu.warn_once("pareto_grid: %s has no run with both %s and %s", name, x_key, y_key)
            continue
        front = frontier(good, x_key, y_key, name=name)   # the METHOD_LINES line
        on_front = on_frontier(good, pooled if POINTS_AGAINST == 'global' else good, x_key, y_key)
        shown = good if POINTS == 'all' else [r for r in good if id(r) in on_front]
        faded = [r for r in good if id(r) not in on_front] if POINTS == 'fade' else []
        drawn += shown + faded
        for run in shown:
            log_point('frontier' if id(run) in on_front else 'point', method, run)

        if faded:
            draw(faded, linestyle='none', alpha=DOMINATED_ALPHA,
                 **marker_style(method, MARKER_SIZE * DOMINATED_SIZE))
        # the line and the markers are drawn separately: the line follows the
        # method's frontier, the markers cover every config POINTS asks for, and
        # a frontier down to one point would only draw a stub of a line
        if METHOD_LINES and len(front) > 1:
            draw(front, color=g.METHOD_COLORS[method], linewidth=LINE_WIDTH, marker='',
                 drawstyle=step_style(x_key) if STEP_LINE else 'default', zorder=2)
        draw(shown, linestyle='none', label=g.METHOD_LABELS[method], **marker_style(method))

    if panel.get('title'):
        ax.set_title(panel['title'].format(dataset=g.DATASET_LABELS[dataset]),
                     fontsize=TITLE_SIZE)

    return dict(xspec=xspec, yspec=yspec, xlabel=xlabel, ylabel=ylabel,
                xs=[r[x_key] * xscale for r in drawn], ys=[r[y_key] * yscale for r in drawn])


def finish_axes(axes, drawn, rows, cols):
    """
    The limits, ticks and labels of every panel, once all of them are drawn.
    Separate from draw_panel because SHARE_Y_ROW fits a row's y limits to the
    whole row, which needs the values of the panels to its right.

    A row whose panels all plot the same y key is labelled once, at the left
    (Y_LABEL_ROW), since repeating 'MRR@10' over four panels of the same row
    says nothing the first one didn't.
    """
    for i, (ax, info) in enumerate(zip(axes, drawn)):
        row, col = i // cols, i % cols
        peers = [d for j, d in enumerate(drawn) if j // cols == row]
        one_y = len({d['yspec']['key'] for d in peers}) == 1
        ys = [y for d in peers for y in d['ys']] if (SHARE_Y_ROW and one_y) else info['ys']

        setup_axis(ax, 'x', info['xspec'], info['xs'])
        setup_axis(ax, 'y', info['yspec'], ys)

        text = ax.set_xlabel(info['xlabel'], fontsize=X_LABEL_SIZE, labelpad=LABEL_PAD)
        if SHOW_ARROWS:
            g.add_arrow(text, info['xspec']['key'], 'x')
        if col == 0 or not (Y_LABEL_ROW and one_y):
            text = ax.set_ylabel(info['ylabel'], fontsize=Y_LABEL_SIZE,
                                 labelpad=LABEL_PAD, y=Y_LABEL_Y)
            if Y_LABEL_X is not None:   # same x in every panel, whatever its tick labels are
                ax.yaxis.set_label_coords(Y_LABEL_X, Y_LABEL_Y)
            if SHOW_ARROWS:
                g.add_arrow(text, info['yspec']['key'], 'y')


# ==========================================
# THE LEGEND
# ==========================================

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
    pu.warn_once("pareto_grid: legend entry %r was not drawn, left out", entry)
    return None


def combined_legend(fig):
    """
    LEGEND_ROWS centred under the figure, one line per row. Same layout as
    pu.method_legend, which can't be used here because the frontier entry has
    to share a line with the method ones: each line is its own legend, placed
    under the one before, since one legend can't centre a lone entry.
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


# ==========================================
# THE FIGURE
# ==========================================

def plot_pareto_grid(nested_data, dataset, name=FIG_NAME):
    """
    One wide figure for one dataset: a PANELS entry per panel, COLS of them per
    row. `dataset` is what every panel that doesn't name one of its own plots,
    so the same PANELS draw the MS MARCO figure and the SciFact one.
    """
    chosen = panels()
    if not chosen:
        pu.warn_once("pareto_grid: PANELS is empty, nothing to draw")
        return None
    rows, cols = grid_shape(len(chosen))

    fig = plt.figure(figsize=(PANEL_SIZE[0] * (cols + PANEL_GAP * (cols - 1)),
                              PANEL_SIZE[1] * (rows + ROW_GAP * (rows - 1))),
                     layout='constrained')
    fig.get_layout_engine().set(wspace=PANEL_GAP, hspace=ROW_GAP)
    axes = list(fig.subplots(rows, cols, squeeze=False).flat)   # a list, not the flat
                                    # iterator, which the first pass below would use up

    drawn = [draw_panel(ax, panel, nested_data, dataset) for ax, panel in zip(axes, chosen)]
    for ax in axes[len(chosen):]:   # a half-filled last row leaves empty panels
        ax.set_axis_off()
    finish_axes(axes[:len(chosen)], drawn, rows, cols)

    if SUPTITLE:
        fig.suptitle(SUPTITLE.format(dataset=g.DATASET_LABELS[dataset]), fontsize=SUPTITLE_SIZE)
    if LEGEND:
        combined_legend(fig)   # the methods and the global frontier (LEGEND_ROWS)

    return pu.save_figure(fig, name.format(dataset=dataset))


def make_plots(nested_data):
    """The same panels once per dataset in DATASETS, i.e. two figures by default."""
    return [plot_pareto_grid(nested_data, dataset) for dataset in DATASETS]


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())