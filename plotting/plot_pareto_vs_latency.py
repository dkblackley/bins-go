"""
plot_metric_vs_latency.py with Pareto frontiers in place of the lines: one 2x2
figure per dataset, one panel per entry of Y_KEYS (MRR@10, recall, relevancy,
faithfulness) against latency, sharing one latency label under the figure

    figures/pareto_comparison_<dataset>.pdf

The look matches plot_pareto_single.py: each method brings its BEST_N configs,
every one gets a marker in its method's colour, and the only line drawn by
default is the GLOBAL_FRONTIER over all methods pooled: the configs nothing
else beats on BOTH axes at once, i.e. the best anyone achieves at each latency,
made of whichever method wins in each region. The runs off that line are drawn
faint (POINTS = 'fade'). Turn METHOD_LINES on to also join each method's own
frontier.

'Better' follows g.METRICS (highest MRR, lowest latency, ...), so swapping X_KEY
or a Y_KEYS entry for something where lower is better needs no other change.
Every point drawn is logged with its folder, so

    python load_results.py <folder>

shows the full run behind it.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

X_KEY = 'wan_time'   # x axis of every panel, swap for 'total_time' (computation) or 'lan_time'
Y_KEYS = ['mrr', 'recall', 'answer_relevancy', 'faithfulness']   # left to right, then top to bottom
K = g.K_MAIN

TITLE = 'PPRAG Comparison - {dataset}'
# '{dataset}' becomes g.DATASET_LABELS[dataset]. None for no title, e.g. when
# the LaTeX caption already says which dataset it is

BEST_N = 5              # configs per method, picked by select_configs.py exactly as
                        # plot_metric_vs_latency.py picks them (None = every config in the sweep)
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs
LOG_X = True

Y_LIM = (0.0, 1.0)      # fallback for any (dataset, metric) not listed in Y_LIMS
Y_LIMS = {              # per-(dataset, metric) ranges, so each plot fills its axes
    ('msmarco', 'mrr'): (0.12, 0.36),
    ('scifact', 'mrr'): (0.52, 0.68),
    ('msmarco', 'recall'): (0.2, 0.6),
    ('scifact', 'recall'): (0.65, 0.85),
    ('msmarco', 'answer_relevancy'): (0.54, 0.78),
    ('scifact', 'answer_relevancy'): (0.55, 0.7),
    ('msmarco', 'faithfulness'): (0.8, 0.96),
    ('scifact', 'faithfulness'): (0.6, 0.72),
}
Y_TICK_STEP = 0.1       # one gridline and one label every 0.1
Y_TICK_STEPS = {
('msmarco', 'mrr'): 0.06,
('scifact', 'mrr'): 0.04,
('scifact', 'recall'): 0.05,
('msmarco', 'faithfulness'): 0.04,
('msmarco', 'answer_relevancy'): 0.06,
('scifact', 'answer_relevancy'): 0.05,
('scifact', 'faithfulness'): 0.03,
}       # per-(dataset, metric) steps, overriding Y_TICK_STEP
Y_TICK_DP = 2           # decimals on the y tick labels, trailing zeros dropped (trim_tick)

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
DOMINATED_SIZE = 0.75   # 'fade' only: marker size of those points, as a fraction of MARKER_SIZE

GLOBAL_FRONTIER = True  # the frontier over ALL methods pooled together, i.e. the best anyone
                        # achieves at each latency. Every point on it is already drawn in its
                        # own method's colour, so the line only says which of them survive the
                        # comparison across methods
GLOBAL_COLOR = '#000000'
GLOBAL_LINESTYLE = '-'
GLOBAL_WIDTH = g.LINE_WIDTH + 0.5
GLOBAL_LABEL = 'Pareto Frontier'

METHOD_LINES = False    # also join each method's own frontier, one line per method. False leaves
                        # the global line as the only line, with every config a scatter point
STEP = False            # True draws a line as a staircase (what is actually attainable between
                        # two configs) instead of joining the points straight
LINE_WIDTH = g.LINE_WIDTH

MARKER = 'o'            # every point's shape: colour alone carries the method
PER_METHOD_MARKERS = False   # True takes each method's shape from g.METHOD_MARKERS instead
MARKER_SIZE = 5         # in points: big enough to read the shape, not just the colour
MARKER_EDGE = 0         # white ring around a marker, in points (0 = none), so two methods
                        # landing on the same spot stay apart
SAME_POINT_TOL = 1e-6   # two frontier points closer than this (relative to the spread of the
                        # data) are the same point: one marker, and no line drawn between them

# ==========================================
# THE FIGURE
# ==========================================

EVEN_LOG_X_TICKS = {    # x keys on a log axis with exactly this many ticks, evenly spaced
    'total_time': 4,    # from the smallest to the largest value plotted
    'comm_kb': 4,
}
EVEN_LOG_X_PAD = 0.0   # room either side of the outer ticks, as a fraction of the log range

X_AXES = {              # (dataset, x key) -> (lo, hi, step): a plain (not log) x axis with these
    ('msmarco', 'wan_time'): (0.0, 2.0, 0.5),   # limits and a tick every step. Anything not
    ('scifact', 'wan_time'): (0.0, 1.4, 0.35),   # listed falls back to EVEN_LOG_X_TICKS / LOG_X
}

X_LABELS = {                           # overrides g.label()
    'comm_kb': 'Bytes Sent',           # the ticks carry the unit (KB/MB/GB)
    'total_time': 'Computation (ms)',  # ticks are drawn in ms (ms_tick), the data stays in s
}

GRID_FIG_SIZE = (2.1 * g.FIG_SIZE[0], 1.55 * g.FIG_SIZE[1])
# (width, height) in inches for one dataset's figure: two single plots wide,
# plus room for a title, the latency label and the legend

PANEL_GAP = 0.1   # space between the left and right panels, as a fraction of the figure width (matplotlib's default is 0.02)
ROW_GAP = 0.0    # space between the top and bottom panels, as a fraction of the figure height

X_TICK_SUBS = (1.0, 2.0, 5.0)   # label these points in each decade: ..., 0.02, 0.05, 0.1, 0.2, ...

TICK_SIZE = 10   # tick labels in this figure only (g.TICK_SIZE is 12 everywhere else)

Y_LABEL_Y = (0.42, 0.2)   # height of each metric name up its own panel, one per row (top, bottom),
                           # 0 = bottom of the panel, 1 = top (raise a number to move that row's labels up)
Y_LABEL_PAD = 4            # gap between the metric name and the y tick labels, in points
Y_LABEL_X = -0.3          # x of every metric name, as a fraction of its panel's width (0 = the
                           # y axis, negative = left of it). Pinned rather than left to Y_LABEL_PAD
                           # because matplotlib places the label off the widest tick label, so a
                           # panel with wider ticks ('0.96' vs '0.3') would sit further out than
                           # the rest. None restores the automatic placement from Y_LABEL_PAD.

X_LABEL_SIZE = g.FONT_SIZE + 2   # the latency name under each bottom-row panel
X_LABEL_PAD = 2                  # gap between it and the latency tick labels, in points

LEGEND_ROWS = [['bins', 'tree'], ['pacmann', 'global']]
# one centred legend line per row, under the figure, in place of pu.method_legend's
# g.METHOD_LEGEND_ROWS. An entry is a method (labelled from g.METHOD_LEGEND_LABELS) or
# 'global' (the GLOBAL_FRONTIER line), which is skipped when nothing was drawn for it,
# so 'global' costs nothing while GLOBAL_FRONTIER is off
LEGEND_MARKER_SIZE = 6          # markers in the legend, where they need less room
LEGEND_STYLE = dict(handlelength=0.6, fontsize=13)   # on top of g.LEGEND_STYLE /
                                                     # g.METHOD_LEGEND_STYLE


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
# TICKS (as in plot_metric_vs_latency.py)
# ==========================================

def decimal_tick(value, _pos=None):
    """Tick label as a plain decimal ('0.02'), not scientific notation ('2 x 10^-2')."""
    return f'{value:g}'


def trim_tick(value, _pos=None):
    """Tick label rounded to Y_TICK_DP decimals with trailing zeros dropped, so a
    0.05 step prints 0.6, 0.65, 0.7 rather than 0.60, 0.65, 0.70 (or 0.6, 0.7, 0.7)."""
    text = f'{value:.{Y_TICK_DP}f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return '0' if text in ('', '-', '-0') else text


def round_short(value):
    """Whole number from 1 up (2.6 -> 3), one significant figure below (0.39 -> 0.4, 0.042 -> 0.04)."""
    return round(value) if value >= 1 else float(f'{value:.1g}')


BYTE_UNITS = ['KB', 'MB', 'GB']
BYTE_UNIT_UP = 100   # move to the next unit from here, so 400 KB shows as 0.4 MB


def byte_unit(kb):
    """(value, unit index) for a value in KB, in whichever unit keeps it under BYTE_UNIT_UP."""
    for i in range(len(BYTE_UNITS)):
        if abs(kb) < BYTE_UNIT_UP or i == len(BYTE_UNITS) - 1:
            return kb, i
        kb /= 1024


def bytes_tick(kb, _pos=None):
    """Tick label for a value in KB: '0.4MB', '3MB', '12KB'."""
    value, i = byte_unit(kb)
    return f'{round_short(value):g}{BYTE_UNITS[i]}'


def round_bytes(kb):
    """kb moved to the value its bytes_tick label shows, so the label is exact."""
    value, i = byte_unit(kb)
    return round_short(value) * 1024 ** i


def ms_tick(seconds, _pos=None):
    """Tick label for a value in seconds, shown in ms: 0.00042 -> '0.4'."""
    return f'{round_short(seconds * 1000):g}'


def round_ms(seconds):
    """seconds moved to the value its ms_tick label shows, so the label is exact."""
    return round_short(seconds * 1000) / 1000


TICK_STYLES = {   # x key -> (rounder, formatter) for even_log_ticks
    'comm_kb': (round_bytes, bytes_tick),
    'total_time': (round_ms, ms_tick),
}


def even_log_ticks(ax, x_key, xs):
    """
    Log x axis with EVEN_LOG_X_TICKS[x_key] ticks spread evenly (in log space)
    from min(xs) to max(xs), each nudged to the value its label shows.
    """
    xs = [x for x in xs if x > 0]
    if not xs:
        return
    lo, hi, n = min(xs), max(xs), EVEN_LOG_X_TICKS[x_key]
    step = (hi / lo) ** (1 / (n - 1)) if hi > lo else 1
    rounder, formatter = TICK_STYLES.get(x_key, (round_short, decimal_tick))
    ticks = sorted({rounder(lo * step ** i) for i in range(n)})

    pad = (hi / lo) ** EVEN_LOG_X_PAD if hi > lo else 1.5
    ax.set_xscale('log')
    ax.set_xlim(min(lo, ticks[0]) / pad, max(hi, ticks[-1]) * pad)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def x_label(x_key):
    return X_LABELS.get(x_key, g.label(x_key))


# ==========================================
# DRAWING
# ==========================================

def method_runs(nested_data, dataset, best_n=BEST_N):
    """{method: runs} for one dataset, cut down to the best_n picks per method when best_n is set."""
    per_method = {}
    for method in g.METHOD_ORDER:
        fixed = BINS_FILTER if method == 'bins' else {}
        runs = pu.get_runs(nested_data, method, dataset, k=K, **fixed)
        if best_n:
            runs = sc.select_configs(runs, n=best_n, name=f'{method}/{dataset}')
        per_method[method] = runs
    return per_method


def marker_for(method):
    """The shape of one method's points: MARKER for everyone, unless PER_METHOD_MARKERS."""
    return g.METHOD_MARKERS[method] if PER_METHOD_MARKERS else MARKER


def marker_style(method, size=MARKER_SIZE):
    """Marker keywords for a point in one method's colour, with a white ring."""
    color = g.METHOD_COLORS[method]
    return dict(marker=marker_for(method), markersize=size, markerfacecolor=color,
                markeredgecolor='white' if MARKER_EDGE else color,
                markeredgewidth=MARKER_EDGE, zorder=2)


def draw_panel(ax, runs_by_method, dataset, y_key, x_key=X_KEY):
    """
    One panel: a marker per config in its method's colour (POINTS), the
    frontier over all methods pooled as one black line (GLOBAL_FRONTIER) and,
    if METHOD_LINES is on, each method's own frontier as a line of its own.
    Returns every x value drawn, which the caller needs for the shared latency
    ticks. Labels, titles and the legend are set by the caller.
    """
    ax.grid(True, which='major')
    ax.tick_params(labelsize=TICK_SIZE + 2)

    def usable(runs):
        return [r for r in runs if pu.is_number(r.get(x_key)) and pu.is_number(r.get(y_key))]

    pooled = [r for runs in runs_by_method.values() for r in usable(runs)]

    def draw(runs, **kwargs):
        return ax.plot([r[x_key] for r in runs], [r[y_key] for r in runs], **kwargs)

    if GLOBAL_FRONTIER:
        front = frontier(pooled, x_key, y_key, name=f'all/{dataset}')
        for run in front:
            pu.log.info("%-9s %-8s %-8s %-18s %s=%-10.4g %s=%-10.4g %s", 'global', 'all',
                        dataset, y_key, x_key, run[x_key], y_key, run[y_key], run['folder'])
        if len(front) > 1:   # a single point is already drawn in its method's colour
            draw(front, color=GLOBAL_COLOR, linewidth=GLOBAL_WIDTH, linestyle=GLOBAL_LINESTYLE,
                 marker='', drawstyle=step_style(x_key) if STEP else 'default',
                 zorder=1)   # under the method markers, which are its own points

    plotted = []
    for method, runs in runs_by_method.items():
        name = f'{method}/{dataset}'
        good = usable(runs)
        if not good:
            pu.warn_once("pareto_comparison: %s has no run with both %s and %s",
                         name, x_key, y_key)
            continue
        front = frontier(good, x_key, y_key, name=name)   # the METHOD_LINES line
        on_front = on_frontier(good, pooled if POINTS_AGAINST == 'global' else good, x_key, y_key)
        shown = good if POINTS == 'all' else [r for r in good if id(r) in on_front]
        faded = [r for r in good if id(r) not in on_front] if POINTS == 'fade' else []
        plotted += [r[x_key] for r in shown] + [r[x_key] for r in faded]
        for run in shown:
            pu.log.info("%-9s %-8s %-8s %-18s %s=%-10.4g %s=%-10.4g %s",
                        'frontier' if id(run) in on_front else 'point', method, dataset,
                        y_key, x_key, run[x_key], y_key, run[y_key], run['folder'])

        if faded:
            draw(faded, linestyle='none', alpha=DOMINATED_ALPHA,
                 **marker_style(method, MARKER_SIZE * DOMINATED_SIZE))
        # the line and the markers are drawn separately: the line follows the
        # method's frontier, the markers cover every config POINTS asks for, and
        # a frontier down to one point would only draw a stub of a line
        if METHOD_LINES and len(front) > 1:
            draw(front, color=g.METHOD_COLORS[method], linewidth=LINE_WIDTH, marker='',
                 drawstyle=step_style(x_key) if STEP else 'default', zorder=2)
        draw(shown, linestyle='none', label=g.METHOD_LABELS[method], **marker_style(method))

    if (dataset, x_key) in X_AXES:
        lo, hi, step = X_AXES[(dataset, x_key)]
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))   # 0.0, 0.5, 1.0, ...
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    elif x_key in EVEN_LOG_X_TICKS:
        even_log_ticks(ax, x_key, plotted)
    elif LOG_X:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=X_TICK_SUBS))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(decimal_tick))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        # set_xscale resets the tick locators, so the minor ticks new_figure()
        # switched off come back (labelled 2x10^-2, 3x10^-2, ... on top of each
        # other on a narrow range) unless they are switched off again here

    ax.set_ylim(*Y_LIMS.get((dataset, y_key), Y_LIM))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(Y_TICK_STEPS.get((dataset, y_key), Y_TICK_STEP)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(trim_tick))
    return plotted


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
    pu.warn_once("pareto_comparison: legend entry %r was not drawn, left out", entry)
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


def plot_dataset(nested_data, dataset, x_key=X_KEY):
    """
    The 2x2 figure for one dataset, a Y_KEYS metric per panel, all sharing the
    latency axis, labelled once per column under the bottom row. One legend
    below everything.
    """
    fig = plt.figure(figsize=GRID_FIG_SIZE, layout='constrained')
    fig.get_layout_engine().set(wspace=PANEL_GAP, hspace=ROW_GAP)
    axes = fig.subplots(2, 2, sharex=True)   # sharex: only the bottom row labels its latency ticks

    picked = method_runs(nested_data, dataset)
    for i, (ax, y_key) in enumerate(zip(axes.flat, Y_KEYS)):
        draw_panel(ax, picked, dataset, y_key, x_key)
        row = i // axes.shape[1]   # the top row's labels sit higher than the bottom row's
        text = ax.set_ylabel(g.label(y_key, K), y=Y_LABEL_Y[row], labelpad=Y_LABEL_PAD)
        if Y_LABEL_X is not None:   # same x in every panel, whatever its tick labels are
            ax.yaxis.set_label_coords(Y_LABEL_X, Y_LABEL_Y[row])
        g.add_arrow(text, y_key, 'y')
    for ax in axes[-1]:   # one latency label per column, under the bottom row, not one per panel
        g.add_arrow(ax.set_xlabel(x_label(x_key), fontsize=X_LABEL_SIZE,
                                  labelpad=X_LABEL_PAD, x=0.4), x_key, 'x')

    if TITLE:
        fig.suptitle(TITLE.format(dataset=g.DATASET_LABELS[dataset]), fontsize=20)

    combined_legend(fig)   # the methods and the global frontier (LEGEND_ROWS)

    return pu.save_figure(fig, f'pareto_comparison_{dataset}')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        plot_dataset(nested_data, dataset)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())