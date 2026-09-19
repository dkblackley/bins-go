"""
One 2x2 figure per dataset comparing the methods, one line per method in
every panel:

    top row      TOP_Y_KEYS (MRR@10, faithfulness) vs latency
    bottom row   PIR rounds per query vs BOTTOM_X_KEYS (computation,
                 bytes sent), sharing the rounds axis

    figures/comparison_<dataset>.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

X_KEY = 'wan_time'   # top row x axis, swap for 'total_time' (computation) or 'lan_time'
TOP_Y_KEYS = ['mrr', 'faithfulness']   # left to right
ROUNDS_KEY = 'pir_rounds'              # bottom row y axis
BOTTOM_X_KEYS = ['total_time', 'comm_kb']   # left to right
K = g.K_MAIN

TITLE = 'PPRAG Comparison - {dataset}'
# '{dataset}' becomes g.DATASET_LABELS[dataset]. None for no title, e.g. when
# the LaTeX caption already says which dataset it is

ONLY_IMPROVING = False   # False plots every config, True drops configs that are slower and no better
BEST_N = 5              # keep only this many configs per method (None = every config)
ROUNDS_ALL_CONFIGS = False   # True: the bottom row shows every config, not just the BEST_N picks
LOG_X = True
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs

Y_LIM = (0.0, 1.0)      # fallback for any (dataset, metric) not listed in Y_LIMS
Y_LIMS = {              # per-(dataset, metric) ranges, so each plot fills its axes
    ('msmarco', 'mrr'): (0.0, 0.4),
    ('scifact', 'mrr'): (0.4, 0.8),
    ('msmarco', 'recall'): (0.2, 0.6),
    ('scifact', 'recall'): (0.5, 0.9),
}
Y_TICK_STEP = 0.1       # one gridline and one label every 0.1

ROUNDS_LOG_Y = False    # True if the round counts span a wide range (e.g. PACMANN at 32+ steps)
ROUNDS_AXIS = (0, 32, 8)   # (min, max, tick step) for the linear rounds axis

EVEN_LOG_X_TICKS = {    # x keys on a log axis with exactly this many ticks, evenly spaced
    'total_time': 4,    # from the smallest to the largest value plotted
    'comm_kb': 4,
}
EVEN_LOG_X_PAD = 0.05   # room either side of the outer ticks, as a fraction of the log range

X_LABELS = {                           # overrides g.label()
    'comm_kb': 'Bytes Sent',           # the ticks carry the unit (KB/MB/GB)
    'total_time': 'Computation (ms)',  # ticks are drawn in ms (ms_tick), the data stays in s
}

GRID_FIG_SIZE = (2 * g.FIG_SIZE[0], 2 * g.FIG_SIZE[1] + 0.8)
# (width, height) in inches for one dataset's figure: two single plots wide,
# plus room for a title, the two row labels and the legend

X_TICK_SUBS = (1.0, 2.0, 5.0)   # label these points in each decade: ..., 0.02, 0.05, 0.1, 0.2, ...

TICK_SIZE = 10   # tick labels in this figure only (g.TICK_SIZE is 12 everywhere else)


def decimal_tick(value, _pos=None):
    """Tick label as a plain decimal ('0.02'), not scientific notation ('2 x 10^-2')."""
    return f'{value:g}'


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


def draw_panel(ax, runs_by_method, dataset, y_key, x_key=X_KEY):
    """One panel, a line per method. Labels, titles and the legend are set by the caller."""
    ax.grid(True, which='major')
    ax.tick_params(labelsize=TICK_SIZE)

    plotted = []
    for method, runs in runs_by_method.items():
        xs, ys = pu.xy(runs, x_key, y_key, ONLY_IMPROVING, name=f'{method}/{dataset}')
        if not xs:
            continue
        plotted += xs
        ax.plot(xs, ys, label=g.METHOD_LABELS[method], color=g.METHOD_COLORS[method],
                marker=g.METHOD_MARKERS[method])

    if x_key in EVEN_LOG_X_TICKS:
        even_log_ticks(ax, x_key, plotted)
    elif LOG_X:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=X_TICK_SUBS))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(decimal_tick))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        # set_xscale resets the tick locators, so the minor ticks new_figure()
        # switched off come back (labelled 2x10^-2, 3x10^-2, ... on top of each
        # other on a narrow range) unless they are switched off again here

    if y_key == ROUNDS_KEY:   # whole numbers only, no 0.1 steps
        if ROUNDS_LOG_Y:
            ax.set_yscale('log', base=2)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(decimal_tick))
            ax.yaxis.set_minor_locator(ticker.NullLocator())
        else:
            lo, hi, step = ROUNDS_AXIS
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        return

    ax.set_ylim(*Y_LIMS.get((dataset, y_key), Y_LIM))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(Y_TICK_STEP))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))


def plot_dataset(nested_data, dataset, x_key=X_KEY):
    """
    The 2x2 figure for one dataset. The top row shares one latency label under
    it (a subfigure each, so it doesn't collide with the bottom row); the
    bottom row shares the rounds axis, labelled once on the left, and each
    panel names its own cost under it. One legend below everything.
    """
    fig = plt.figure(figsize=GRID_FIG_SIZE, layout='constrained')
    top, bottom = fig.subfigures(2, 1)
    top_axes = top.subplots(1, len(TOP_Y_KEYS), sharex=True)
    bottom_axes = bottom.subplots(1, len(BOTTOM_X_KEYS), sharey=True)

    picked = method_runs(nested_data, dataset)
    rounds_runs = method_runs(nested_data, dataset, best_n=None) if ROUNDS_ALL_CONFIGS else picked

    for ax, y_key in zip(top_axes, TOP_Y_KEYS):
        draw_panel(ax, picked, dataset, y_key, x_key)
        g.add_arrow(ax.set_ylabel(g.label(y_key, K)), y_key, 'y')
    g.add_arrow(top.supxlabel(g.label(x_key), fontsize=g.FONT_SIZE), x_key, 'x')

    for ax, bottom_x in zip(bottom_axes, BOTTOM_X_KEYS):
        draw_panel(ax, rounds_runs, dataset, ROUNDS_KEY, bottom_x)
        g.add_arrow(ax.set_xlabel(x_label(bottom_x)), bottom_x, 'x')
    g.add_arrow(bottom_axes[0].set_ylabel(g.label(ROUNDS_KEY)), ROUNDS_KEY, 'y')
    if TITLE:
        fig.suptitle(TITLE.format(dataset=g.DATASET_LABELS[dataset]))

    pu.method_legend(fig)   # under everything; styled and placed in globals (METHOD_LEGEND_*)

    return pu.save_figure(fig, f'comparison_{dataset}')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        plot_dataset(nested_data, dataset)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
