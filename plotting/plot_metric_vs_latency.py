"""
Quality vs latency, one line per method, all in one grid PDF
(rows = metrics in Y_KEYS, columns = datasets):

    figures/latency_grid_<x_key>.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

X_KEY = 'wan_time'   # swap for 'total_time' (computation) or 'wan_time'
# Y_KEYS = ['mrr', 'recall', 'faithfulness', 'answer_relevancy']
Y_KEYS = ['mrr', 'recall']   # one grid row each, top to bottom
K = g.K_MAIN

ONLY_IMPROVING = False   # False plots every config, True drops configs that are slower and no better
BEST_N = 5              # keep only this many configs per method (None = every config)
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

GRID_FIG_SIZE = (2 * g.FIG_SIZE[0], 2 * g.FIG_SIZE[1])
# (width, height) in inches for the whole grid: two single plots wide, plus
# room for the shared labels and the legend. Rows are taller than FIG_SIZE
# so each row's y label ('Recall@10') fits beside its own row without
# running into the next one

X_TICK_SUBS = (1.0, 2.0, 5.0)   # label these points in each decade: ..., 0.02, 0.05, 0.1, 0.2, ...


def decimal_tick(value, _pos=None):
    """Tick label as a plain decimal ('0.02'), not scientific notation ('2 x 10^-2')."""
    return f'{value:g}'


def draw_panel(ax, nested_data, dataset, y_key, x_key=X_KEY):
    """One (dataset, metric) panel. Labels, titles and the legend are set by the caller."""
    ax.grid(True, which='major')

    for method in g.METHOD_ORDER:
        fixed = BINS_FILTER if method == 'bins' else {}
        runs = pu.get_runs(nested_data, method, dataset, k=K, **fixed)
        if BEST_N:
            runs = sc.select_configs(runs, n=BEST_N, name=f'{method}/{dataset}')
        xs, ys = pu.xy(runs, x_key, y_key, ONLY_IMPROVING, name=f'{method}/{dataset}')
        if not xs:
            continue
        ax.plot(xs, ys, label=g.METHOD_LABELS[method], color=g.METHOD_COLORS[method],
                marker=g.METHOD_MARKERS[method])

    if LOG_X:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=X_TICK_SUBS))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(decimal_tick))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        # set_xscale resets the tick locators, so the minor ticks new_figure()
        # switched off come back (labelled 2x10^-2, 3x10^-2, ... on top of each
        # other on a narrow range) unless they are switched off again here

    ax.set_ylim(*Y_LIMS.get((dataset, y_key), Y_LIM))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(Y_TICK_STEP))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))


def plot_metric_vs_latency(nested_data, x_key=X_KEY):
    """
    One grid: a row per metric in Y_KEYS, a column per dataset in g.DATASETS.
    Each label is printed once: dataset names above the top row, the metric
    on the left of each row, the latency centred under the whole grid, and a
    single legend above everything.
    """
    n_rows, n_cols = len(Y_KEYS), len(g.DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=GRID_FIG_SIZE, sharex='col',
                             squeeze=False, layout='constrained')
    # sharex='col': both rows of a column show the same dataset, so they share
    # the latency range and only the bottom row needs x tick labels

    for row, y_key in enumerate(Y_KEYS):
        for col, dataset in enumerate(g.DATASETS):
            ax = axes[row][col]
            draw_panel(ax, nested_data, dataset, y_key, x_key)
            if row == 0:
                ax.set_title(g.DATASET_LABELS[dataset])
            if col == 0:
                ax.set_ylabel(g.label(y_key, K, axis='y'), fontsize=16)
            if row < n_rows - 1:
                ax.tick_params(labelbottom=False)

    fig.supxlabel(g.label(x_key, axis='x'), fontsize=g.FONT_SIZE, y=-0.07)

    # every panel draws the same methods, so take the handles from whichever has the most
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes.flat),
                          key=lambda hl: len(hl[0]))
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=len(handles),
                   **g.LEGEND_STYLE)
        # 'outside' makes constrained layout reserve room above the top row

    return pu.save_figure(fig, f'latency_grid_{x_key}')


def make_plots(nested_data):
    plot_metric_vs_latency(nested_data)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
