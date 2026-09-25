"""
BM25-Tree ablation: where the latency goes. One bar per selected config, split
into the three PIR stages. One panel per dataset, side by side, one legend
under the figure, a super title above it.

    figures/tree_stages.pdf

The configs come from select_configs.py, so they are the same five the other
tree plots use. Set CONFIGS below to pin specific ones by hand instead. The
x axis just says C1, C2, ... (smallest total first, largest on the right); which config each one is
gets printed. Both panels share one y label.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

Y_KEY = 'wan_time'    # stacked as lan_time_s1/_s2/_s3; 'wan_time' and 'comm_kb' also split
Y_LABEL = None        # None = g.label(Y_KEY)
K = g.K_ABLATION
N_CONFIGS = 5
DATASETS = g.DATASETS   # one panel per dataset, left to right

# {dataset: [config, ...]} to choose the bars by hand; empty = use select_configs.
CONFIGS = {}
CONFIG_LABEL = 'C{}'   # x tick label, filled with 1, 2, 3, ...
X_LABELS = {'msmarco': 'Tree MS MARCO Configs', 'scifact': 'Tree SciFact Configs'}   # missing = no x label
SORT_BY_TOTAL = True   # bars left to right from smallest total to largest; False = selection order

# bars
BAR_WIDTH = 0.65
BAR_EDGE_COLOR = '#333333'
BAR_EDGE_WIDTH = 0.6
SHOW_TOTALS = False   # print the total above each bar
TOTALS_SIZE = g.LEGEND_SIZE - 1

# y axis: {dataset: ticks} pins that panel's y range/ticks; missing = automatic
Y_TICKS = {'msmarco': [0, 0.35, 0.65, 0.95], 'scifact': [0, 0.1, 0.2, 0.3]}
LOG_Y = False
Y_LABEL_X = 0.0  # x of the one shared y label, as a figure fraction; lower = further left

# panels
SHOW_PANEL_TITLES = True   # dataset name above each panel
PANEL_TITLE_SIZE = g.FONT_SIZE
PANEL_TITLE_PAD = 0

FIG_SIZE = (g.FIG_SIZE[0] * 2 + 0.8, g.FIG_SIZE[1] - 0.3)   # two panels side by side, wider than tall
WSPACE = 0.3   # gap between the panels (fraction of the mean axes width)
TITLE = 'PILLAR-Tree Ablation'
TITLE_SIZE = 22
TITLE_Y = 1.23   # top of the super title, as a figure fraction; lower it if SHOW_PANEL_TITLES is False
LABEL_SIZE = g.FONT_SIZE + 2   # x/y axis labels, a little bigger than the other plots
TICK_SIZE = g.TICK_SIZE
LEGEND_NCOL = 3   # 3 = all stages on one line
LEGEND_GAP = -1    # gap between the figure and the legend under it, in points
# tighter than g.LEGEND_STYLE: labelspacing = gap between rows, columnspacing = between
# columns, handletextpad = between a patch and its text (all in font-size units)
LEGEND_SPACING = dict(labelspacing=0.2, columnspacing=0.5, handletextpad=0.15)


def pick_runs(nested_data, dataset):
    runs = pu.get_runs(nested_data, 'tree', dataset, k=K)
    if CONFIGS.get(dataset):
        by_config = {r['config']: r for r in runs}
        runs = [by_config[c] for c in CONFIGS[dataset] if c in by_config]
        missing = [c for c in CONFIGS[dataset] if c not in by_config]
        if missing:
            pu.log.warning("tree/%s: no run for config(s) %s", dataset, ', '.join(missing))
    else:
        runs = sc.select_configs(runs, n=N_CONFIGS, name=f'tree/{dataset}')
    if not runs:
        pu.log.warning("tree/%s: nothing to plot", dataset)
    return runs


def stage_total(run, y_key):
    """Sum of the three stages, missing stages as 0."""
    return sum(np.nan_to_num(run.get(f'{y_key}_{stage}', np.nan)) for stage in g.STAGE_ORDER)


def plot_panel(ax, nested_data, dataset, y_key):
    runs = pick_runs(nested_data, dataset)
    if SORT_BY_TOTAL:
        runs = sorted(runs, key=lambda r: stage_total(r, y_key))
    slots = np.arange(len(runs), dtype=float)
    bottoms = np.zeros(len(runs))

    for stage in g.STAGE_ORDER:
        heights = np.array([run.get(f'{y_key}_{stage}', np.nan) for run in runs])
        if np.all(np.isnan(heights)) and runs:
            pu.log.warning("tree/%s: %s missing from every run", dataset, stage)
        heights = np.nan_to_num(heights)
        ax.bar(slots, heights, bottom=bottoms, width=BAR_WIDTH, color=g.STAGE_COLORS[stage],
               edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        bottoms += heights

    if SHOW_TOTALS:
        for x, total in zip(slots, bottoms):
            ax.text(x, total, f'{total:.3g}', ha='center', va='bottom', fontsize=TOTALS_SIZE)

    labels = [CONFIG_LABEL.format(i + 1) for i in range(len(runs))]
    print(f"{g.DATASET_LABELS[dataset]} ({y_key}):")
    for text, run in zip(labels, runs):
        print(f"  {text} = {run['config']}  (total {stage_total(run, y_key):.3g})")
    ax.set_xticks(slots)
    ax.set_xticklabels(labels)
    if dataset in X_LABELS:
        ax.set_xlabel(X_LABELS[dataset], fontsize=LABEL_SIZE, x=0.46)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(False, axis='x')

    if LOG_Y:
        ax.set_yscale('log')
    if dataset in Y_TICKS:
        ticks = list(Y_TICKS[dataset])
        ax.set_ylim(ticks[0], ticks[-1])
        ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    if SHOW_PANEL_TITLES:
        ax.set_title(g.DATASET_LABELS[dataset], fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)


def legend_below(fig):
    """One legend under the whole figure, one entry per PIR stage."""
    handles = [Patch(facecolor=g.STAGE_COLORS[s], edgecolor=BAR_EDGE_COLOR,
                     linewidth=BAR_EDGE_WIDTH, label=g.STAGE_LABELS[s]) for s in g.STAGE_ORDER]

    fig.draw_without_rendering()   # lay out the figure so the legend knows where the bottom is
    bottom = fig.get_tightbbox().y0 / fig.get_figheight()
    gap = LEGEND_GAP / 72 / fig.get_figheight()   # points -> figure fraction
    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, bottom - gap),
                      ncol=LEGEND_NCOL, **{**g.LEGEND_STYLE, **LEGEND_SPACING})


def plot_tree_stages(nested_data, y_key=Y_KEY):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=FIG_SIZE, squeeze=False)
    axes = axes[0, :]
    fig.subplots_adjust(wspace=WSPACE)
    for ax, dataset in zip(axes, DATASETS):
        ax.grid(True, which='major')
        ax.minorticks_off()
        plot_panel(ax, nested_data, dataset, y_key)

    fig.supylabel(Y_LABEL or g.label(y_key, K), fontsize=LABEL_SIZE + 2, x=Y_LABEL_X)   # one for both panels
    fig.suptitle(TITLE, fontsize=TITLE_SIZE, y=TITLE_Y)
    legend_below(fig)

    return pu.save_figure(fig, 'tree_stages')


def make_plots(nested_data):
    plot_tree_stages(nested_data)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())