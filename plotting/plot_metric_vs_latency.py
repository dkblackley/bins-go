"""
Quality vs latency, one line per method, one PDF per (dataset, metric):

    figures/latency_<dataset>_<y_key>.pdf     (4 metrics x 2 datasets = 8 PDFs)
"""

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

X_KEY = 'lan_time'   # swap for 'total_time' (computation) or 'wan_time'
# Y_KEYS = ['mrr', 'recall', 'faithfulness', 'answer_relevancy']
Y_KEYS = ['mrr']
K = g.K_MAIN

ONLY_IMPROVING = False   # False plots every config, True drops configs that are slower and no better
BEST_N = 5              # keep only this many configs per method (None = every config)
LOG_X = True
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs


def plot_metric_vs_latency(nested_data, dataset, y_key, x_key=X_KEY):
    fig, ax = pu.new_figure()

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
    ax.set_xlabel(g.label(x_key))
    ax.set_ylabel(g.label(y_key, K))
    ax.set_title(g.DATASET_LABELS[dataset], pad=18)
    # pad pushes the title up to leave room for the legend

    if ax.lines:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3, **g.LEGEND_STYLE)
        # bbox_to_anchor moves the legend: (0.5, 0.98) = centred, just above the axes

    return pu.save_figure(fig, f'latency_{dataset}_{y_key}')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        for y_key in Y_KEYS:
            plot_metric_vs_latency(nested_data, dataset, y_key)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
