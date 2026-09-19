"""
Retrieval quality against the costs that are NOT per-query: preprocessing
rounds, maintenance time and client storage. One line per method, one PDF per
(dataset, cost):

    figures/cost_<dataset>_<cost_key>.pdf

Why this shape. Those three are fixed costs: paid once at setup, or held on the
client for the whole run, so there is no natural sweep parameter to put on the
x-axis and no single number per method either, since every config pays a
different amount. Putting the cost on the x-axis and MRR on the y-axis asks the
question the paper actually wants answered: for a given setup budget, how good
is retrieval? BM25-Bin sitting up and to the LEFT of the other two is the
claim, and the gap is readable straight off the axis. It reads exactly like the
latency plots, just with a different x, so the two can sit side by side.

If you would rather show one number per method, the same keys work in
plot_best_histograms.py: add 'preproc_rounds' to Y_KEYS there and give each
method a value in HAND_PICKED.
"""

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

COST_KEYS = ['preproc_rounds', 'maintenance_time', 'client_storage_mb']
Y_KEY = 'mrr'      # swap for 'recall', 'faithfulness', ...
K = g.K_ABLATION   # tree only has k10 runs for now

BEST_N = None           # e.g. 5 to show only the selected configs
ONLY_IMPROVING = True   # drops configs that cost more without scoring better
LOG_X = True
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs


def plot_quality_vs_cost(nested_data, dataset, cost_key, y_key=Y_KEY):
    fig, ax = pu.new_figure()

    for method in g.METHOD_ORDER:
        fixed = BINS_FILTER if method == 'bins' else {}
        runs = pu.get_runs(nested_data, method, dataset, k=K, **fixed)
        if BEST_N:
            runs = sc.select_configs(runs, n=BEST_N, name=f'{method}/{dataset}')
        xs, ys = pu.xy(runs, cost_key, y_key, ONLY_IMPROVING, name=f'{method}/{dataset}')
        if not xs:
            continue
        ax.plot(xs, ys, label=g.METHOD_LABELS[method], color=g.METHOD_COLORS[method],
                marker=g.METHOD_MARKERS[method])

    if LOG_X:
        ax.set_xscale('log')
    ax.set_xlabel(g.label(cost_key))
    ax.set_ylabel(g.label(y_key, K))
    ax.set_title(g.DATASET_LABELS[dataset])

    pu.method_legend(fig)   # under the x label; styled and placed in globals (METHOD_LEGEND_*)

    return pu.save_figure(fig, f'cost_{dataset}_{cost_key}')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        for cost_key in COST_KEYS:
            plot_quality_vs_cost(nested_data, dataset, cost_key)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
