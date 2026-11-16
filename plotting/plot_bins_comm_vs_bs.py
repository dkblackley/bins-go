"""
BM25-Bin communication per query as the bin size grows, docs per bin fixed.
One line per dataset.

    figures/bins_comm_vs_bs.pdf
"""

import globals as g
import load_results
import plot_utils as pu

X_KEY = 'bs'
Y_KEY = 'comm_kb'    # 'comm_kb_bm25' / 'comm_kb_vec' for one DB of a vec0 run
K = g.K_ABLATION
FIXED_DPB = 10
BINS_VEC = 1         # 1 = single DB, 0 = split DBs
ONLY_IMPROVING = False
LOG_Y = True


def plot_bins_comm_vs_bs(nested_data):
    fig, ax = pu.new_figure()

    for dataset in g.DATASETS:
        runs = pu.bins_sweep(nested_data, dataset, X_KEY, K, vec=BINS_VEC, dpb=FIXED_DPB)
        xs, ys = pu.xy(runs, X_KEY, Y_KEY, ONLY_IMPROVING, name=f'bins_comm/{dataset}')
        if not xs:
            continue
        ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset])

    ax.set_xscale('log')
    if LOG_Y:
        ax.set_yscale('log')
    ax.set_xlabel(g.label(X_KEY))
    ax.set_ylabel(g.label(Y_KEY))

    pu.dataset_legend(ax, g.DATASETS, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2)
    # bbox_to_anchor moves the legend: (0.5, 0.98) = centred, just above the axes

    return pu.save_figure(fig, 'bins_comm_vs_bs')


def make_plots(nested_data):
    plot_bins_comm_vs_bs(nested_data)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
