"""
BM25-Bin with one DB (vec1, "1-stage") against two DBs (vec0, "2-stage"), bin
size fixed at g.BINS_FIXED_BS, docs per bin on the x-axis. Colour = dataset,
line style = number of stages.

    figures/bins_stages_total_time.pdf
    figures/bins_stages_comm_kb.pdf
"""

import globals as g
import load_results
import plot_utils as pu

X_KEY = 'dpb'
Y_KEYS = ['total_time', 'comm_kb']   # 'lan_time' / 'wan_time' also work
K = g.K_ABLATION
ONLY_IMPROVING = False
LOG_Y = True

# vec flag in the folder name -> (legend label, line style)
STAGES = {
    1: ('1-stage', '-'),
    0: ('2-stage', '--'),
}


def plot_bins_stages(nested_data, y_key):
    fig, ax = pu.new_figure()

    for dataset in g.DATASETS:
        for vec, (stage_label, linestyle) in STAGES.items():
            runs = pu.bins_sweep(nested_data, dataset, X_KEY, K, vec=vec)
            xs, ys = pu.xy(runs, X_KEY, y_key, ONLY_IMPROVING, name=f'bins_stages/{dataset}/vec{vec}')
            if not xs:
                continue
            ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset],
                    linestyle=linestyle)

    ax.set_xscale('log')
    if LOG_Y:
        ax.set_yscale('log')
    ax.set_xlabel(g.label(X_KEY))
    ax.set_ylabel(g.label(y_key))

    pu.dataset_legend(ax, g.DATASETS, list(STAGES.values()),
                      loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2)
    # bbox_to_anchor moves the legend; ncol=2 puts datasets in one column, stages in the other

    return pu.save_figure(fig, f'bins_stages_{y_key}')


def make_plots(nested_data):
    for y_key in Y_KEYS:
        plot_bins_stages(nested_data, y_key)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
