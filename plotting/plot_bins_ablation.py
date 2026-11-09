"""
BM25-Bin ablation: how each metric changes with bin size (docs per bin fixed
at g.BINS_FIXED_DPB) and with docs per bin (bin size fixed at g.BINS_FIXED_BS).
Colour = dataset; when a panel holds two metrics, line style tells them apart.

    figures/bins_ablation_<x_key>_<panel>.pdf    (2 sweeps x 6 panels = 12 PDFs)
"""

import globals as g
import load_results
import plot_utils as pu

X_KEYS = ['bs', 'dpb']
K = g.K_ABLATION
BINS_VEC = 1             # 1 = single DB, 0 = split DBs, None = both (don't mix them here)
ONLY_IMPROVING = False

# panel name -> run keys drawn on it. Add/remove/reorder panels here.
PANELS = {
    'retrieval':  ['mrr', 'recall'],
    'generation': ['faithfulness', 'answer_relevancy'],
    'runtime':    ['total_time'],
    'lan':        ['lan_time'],
    'wan':        ['wan_time'],
    'comm':       ['comm_kb'],
}
PANEL_YLABELS = {'retrieval': 'Score', 'generation': 'Score'}   # single-metric panels use g.label
LOG_Y_PANELS = ['runtime', 'lan', 'wan', 'comm']
LINESTYLES = ['-', '--', ':']   # 1st, 2nd, 3rd metric on a panel


def plot_bins_ablation(nested_data, x_key, panel, y_keys):
    fig, ax = pu.new_figure()

    for dataset in g.DATASETS:
        runs = pu.bins_sweep(nested_data, dataset, x_key, K, vec=BINS_VEC)
        for y_key, linestyle in zip(y_keys, LINESTYLES):
            xs, ys = pu.xy(runs, x_key, y_key, ONLY_IMPROVING, name=f'bins_ablation/{dataset}')
            if not xs:
                continue
            ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset],
                    linestyle=linestyle)

    ax.set_xscale('log')
    if panel in LOG_Y_PANELS:
        ax.set_yscale('log')
    ax.set_xlabel(g.label(x_key))
    ax.set_ylabel(PANEL_YLABELS.get(panel, g.label(y_keys[0], K)))

    styles = [(g.label(y, K), ls) for y, ls in zip(y_keys, LINESTYLES)] if len(y_keys) > 1 else []
    pu.dataset_legend(ax, g.DATASETS, styles, loc='lower center', bbox_to_anchor=(0.5, 0.98),
                      ncol=2)
    # bbox_to_anchor moves the legend; ncol=2 puts datasets in one column, metrics in the other

    return pu.save_figure(fig, f'bins_ablation_{x_key}_{panel}')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        pu.log.info("bins ablation on %s: dpb=%s for the bs sweep, bs=%s for the dpb sweep, k=%d, vec=%s",
                    dataset, g.BINS_FIXED_DPB, g.BINS_FIXED_BS.get(dataset), K, BINS_VEC)
    for x_key in X_KEYS:
        for panel, y_keys in PANELS.items():
            plot_bins_ablation(nested_data, x_key, panel, y_keys)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
