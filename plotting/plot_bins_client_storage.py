"""
Client storage of the two BM25-Bin layouts as docs per bin grows: one bar pair
per dpb value, 1-stage (vec1) against 2-stage (vec0). Bin size is fixed at
g.BINS_FIXED_BS, so only dpb changes along the x-axis. One PDF per dataset.

    figures/bins_client_storage_<dataset>.pdf
"""

import numpy as np
from matplotlib.patches import Patch

import globals as g
import load_results
import plot_utils as pu

Y_KEY = 'client_storage_mb'   # total over both DBs; '_bm25' / '_vec' for one of them
X_KEY = 'dpb'
K = g.K_ABLATION

VECS = [1, 0]           # bar order within each group, see g.VEC_LABELS
GROUP_WIDTH = 0.8       # share of each x slot filled by its bars
LOG_Y = True


def plot_bins_client_storage(nested_data, dataset, y_key=Y_KEY):
    fig, ax = pu.new_figure()

    # One x slot per dpb value present for either layout, smallest first.
    per_vec = {vec: {run[X_KEY]: run for run in
                     pu.bins_sweep(nested_data, dataset, X_KEY, K, vec=vec)}
               for vec in VECS}
    xs = sorted({x for runs in per_vec.values() for x in runs})
    if not xs:
        pu.log.warning("bins/%s: no runs at bs=%s, nothing to plot",
                       dataset, g.BINS_FIXED_BS.get(dataset))

    slots = np.arange(len(xs), dtype=float)
    bar_width = GROUP_WIDTH / len(VECS)

    for i, vec in enumerate(VECS):
        heights = [per_vec[vec].get(x, {}).get(y_key, np.nan) for x in xs]
        missing = [x for x, h in zip(xs, heights) if not pu.is_number(h)]
        if missing:
            pu.log.warning("bins/%s vec%d: no %s at dpb=%s", dataset, vec, y_key,
                           ', '.join(map(str, missing)))
        offset = (i - (len(VECS) - 1) / 2) * bar_width
        ax.bar(slots + offset, heights, width=bar_width * 0.92,
               color=g.VEC_COLORS[vec], hatch=g.VEC_HATCHES[vec],
               edgecolor='#333333', linewidth=0.6)

    if LOG_Y:
        ax.set_yscale('log')
    ax.set_xticks(slots)
    ax.set_xticklabels([f'{x:g}' for x in xs],
                       rotation=45 if len(xs) > 6 else 0,   # rotate once the labels crowd
                       ha='right' if len(xs) > 6 else 'center')
    ax.tick_params(axis='x', length=0)
    ax.grid(False, axis='x')
    ax.set_xlabel(g.label(X_KEY))
    ax.set_ylabel(g.label(y_key))
    ax.set_title(f"{g.DATASET_LABELS[dataset]} (bin size {g.BINS_FIXED_BS.get(dataset):,})", pad=18)
    # pad pushes the title up to leave room for the legend

    handles = [Patch(facecolor=g.VEC_COLORS[v], hatch=g.VEC_HATCHES[v],
                     edgecolor='#333333', linewidth=0.6, label=g.VEC_LABELS[v]) for v in VECS]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2,
              **g.LEGEND_STYLE)
    # bbox_to_anchor moves the legend: (0.5, 0.98) = centred, just above the axes

    return pu.save_figure(fig, f'bins_client_storage_{dataset}')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        plot_bins_client_storage(nested_data, dataset)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
