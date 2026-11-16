"""
Bins DB size as bin size / docs per bin changes, with PACMANN's DB size as a
dotted baseline. Colour = dataset, solid = BM25-Bin, dotted = PACMANN.

    figures/db_size_vs_bs.pdf    (docs per bin fixed at g.BINS_FIXED_DPB)
    figures/db_size_vs_dpb.pdf   (bin size fixed at g.BINS_FIXED_BS)
"""

import globals as g
import load_results
import plot_utils as pu

X_KEYS = ['bs', 'dpb']
Y_KEY = 'db_size_mb'       # total over both DBs; 'db_size_mb_bm25' for the bins DB only
PACMANN_Y_KEY = 'db_size_mb'
K = g.K_ABLATION
BINS_VEC = 1               # 1 = single DB, 0 = split DBs, None = both

# PACMANN's DB grows with neighb, so the baseline comes from one config.
PACMANN_CONFIG = 'steps15_neighb32'

ONLY_IMPROVING = False
LOG_Y = True


def pacmann_baseline(nested_data, dataset):
    """PACMANN DB size for PACMANN_CONFIG (any k), or None."""
    runs = pu.get_runs(nested_data, 'pacmann', dataset, config=PACMANN_CONFIG)
    sizes = sorted({r[PACMANN_Y_KEY] for r in runs if pu.is_number(r[PACMANN_Y_KEY])})
    if not sizes:
        return None
    if len(sizes) > 1:
        pu.warn_once("pacmann %s %s: several DB sizes %s, using the largest",
                     dataset, PACMANN_CONFIG, sizes)
    return sizes[-1]


def plot_db_size(nested_data, x_key):
    fig, ax = pu.new_figure()

    for dataset in g.DATASETS:
        runs = pu.bins_sweep(nested_data, dataset, x_key, K, vec=BINS_VEC)
        xs, ys = pu.xy(runs, x_key, Y_KEY, ONLY_IMPROVING, name=f'db_size/{dataset}')
        if xs:
            ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset])

        baseline = pacmann_baseline(nested_data, dataset)
        if baseline is not None:
            ax.axhline(baseline, color=g.DATASET_COLORS[dataset], linestyle=g.BASELINE_LINESTYLE)

    ax.set_xscale('log')
    if LOG_Y:
        ax.set_yscale('log')
    ax.set_xlabel(g.label(x_key))
    ax.set_ylabel(g.label(Y_KEY))

    styles = [(g.METHOD_LABELS['bins'], '-'), (g.METHOD_LABELS['pacmann'], g.BASELINE_LINESTYLE)]
    pu.dataset_legend(ax, g.DATASETS, styles, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=2)
    # bbox_to_anchor moves the legend; ncol=2 puts datasets in one column, line styles in the other

    return pu.save_figure(fig, f'db_size_vs_{x_key}')


def make_plots(nested_data):
    for x_key in X_KEYS:
        plot_db_size(nested_data, x_key)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
