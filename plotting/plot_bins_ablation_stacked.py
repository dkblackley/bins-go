"""
BM25-Bin ablation as one figure with two panels, side by side:
  left:  communication per query vs bin size (docs per bin fixed at FIXED_DPB)
  right: MRR / Recall vs docs per bin (bin size fixed at g.BINS_FIXED_BS)
Colour = dataset; on the right panel line style = metric. One legend under
the figure, a super title above it.

    figures/bins_ablation_stacked.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_utils as pu

K = g.K_ABLATION
BINS_VEC = 1         # 1 = single DB, 0 = split DBs
ONLY_IMPROVING = False

# left panel: comm vs bin size
COMM_KEY = 'comm_kb'
FIXED_DPB = 10
BS_LOG_X = True   # log bin size axis, a tick per power of 10 (1K, 10K, 0.1M, 1M)
COMM_TICKS = range(70, 145, 15)

# right panel: retrieval scores vs docs per bin
SCORE_KEYS = ['mrr', 'recall']
SCORE_LINESTYLES = ['--', ':']   # 1st, 2nd metric
SCORE_LABELS = {'mrr': 'MRR', 'recall': 'Recall'}
DPB_LOG_X = True   # log docs per bin axis, a tick per power of 10; False = linear with DPB_TICKS
DPB_TICKS = range(0, 2501, 500)
SCORE_TICKS = [0.0, 0.25, 0.45, 0.65, 0.85]

FIG_SIZE = (g.FIG_SIZE[0] * 2 + 1.2, g.FIG_SIZE[1] + 0.3)   # two panels side by side, wider than tall
WSPACE = 0.5   # gap between the panels (fraction of the mean axes width), room for the right y label
TITLE = 'PILLAR.Bins Ablation'
TITLE_SIZE = 22
LABEL_SIZE = g.FONT_SIZE + 4   # x/y axis labels, a little bigger than the other plots
LEGEND_GAP = -5   # gap between the figure and the legend under it, in points
# tighter than g.LEGEND_STYLE: labelspacing = gap between rows, columnspacing = between
# columns, handletextpad = between a line and its text (all in font-size units)
LEGEND_SPACING = dict(labelspacing=0.1, columnspacing=0.5, handletextpad=0.15)


def short_number(value, _pos=None):
    """0 -> '0', 400 -> '0.4K', 6000 -> '6K', 200000 -> '0.2M', 1000000 -> '1M'."""
    if value == 0:
        return '0'
    if abs(value) >= 1e5:
        return f'{value / 1e6:g}M'
    return f'{value / 1e3:g}K'


def set_axis(axis, ticks, short=True):
    """Fixed ticks; short=True labels them with short_number, else as plain numbers."""
    axis.set_major_locator(ticker.FixedLocator(list(ticks)))
    if short:
        axis.set_major_formatter(ticker.FuncFormatter(short_number))


def set_log_x(ax):
    """Log x axis with a labelled tick at every power of 10 inside the data range.
    numticks is set explicitly: the default ('auto') skips every other power of 10
    on an axis this short."""
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(short_number))
    ax.xaxis.set_minor_locator(ticker.NullLocator())   # set_xscale turns the minor ticks back on


def plot_comm(ax, nested_data):
    for dataset in g.DATASETS:
        runs = pu.bins_sweep(nested_data, dataset, 'bs', K, vec=0, dpb=FIXED_DPB)
        xs, ys = pu.xy(runs, 'bs', COMM_KEY, ONLY_IMPROVING, name=f'bins_stacked_comm/{dataset}')
        if not xs:
            continue
        ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset])

    if BS_LOG_X:
        set_log_x(ax)
    ax.set_ylim(COMM_TICKS[0], COMM_TICKS[-1])
    set_axis(ax.yaxis, COMM_TICKS, short=False)   # plain numbers
    ax.set_xlabel(g.label('bs'), fontsize=LABEL_SIZE)
    ax.set_ylabel(g.label(COMM_KEY), fontsize=LABEL_SIZE)


def plot_scores(ax, nested_data):
    for dataset in g.DATASETS:
        runs = pu.bins_sweep(nested_data, dataset, 'dpb', K, vec=BINS_VEC)
        for y_key, linestyle in zip(SCORE_KEYS, SCORE_LINESTYLES):
            xs, ys = pu.xy(runs, 'dpb', y_key, ONLY_IMPROVING, name=f'bins_stacked_score/{dataset}')
            if not xs:
                continue
            ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset],
                    linestyle=linestyle)

    if DPB_LOG_X:
        set_log_x(ax)
    else:
        ax.set_xlim(DPB_TICKS[0], DPB_TICKS[-1])
        set_axis(ax.xaxis, DPB_TICKS)
    ax.set_ylim(SCORE_TICKS[0], SCORE_TICKS[-1])
    ax.yaxis.set_major_locator(ticker.FixedLocator(SCORE_TICKS))
    ax.set_xlabel(g.label('dpb'), fontsize=LABEL_SIZE)
    ax.set_ylabel('Score', fontsize=LABEL_SIZE)


def legend_below(fig):
    """One legend under the whole figure: dataset colours, then the MRR/Recall line styles."""
    handles = [Line2D([], [], color=g.DATASET_COLORS[d], marker=g.DATASET_MARKERS[d],
                      label=g.DATASET_LABELS[d]) for d in g.DATASETS]
    handles += [Line2D([], [], color='black', linestyle=ls, label=SCORE_LABELS[y])
                for y, ls in zip(SCORE_KEYS, SCORE_LINESTYLES)]

    fig.draw_without_rendering()   # lay out the figure so the legend knows where the bottom is
    bottom = fig.get_tightbbox().y0 / fig.get_figheight()
    gap = LEGEND_GAP / 72 / fig.get_figheight()   # points -> figure fraction
    # all entries on one line, the figure is wide enough
    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.45, bottom - gap),
                      ncol=len(handles), **{**g.LEGEND_STYLE, **LEGEND_SPACING})


def plot_bins_ablation_stacked(nested_data):
    fig, (ax_comm, ax_score) = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.subplots_adjust(wspace=WSPACE)
    for ax in (ax_comm, ax_score):
        ax.grid(True, which='major')
        ax.minorticks_off()

    plot_comm(ax_comm, nested_data)
    plot_scores(ax_score, nested_data)
    fig.suptitle(TITLE, fontsize=TITLE_SIZE, fontweight='bold', y=1.06)
    legend_below(fig)

    return pu.save_figure(fig, 'bins_ablation_stacked')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        pu.log.info("bins stacked ablation on %s: dpb=%s for the comm sweep, bs=%s for the score sweep, k=%d, vec=%s",
                    dataset, FIXED_DPB, g.BINS_FIXED_BS.get(dataset), K, BINS_VEC)
    plot_bins_ablation_stacked(nested_data)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())