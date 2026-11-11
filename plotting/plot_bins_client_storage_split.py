"""
Where PILLAR-Bin's client storage goes, in the look of plot_bins_ablation_split.py
(a dataset per column, its sizes, ticks and legend under the figure):

    figures/bins_client_storage_split_bs.pdf    vs hash table size (dpb fixed, split.FIXED_DPB)
    figures/bins_client_storage_split_dpb.pdf   vs docs per bin (bs fixed, g.BINS_FIXED_BS)

A line per LINES entry: the single-DB layout (vec1) and the split layout (vec0),
the latter both in total and broken into its two hints, the BM25 ID DB alone
(client_storage_mb_bm25) and the embedding DB alone (client_storage_mb_vec).
The ID DB line is what the client holds when it only fetches IDs, i.e. the
non-embedding DB on its own; the gap up to the vec0 total is the price of also
fetching the embeddings privately, and the gap up to vec1 is what splitting the
DB saves.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_bins_ablation_split as split
import plot_pareto_grid as pg
import plot_utils as pu

K = split.K
FIXED_DPB = split.FIXED_DPB   # docs per bin during the hash table size sweep
MAX_DPB = split.MAX_DPB       # runs with more docs per bin are left out (None keeps all)
SWEEPS = ['bs', 'dpb']
DATASETS = g.DATASETS

VEC_COLORS = {1: '#56B4E9', 0: '#D55E00'}   # as plot_bins_vec_db.py
# (label, vec, run key, line style), drawn in this order and listed in the legend
LINES = [
    ('Single DB',             1, 'client_storage_mb',      '-'),
    ('Split: total',          0, 'client_storage_mb',      '-'),
    ('Split: ID DB only',     0, 'client_storage_mb_bm25', '--'),
    ('Split: embedding DB',   0, 'client_storage_mb_vec',  ':'),
]
LEGEND_ROWS = [[0, 1], [2, 3]]   # LINES indices per legend line

LOG_Y = True        # the single DB is often 10x+ the split one; False for a plain axis
Y_LIMS = {}         # (sweep, dataset) or sweep -> (lo, hi) in MB; unlisted = fitted to the data
N_TICKS = split.N_TICKS
Y_LABEL = 'Client'  # every tick carries its unit (MB/GB)
TITLE = 'Client Storage by {sweep}'   # None for no title

PANEL_SIZE = split.PANEL_SIZE
PANEL_GAP = split.PANEL_GAP
HEADROOM = split.HEADROOM
TITLE_SIZE = split.TITLE_SIZE
COL_TITLE_SIZE = split.COL_TITLE_SIZE
X_LABEL_SIZE = split.X_LABEL_SIZE
Y_LABEL_SIZE = split.Y_LABEL_SIZE
LABEL_PAD = split.LABEL_PAD
TICK_SIZE = split.TICK_SIZE
MARKER_SIZE = split.MARKER_SIZE
LEGEND_GAP = split.LEGEND_GAP
LEGEND_SPACING = split.LEGEND_SPACING


def sweep_runs(nested_data, dataset, x_key, vec):
    dpb = FIXED_DPB if x_key == 'bs' else None
    runs = pu.bins_sweep(nested_data, dataset, x_key, K, vec=vec, dpb=dpb)
    if MAX_DPB is not None:
        runs = [r for r in runs if r['dpb'] <= MAX_DPB]
    return runs


def setup_y(ax, x_key, dataset, ys):
    lim = Y_LIMS.get((x_key, dataset), Y_LIMS.get(x_key))
    log = LOG_Y and ys and min(ys) > 0
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    if not (lim or ys):
        return
    lo, hi = lim or (min(ys) if log else 0, max(ys))
    ticks, (lo, hi) = pg.even_ticks(lo, hi, log, bool(lim), n=N_TICKS)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda value, _pos=None: pg.unit_tick(value, 'bytes', 'MB')))


def draw_panel(ax, nested_data, x_key, dataset):
    xs_drawn, ys_drawn = [], []
    for label, vec, y_key, style in LINES:
        xs, ys = pu.xy(sweep_runs(nested_data, dataset, x_key, vec), x_key, y_key, False,
                       name=f'bins_client_storage_split/{x_key}/{dataset}/{label}')
        keep = [(x, y) for x, y in zip(xs, ys) if not (LOG_Y and y <= 0)]   # vec1 has no Vec DB
        if not keep:
            continue
        xs, ys = map(list, zip(*keep))
        ax.plot(xs, ys, color=VEC_COLORS[vec], linestyle=style, marker=g.DATASET_MARKERS[dataset],
                markersize=MARKER_SIZE, clip_on=False)
        xs_drawn += xs
        ys_drawn += ys
    ax.grid(True, which='major')
    ax.tick_params(labelsize=TICK_SIZE)
    split.setup_x(ax, x_key, dataset, xs_drawn)
    setup_y(ax, x_key, dataset, ys_drawn)


def plot_sweep(nested_data, x_key):
    n_cols = len(DATASETS)
    fig = plt.figure(figsize=(PANEL_SIZE[0] * (n_cols + PANEL_GAP * (n_cols - 1)),
                              PANEL_SIZE[1] + HEADROOM), layout='constrained')
    fig.get_layout_engine().set(wspace=PANEL_GAP)
    axes = fig.subplots(1, n_cols, squeeze=False)[0]
    for c, dataset in enumerate(DATASETS):
        draw_panel(axes[c], nested_data, x_key, dataset)
        axes[c].set_title(g.DATASET_LABELS[dataset], fontsize=COL_TITLE_SIZE)
    text = axes[0].set_ylabel(Y_LABEL, fontsize=Y_LABEL_SIZE, labelpad=LABEL_PAD)
    if split.SHOW_ARROWS:
        g.add_arrow(text, 'client_storage_mb', 'y')
    sweep = split.sweep_label(x_key)
    fig.supxlabel(sweep, fontsize=X_LABEL_SIZE)
    if TITLE:
        fig.suptitle(TITLE.format(sweep=sweep.replace(' (log)', '')), fontsize=TITLE_SIZE)
    handles = [Line2D([], [], color=VEC_COLORS[vec], linestyle=style, label=label)
               for label, vec, _, style in LINES]
    pu.legend_rows(fig, [[handles[i] for i in row] for row in LEGEND_ROWS],
                   gap=LEGEND_GAP, style=LEGEND_SPACING)
    return pu.save_figure(fig, f'bins_client_storage_split_{x_key}')


def make_plots(nested_data):
    return [plot_sweep(nested_data, x_key) for x_key in SWEEPS]


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
