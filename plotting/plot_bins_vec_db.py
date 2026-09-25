"""
What the separate embedding DB costs, in the look of plot_bins_ablation_grid.py
and plot_pareto_grid.py (their panel size, gaps, font sizes and evenly spaced
ticks), with bars drawn like plot_best_histograms.py:

    figures/bins_vec_db_quality.pdf   bars: one panel per QUALITY_KEYS metric, and in
                                      each a slot per dataset holding one bar per layout,
                                      at bin size g.BINS_FIXED_BS and docs per bin FIXED_DPB
    figures/bins_vec_db_cost.pdf      lines: a 2x2 of the COST_KEYS metrics whose two
                                      datasets share a range (WAN, LAN, Compute, Data Sent),
                                      docs per bin (up to MAX_DPB) on x, a line per
                                      (dataset, layout)
    figures/bins_vec_db_storage.pdf           lines, a dataset per column (SPLIT_FIGURES),
    figures/bins_vec_db_preproc_quality.pdf   for the metrics whose datasets are too far
                                              apart to share an axis (Server / Client
                                              storage, Preproc. / Quality): a row per metric,
                                              a line per layout, as plot_bins_ablation_split.py

The two layouts are the vec flag in the run folder: vec1 keeps everything in
one PIR DB (embeddings off), vec0 puts the embeddings in a second DB of their
own (embeddings on), which is one more PIR round. The bars show whether that
buys any quality; the lines show what it costs.

Every y axis gets N_TICKS evenly spaced ticks with the first and the last on
the ends of the axis (plot_pareto_grid.even_ticks), and the costs carry their
unit in each tick ('1.3ms', '0.4MB').

This replaces plot_bins_stages.py and the vec0/vec1 half of
plot_bins_client_storage.py.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import globals as g
import load_results
import plot_pareto_grid as pg
import plot_utils as pu

K = g.K_ABLATION
ONLY_IMPROVING = False

VECS = [1, 0]        # bar order in each slot and line order in the legend
VEC_LABELS = {1: 'Single DB', 0: 'Separate Embedding DB'}   # g.VEC_LABELS says it in stages
VEC_COLORS = {1: '#56B4E9', 0: '#D55E00'}   # the bars, Okabe-Ito sky blue and vermillion
VEC_LINESTYLES = {1: '-', 0: '--'}          # the lines, where colour is the dataset
FIXED_DPB = 500      # the bars are one config: this many docs per bin, bin size g.BINS_FIXED_BS

# ==========================================
# WHAT GOES IN EACH FIGURE
# ==========================================

QUALITY_KEYS = ['mrr', 'recall']   # the bar panels
COST_KEYS = ['wan_time', 'lan_time', 'total_time', 'comm_kb']   # the 2x2 line panels, both
                                                                 # datasets on each

# A dataset per column, a row per entry of 'keys'. A row is a run key, or a tuple of
# run keys sharing one axes (line style = metric, METRIC_LINESTYLES); colour is the
# layout (VEC_COLORS), as on the bars. One figure per entry: figures/<name>.pdf.
SPLIT_FIGURES = [
    dict(name='bins_vec_db_storage', title='Embedding DB: Storage',
         keys=['db_size_mb', 'client_storage_mb']),
    dict(name='bins_vec_db_preproc_quality', title='Embedding DB: Preprocessing and Quality',
         keys=['maintenance_time', ('mrr', 'recall')]),
]
DATASETS = g.DATASETS   # the split figures' columns, left to right

# per figure: file name, super title (None for none), panels per row (None = all
# on one row), (width, height) in inches of one panel, and inches added to the
# height for the title and the x label. The bar panels are wider than the pareto
# grid's (PANEL_SIZE), about plot_best_histograms.py's, so the dataset names fit
FIGURES = {
    'quality': dict(name='bins_vec_db_quality', title='Embedding DB: Quality',
                    cols=None, panel=(2.5, 1.5), headroom=0.6),
    'cost':    dict(name='bins_vec_db_cost', title='Embedding DB Ablation',
                    cols=2, panel=(2.5, 1.5), headroom=0.0),   # None = PANEL_SIZE
}

# Shorter than g.label() where the panel is narrow; anything not listed uses g.label().
# The costs have no unit here: every tick carries its own (Y_FORMATS).
Y_LABELS = {
    ('mrr', 'recall'): 'Quality',
    'mrr': 'MRR@{k}', 'recall': 'Recall@{k}', 'answer_relevancy': 'Relevancy',
    'faithfulness': 'Faithful.', 'wan_time': 'WAN', 'lan_time': 'LAN',
    'total_time': 'Compute', 'comm_kb': 'Data Sent', 'db_size_mb': 'Server',
    'client_storage_mb': 'Client', 'maintenance_time': 'Preproc.',
}

METRIC_LINESTYLES = {'mrr': '--', 'recall': ':'}   # in a split panel of several metrics; a
                                                   # panel of one is always solid
METRIC_LABELS = {'mrr': 'MRR', 'recall': 'Recall'}   # their legend entries

# ==========================================
# THE Y AXES
# ==========================================

LOG_Y_KEYS = []      # e.g. ['db_size_mb', 'client_storage_mb'] for a log axis on those
Y_FORMATS = {        # y key -> (unit table, the unit the key is stored in): every tick
    'wan_time':          ('seconds', 's'),        # labelled in its own unit, '1.3ms', '2m'
    'lan_time':          ('seconds', 's'),
    'total_time':        ('seconds', 's'),
    'maintenance_time':  ('seconds', 's'),
    'comm_kb':           ('bytes', 'KB'),
    'db_size_mb':        ('bytes', 'MB'),
    'client_storage_mb': ('bytes', 'MB'),
}                    # anything else is a plain number of pg.TICK_SIG figures
# y key -> (lo, hi), IN THE STORED UNIT (seconds for the times, KB for comm_kb,
# MB for db_size_mb and client_storage_mb, so 1 GB is (0, 1024)). N_TICKS still
# divides a pinned range evenly. A key with no entry is fitted to the data, with
# round ticks (from 0 when Y_FROM_ZERO).
Y_LIMS = {
    # 'mrr':        (0.0, 0.8),
    # 'recall':     (0.0, 0.9),
    # 'wan_time':   (0.0, 0.4),
    # 'lan_time':   (0.0, 0.04),
    # 'comm_kb':    (0, 1024),
}
# The split figures' panels: (dataset, row key) for one panel, or a bare row key for
# that row's panel in every column, -> (lo, hi) in the stored unit, as Y_LIMS.
SPLIT_Y_LIMS = {
    # ('msmarco', 'db_size_mb'):        (0, 15024),
    # ('scifact', 'db_size_mb'):        (0, 1024),
    # ('msmarco', 'client_storage_mb'): (0, 15024),
    # ('scifact', 'client_storage_mb'): (0, 1024),
    # ('msmarco', 'maintenance_time'):  (0, 900),
    # ('scifact', 'maintenance_time'):  (0, 900),
    # ('mrr', 'recall'):                (0.0, 0.8),
}
Y_FROM_ZERO = True   # a plain y axis with no Y_LIMS entry starts at 0 (bars always need it)
N_TICKS = pg.N_TICKS   # y ticks per panel, the first and the last on the ends of the axis

# ==========================================
# THE X AXIS (the lines)
# ==========================================

X_KEY = 'dpb'        # the swept parameter on the line figure (a plain, not log, axis)
MAX_DPB = 500        # runs with more docs per bin than this are left out (the ones past it
                     # broke); None keeps every run
X_TICKS = [0, 250, 500]   # more than 3 run into each other on these narrow panels
X_LIM = (0, 500)
X_LABEL = 'Docs per Bin'

# ==========================================
# SIZES (borrowed from plot_pareto_grid.py, as plot_bins_ablation_grid.py does;
# put numbers in place of the pg.* ones to tune this file on its own)
# ==========================================

PANEL_SIZE = pg.PANEL_SIZE   # (width, height) in inches of one panel, before the gaps, for a
                             # figure whose FIGURES entry has no 'panel' of its own
PANEL_GAP = pg.PANEL_GAP     # space between two columns, as a fraction of a panel's width
ROW_GAP = pg.ROW_GAP         # space between two rows, as a fraction of a panel's height

TITLE_SIZE = pg.SUPTITLE_SIZE
X_LABEL_SIZE = pg.X_LABEL_SIZE
Y_LABEL_SIZE = pg.Y_LABEL_SIZE
LABEL_PAD = pg.LABEL_PAD
TICK_SIZE = pg.TICK_SIZE
COL_TITLE_SIZE = pg.TITLE_SIZE   # the dataset's name over each column of a split figure
Y_LABEL_Y = pg.Y_LABEL_Y     # height of a y label up its panel, 0 = bottom, 1 = top
SHOW_ARROWS = pg.SHOW_ARROWS   # the better-direction badge after each y label

MARKER_SIZE = g.MARKER_SIZE - 1.5   # the points sit close together
LEGEND_GAP = 2   # gap between the figure and the legend under it, in points
LEGEND_LINE_GAP = 0   # gap between two lines of the legend, in points (the datasets on one
                      # line, the layouts on the next)
# labelspacing = gap between rows, columnspacing = between columns, handletextpad =
# between a line and its text (all in font-size units)
LEGEND_SPACING = dict(labelspacing=0.0, columnspacing=0.4, handletextpad=0.1,
                      handlelength=0.6,   # long enough to show a dashed line
                      fontsize=pg.LEGEND_STYLE.get('fontsize', g.LEGEND_SIZE))

# ==========================================
# THE BARS (as in plot_best_histograms.py)
# ==========================================

BAR_GROUP_WIDTH = 0.7   # share of each dataset slot filled by its bars
BAR_FILL = 0.8          # a bar's width as a share of its place in the slot (the rest is the gap)
EDGE_DARKEN = 0.6       # bar outline = fill colour with RGB scaled by this (0 = black, 1 = same)
EDGE_WIDTH = 1.0
SHOW_VALUES = False     # print each bar's height above it
DATASET_LABEL_SIZE = g.FONT_SIZE   # the dataset names under the bars


def darker(color, factor=EDGE_DARKEN):
    """The same hue, darker: each RGB channel scaled by factor."""
    r, gr, b = mcolors.to_rgb(color)
    return (r * factor, gr * factor, b * factor)


def metrics(panel):
    """The run keys one panel draws: itself, or each key of a tuple."""
    return list(panel) if isinstance(panel, tuple) else [panel]


def y_label(panel):
    default = ' / '.join(g.label(key, K) for key in metrics(panel))
    return Y_LABELS.get(panel, default).format(k=K)


# ==========================================
# THE GRID
# ==========================================

def new_grid(keys, cols, panel, headroom):
    """A constrained-layout figure of len(keys) panels, `cols` across; spare cells hidden."""
    panel = panel or PANEL_SIZE
    n_cols = min(cols or len(keys), len(keys))
    n_rows = -(-len(keys) // n_cols)
    fig = plt.figure(figsize=(panel[0] * (n_cols + PANEL_GAP * (n_cols - 1)),
                              panel[1] * (n_rows + ROW_GAP * (n_rows - 1)) + headroom),
                     layout='constrained')
    fig.get_layout_engine().set(wspace=PANEL_GAP, hspace=ROW_GAP)
    axes = list(fig.subplots(n_rows, n_cols, squeeze=False).flat)
    for ax in axes[len(keys):]:
        ax.set_axis_off()   # spare cell on a short last row
    return fig, axes[:len(keys)]


def setup_y(ax, panel, ys, lim=None, label=True, name=''):
    """
    N_TICKS evenly spaced y ticks, the outer two on the ends of the axis, and the
    label (unless `label` is False). `lim` pins the range, else it is fitted to ys.
    """
    key = metrics(panel)[0]
    log = key in LOG_Y_KEYS and ys and min(ys) > 0
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(ticker.NullLocator())   # set_yscale brings them back
    if lim and ys and (min(ys) < lim[0] or max(ys) > lim[1]):
        pu.warn_once("bins_vec_db: %s%s runs %.4g to %.4g, outside its Y_LIMS range %s "
                     "(in the stored unit), so part of it is off the panel",
                     panel, name, min(ys), max(ys), lim)
    if lim or ys:
        lo, hi = lim or (0 if Y_FROM_ZERO and not log and min(ys) >= 0 else min(ys), max(ys))
        if hi <= lo:   # a flat line: give it some room rather than a zero-height axis
            lo, hi = (lo / 2, hi * 2) if log else (lo - (abs(lo) or 1) / 2, hi + (abs(hi) or 1) / 2)
        ticks, (lo, hi) = pg.even_ticks(lo, hi, log, bool(lim), n=N_TICKS)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))

    if key in Y_FORMATS:
        kind, base = Y_FORMATS[key]
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda value, _pos=None: pg.unit_tick(value, kind, base)))
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _pos=None: pg.short(value)))

    ax.tick_params(labelsize=TICK_SIZE)
    if not label:
        return
    text = ax.set_ylabel(y_label(panel), fontsize=Y_LABEL_SIZE, labelpad=LABEL_PAD, y=Y_LABEL_Y)
    if SHOW_ARROWS and len({g.METRICS.get(k, {}).get('better') for k in metrics(panel)}) == 1:
        g.add_arrow(text, key, 'y')   # a tuple only when its metrics agree on 'better'


def setup_x(ax):
    """The docs per bin axis of the lines: a plain axis, X_TICKS over X_LIM."""
    ax.xaxis.set_major_locator(ticker.FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(pu.short_number))
    ax.set_xlim(*X_LIM)


def clip_if_outside(ax, xs, ys):
    """Lines are drawn unclipped (a point on the end tick would be cut in half), but one
    running past a pinned limit would then spill over the rest of the figure."""
    (x_lo, x_hi), (y_lo, y_hi) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    if any(not (x_lo <= x <= x_hi) for x in xs) or any(not (y_lo <= y <= y_hi) for y in ys):
        for line in ax.lines:
            line.set_clip_on(True)


def finish(fig, figure, legend, x_label=None):
    """The shared x label, the super title and the legend under everything, a line
    per entry of `legend` (a list of handle lists)."""
    if x_label:
        fig.supxlabel(x_label, fontsize=X_LABEL_SIZE)
    if figure['title']:
        fig.suptitle(figure['title'], fontsize=TITLE_SIZE)
    pu.legend_rows(fig, legend, gap=LEGEND_GAP, line_gap=LEGEND_LINE_GAP, style=LEGEND_SPACING)
    return pu.save_figure(fig, figure['name'])


def sweep_runs(nested_data, dataset, vec):
    """The docs per bin sweep of one layout, bin size g.BINS_FIXED_BS, up to MAX_DPB."""
    runs = pu.bins_sweep(nested_data, dataset, X_KEY, K, vec=vec)
    if MAX_DPB is not None:
        runs = [r for r in runs if r['dpb'] <= MAX_DPB]
    return runs


def dataset_handles():
    return [Line2D([], [], color=g.DATASET_COLORS[d], marker=g.DATASET_MARKERS[d],
                   markersize=MARKER_SIZE, label=g.DATASET_LABELS[d]) for d in g.DATASETS]


# ==========================================
# THE BARS
# ==========================================

def fixed_run(nested_data, dataset, vec):
    """The one run behind a bar: bin size g.BINS_FIXED_BS, docs per bin FIXED_DPB."""
    runs = pu.get_runs(nested_data, 'bins', dataset, k=K, vec=vec,
                       bs=g.BINS_FIXED_BS.get(dataset), dpb=FIXED_DPB)
    if not runs:
        return None
    if len(runs) > 1:
        pu.warn_once("bins_vec_db/%s vec%d: %d runs at bs=%s dpb=%s, using %s", dataset, vec,
                     len(runs), g.BINS_FIXED_BS.get(dataset), FIXED_DPB, runs[0]['folder'])
    return runs[0]


def draw_bars(ax, picked, y_key):
    """A slot per dataset, a bar per layout. Missing values draw nothing."""
    slots = np.arange(len(g.DATASETS), dtype=float)
    bar_width = BAR_GROUP_WIDTH / len(VECS)

    values = []
    for i, vec in enumerate(VECS):
        heights = []
        for dataset in g.DATASETS:
            run = picked.get((dataset, vec))
            value = run.get(y_key) if run else None
            if not pu.is_number(value):
                pu.warn_once("bins_vec_db/%s vec%d: no %s", dataset, vec, y_key)
                heights.append(np.nan)
                continue
            pu.log.info("%-18s %-8s vec%d  %.4g  %s", y_key, dataset, vec, value, run['folder'])
            heights.append(value)
        values += [h for h in heights if pu.is_number(h)]
        offset = (i - (len(VECS) - 1) / 2) * bar_width
        color = VEC_COLORS[vec]
        drawn = ax.bar(slots + offset, heights, width=bar_width * BAR_FILL, color=color,
                       edgecolor=darker(color), linewidth=EDGE_WIDTH)
        if SHOW_VALUES:
            ax.bar_label(drawn, fmt='%.2f', padding=1, fontsize=TICK_SIZE)

    ax.grid(True, axis='y', which='major')
    ax.grid(False, axis='x')
    ax.set_axisbelow(True)   # grid lines behind the bars
    setup_y(ax, y_key, values, Y_LIMS.get(y_key))
    ax.set_xlim(-0.5, len(g.DATASETS) - 0.5)
    ax.set_xticks(slots)
    ax.set_xticklabels([g.DATASET_LABELS[d] for d in g.DATASETS], fontsize=DATASET_LABEL_SIZE)
    ax.tick_params(axis='x', length=0)   # no tick marks under the dataset names


def plot_quality_bars(nested_data):
    figure = FIGURES['quality']
    fig, axes = new_grid(QUALITY_KEYS, figure['cols'], figure['panel'], figure['headroom'])
    picked = {(dataset, vec): fixed_run(nested_data, dataset, vec)
              for dataset in g.DATASETS for vec in VECS}
    for ax, y_key in zip(axes, QUALITY_KEYS):
        draw_bars(ax, picked, y_key)

    handles = [Patch(facecolor=VEC_COLORS[v], edgecolor=darker(VEC_COLORS[v]),
                     linewidth=EDGE_WIDTH, label=VEC_LABELS[v]) for v in VECS]
    return finish(fig, figure, [handles])


# ==========================================
# THE LINES, BOTH DATASETS ON EACH PANEL
# ==========================================

def draw_lines(ax, nested_data, y_key):
    """One cost against docs per bin: colour = dataset, line style = layout."""
    xs_drawn, values = [], []
    for dataset in g.DATASETS:
        for vec in VECS:
            xs, ys = pu.xy(sweep_runs(nested_data, dataset, vec), X_KEY, y_key, ONLY_IMPROVING,
                           name=f'bins_vec_db/{y_key}/{dataset}/vec{vec}')
            if not xs:
                continue
            ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset],
                    linestyle=VEC_LINESTYLES[vec], markersize=MARKER_SIZE,
                    clip_on=False)   # a point on the end tick would otherwise be cut in half
            xs_drawn += xs
            values += ys

    ax.grid(True, which='major')
    setup_x(ax)
    setup_y(ax, y_key, values, Y_LIMS.get(y_key))
    clip_if_outside(ax, xs_drawn, values)


def plot_cost_lines(nested_data):
    figure = FIGURES['cost']
    fig, axes = new_grid(COST_KEYS, figure['cols'], figure['panel'], figure['headroom'])
    for ax, y_key in zip(axes, COST_KEYS):
        draw_lines(ax, nested_data, y_key)

    layouts = [Line2D([], [], color='black', linestyle=VEC_LINESTYLES[v], label=VEC_LABELS[v])
               for v in VECS]
    return finish(fig, figure, [dataset_handles(), layouts], x_label=X_LABEL)


# ==========================================
# THE LINES, A DATASET PER COLUMN
# ==========================================

def draw_split_panel(ax, nested_data, dataset, panel):
    """One dataset's panel: a line per layout (colour) and per metric of the row (style)."""
    keys = metrics(panel)
    xs_drawn, values = [], []
    for vec in VECS:
        runs = sweep_runs(nested_data, dataset, vec)
        for y_key in keys:
            xs, ys = pu.xy(runs, X_KEY, y_key, ONLY_IMPROVING,
                           name=f'bins_vec_db/{y_key}/{dataset}/vec{vec}')
            if not xs:
                continue
            ax.plot(xs, ys, color=VEC_COLORS[vec], marker=g.DATASET_MARKERS[dataset],
                    linestyle=METRIC_LINESTYLES.get(y_key, '-') if len(keys) > 1 else '-',
                    markersize=MARKER_SIZE, clip_on=False)
            xs_drawn += xs
            values += ys

    ax.grid(True, which='major')
    setup_x(ax)
    lim = SPLIT_Y_LIMS.get((dataset, panel), SPLIT_Y_LIMS.get(panel))
    setup_y(ax, panel, values, lim, label=dataset == DATASETS[0], name=f' on {dataset}')
    clip_if_outside(ax, xs_drawn, values)


def plot_split(nested_data, figure):
    keys = figure['keys']
    n_rows, n_cols = len(keys), len(DATASETS)
    fig = plt.figure(figsize=(PANEL_SIZE[0] * (n_cols + PANEL_GAP * (n_cols - 1)),
                              PANEL_SIZE[1] * (n_rows + ROW_GAP * (n_rows - 1))),
                     layout='constrained')
    fig.get_layout_engine().set(wspace=PANEL_GAP, hspace=ROW_GAP)
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    for r, panel in enumerate(keys):
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            draw_split_panel(ax, nested_data, dataset, panel)
            if r == 0:
                ax.set_title(g.DATASET_LABELS[dataset], fontsize=COL_TITLE_SIZE)
            if r < n_rows - 1:
                ax.tick_params(labelbottom=False)   # the same docs per bin axis as the row under it

    layouts = [Line2D([], [], color=VEC_COLORS[v], label=VEC_LABELS[v]) for v in VECS]
    styled = []
    for panel in keys:
        if isinstance(panel, tuple):
            styled += [k for k in panel if k not in styled]
    styles = [Line2D([], [], color='black', linestyle=METRIC_LINESTYLES.get(k, '-'),
                     label=METRIC_LABELS.get(k, g.label(k, K))) for k in styled]
    return finish(fig, figure, [layouts, styles], x_label=X_LABEL)


def make_plots(nested_data):
    for dataset in g.DATASETS:
        pu.log.info("bins vec DB on %s: bars at bs=%s dpb=%s, lines sweep dpb at bs=%s, k=%d",
                    dataset, g.BINS_FIXED_BS.get(dataset), FIXED_DPB,
                    g.BINS_FIXED_BS.get(dataset), K)
    return ([plot_quality_bars(nested_data), plot_cost_lines(nested_data)]
            + [plot_split(nested_data, figure) for figure in SPLIT_FIGURES])


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
