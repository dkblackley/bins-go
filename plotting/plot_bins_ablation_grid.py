"""
The PILLAR-Bin ablation costs whose two datasets share a range (WAN, LAN,
Compute, Data Sent), a 2x2 figure per sweep, in the look of plot_pareto_grid.py
(its panel size, gaps, font sizes and evenly spaced ticks) with the legend of
plot_bins_ablation_stacked.py:

    figures/bins_ablation_bs.pdf    ABLATION_KEYS vs hash table size
    figures/bins_ablation_dpb.pdf   ABLATION_KEYS vs docs per bin

The storage, preprocessing and quality panels, where the datasets are too far
apart to share an axis, are drawn a dataset per panel by plot_bins_ablation_split.py.

The bin size sweep has docs per bin fixed at FIXED_DPB, the docs per bin sweep
has bin size fixed at g.BINS_FIXED_BS. Colour = dataset. A panel is one metric,
or a tuple of them drawn on the same axes, told apart by line style
(METRIC_LINESTYLES) the way plot_bins_ablation_stacked.py draws MRR and Recall;
the legend then gains a black line per style.

Every y axis gets N_TICKS evenly spaced ticks with the first and the last on
the ends of the axis (plot_pareto_grid.even_ticks), so no panel has a bare
stretch above its top tick, and the costs carry their unit in each tick
('1.3ms', '0.4MB') like plot_pareto_grid.py does.

This replaces plot_bins_ablation.py, plot_bins_comm_vs_bs.py, plot_db_size.py
and plot_bins_client_storage.py, which drew one small PDF per (sweep, metric).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_pareto_grid as pg
import plot_utils as pu

K = g.K_ABLATION
BINS_VEC = 0         # 1 = single DB, 0 = split DBs (None would mix the two, don't)
ONLY_IMPROVING = False
FIXED_DPB = 10       # docs per bin during the bin size sweep (g.BINS_FIXED_DPB for the 1500 runs)
MAX_DPB = 500        # runs with more docs per bin than this are left out (the ones past it
                     # broke); None keeps every run

# ==========================================
# WHAT GOES IN EACH FIGURE
# ==========================================

# A panel is a run key, or a tuple of run keys sharing one axes (line style =
# metric, see METRIC_LINESTYLES).
ABLATION_KEYS = ['wan_time', 'lan_time', 'total_time', 'comm_kb']
# 'db_size_mb', 'client_storage_mb', 'maintenance_time' and ('mrr', 'recall') (MRR and
# Recall on one panel) still work here, but plot_bins_ablation_split.py draws them

# One figure per entry AND per sweep in SWEEPS: figures/<name>_<sweep>.pdf.
# 'cols' is panels per row (None = all on one row); '{sweep}' in the title
# becomes that sweep's X_LABELS name.
FIGURES = [
    dict(name='bins_ablation', title='Ablation: {sweep}',
         keys=ABLATION_KEYS, cols=2),
]

SWEEPS = ['bs', 'dpb']   # a figure per sweep; ['dpb'] for the docs per bin figures only

# Shorter than g.label() where the panel is narrow; anything not listed uses g.label().
# The costs have no unit here: every tick carries its own (Y_FORMATS).
Y_LABELS = {
    ('mrr', 'recall'): 'Quality',
    'mrr': 'MRR@{k}', 'recall': 'Recall@{k}', 'answer_relevancy': 'Relevancy',
    'faithfulness': 'Faithful.', 'wan_time': 'WAN', 'lan_time': 'LAN',
    'total_time': 'Compute', 'comm_kb': 'Data Sent', 'db_size_mb': 'Server',
    'client_storage_mb': 'Client', 'maintenance_time': 'Preproc.',
}

METRIC_LINESTYLES = {'mrr': '--', 'recall': ':'}   # in a panel of several metrics; a
                                                   # panel of one is always solid
METRIC_LABELS = {'mrr': 'MRR', 'recall': 'Recall'}   # their legend entries

# ==========================================
# THE Y AXES
# ==========================================

# LOG_Y_KEYS = ['db_size_mb',
#                'client_storage_mb']
LOG_Y_KEYS = []

Y_FORMATS = {        # y key -> (unit table, the unit the key is stored in): every tick
    'wan_time':          ('seconds', 's'),        # labelled in its own unit, '1.3ms', '2m'
    'lan_time':          ('seconds', 's'),
    'total_time':        ('seconds', 's'),
    'maintenance_time':  ('seconds', 's'),
    'comm_kb':           ('bytes', 'KB'),
    'db_size_mb':        ('bytes', 'MB'),
    'client_storage_mb': ('bytes', 'MB'),
}                    # anything else is a plain number of pg.TICK_SIG figures
# Per sweep, because the two sweeps put the same metric on very different ranges:
# sweep -> {panel key -> (lo, hi)}, IN THE STORED UNIT (seconds for the times, KB
# for comm_kb, MB for db_size_mb and client_storage_mb, so 1 GB of storage is
# (0, 1024) and 10 minutes of preprocessing (0, 600)). N_TICKS still divides a
# pinned range evenly. A key with no entry is fitted to the data, with round
# ticks (from 0 when Y_FROM_ZERO).
Y_LIMS = {
    'bs': {     # the hash table size figure
        'lan_time':   (0.0, 0.04),
        'wan_time':   (0.0, 0.4),
        'total_time': (0.0, 0.008),
        'comm_kb':    (70, 130),
        # 'maintenance_time':  (0, 600),
        # 'db_size_mb':        (0, 1024),
        # 'client_storage_mb': (0, 1024),
        # ('mrr', 'recall'):   (0.0, 0.9),
    },
    'dpb': {    # the docs per bin figure
        'lan_time':   (0.0, 0.275),
        'wan_time':   (0.0, 1.25),
        'total_time': (0.0, 0.125),
        'comm_kb':    (0, 25000),
        'maintenance_time':  (0, 900),
        'db_size_mb':        (0, 15024),
        'client_storage_mb': (0, 15024),
        ('mrr', 'recall'):   (0.0, 0.8),
    },
}
Y_FROM_ZERO = True   # a plain (not log) y axis with no Y_LIMS entry starts at 0 rather than
                     # at the lowest value, so the panels read as magnitudes
N_TICKS = pg.N_TICKS   # y ticks per panel, the first and the last on the ends of the axis

# ==========================================
# THE X AXES
# ==========================================

# x ticks per sweep: None labels every power of 10, a list fixes the positions.
# dpb only spans ~2.5 powers of 10, so a tick per power would give just three.
LOG_X_SWEEPS = ['bs']   # sweeps on a log x axis; the rest get a plain one
# x ticks per sweep: a list fixes the positions; None labels every power of 10 on a log
# axis, matplotlib's own on a plain one
X_TICKS = {'bs': None, 'dpb': [0, 250, 500]}   # more run into each other on these narrow panels
X_LIMS = {'dpb': (0, 500)}   # None/missing = fit the data
X_LABELS = {'bs': 'Hash Table Size (log)', 'dpb': 'Docs per Bin'}   # else g.label()
X_LABEL_EACH = False   # False: one x label under the whole figure (every panel shares the
                       # sweep); True: one under every panel, as plot_pareto_grid.py does

# ==========================================
# THE FIGURE (sizes borrowed from plot_pareto_grid.py, so the two read alike;
# put numbers in place of the pg.* ones to tune this file on its own)
# ==========================================

PANEL_SIZE = pg.PANEL_SIZE   # (width, height) in inches of one panel, before the gaps
PANEL_GAP = pg.PANEL_GAP     # space between two columns, as a fraction of a panel's width
ROW_GAP = pg.ROW_GAP         # space between two rows, as a fraction of a panel's height
HEADROOM = 0.0               # inches added to the figure height for the super title and the
                             # shared x label, which the pareto grid's rows don't have to fit

TITLE_SIZE = pg.SUPTITLE_SIZE
X_LABEL_SIZE = pg.X_LABEL_SIZE
Y_LABEL_SIZE = pg.Y_LABEL_SIZE
LABEL_PAD = pg.LABEL_PAD
TICK_SIZE = pg.TICK_SIZE
Y_LABEL_X = None   # x of every y label, as a fraction of its panel's width (pg.Y_LABEL_X is
                   # -0.3), so they all sit at the same place; None (as here) = off the widest
                   # tick label, since '180ms' / '1000MB' are wider than the pareto grid's
Y_LABEL_Y = pg.Y_LABEL_Y   # height of a y label up its panel, 0 = bottom, 1 = top
SHOW_ARROWS = pg.SHOW_ARROWS   # the better-direction badge after each y label

MARKER_SIZE = g.MARKER_SIZE - 1.5   # the points sit close together
LEGEND_GAP = 2   # gap between the figure and the legend under it, in points
# tighter than g.LEGEND_STYLE: labelspacing = gap between rows, columnspacing = between
# columns, handletextpad = between a line and its text (all in font-size units)
LEGEND_SPACING = dict(labelspacing=0.1, columnspacing=0.8, handletextpad=0.3,
                      handlelength=1.6,   # long enough to show a dashed / dotted line
                      fontsize=pg.LEGEND_STYLE.get('fontsize', g.LEGEND_SIZE))


def metrics(panel):
    """The run keys one panel draws: itself, or each key of a tuple."""
    return list(panel) if isinstance(panel, tuple) else [panel]


def y_label(panel):
    default = ' / '.join(g.label(key, K) for key in metrics(panel))
    return Y_LABELS.get(panel, default).format(k=K)


def setup_y(ax, x_key, panel, ys):
    """N_TICKS evenly spaced y ticks, the outer two on the ends of the axis."""
    key = metrics(panel)[0]
    log = key in LOG_Y_KEYS and ys and min(ys) > 0
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(ticker.NullLocator())   # set_yscale brings them back
    lim = Y_LIMS.get(x_key, {}).get(panel)
    if not (lim or ys):
        return
    if lim and ys and (min(ys) < lim[0] or max(ys) > lim[1]):
        pu.warn_once("bins_ablation_grid: %s on the %s sweep runs %.4g to %.4g, outside its "
                     "Y_LIMS['%s'] range %s (in the stored unit), so part of it is off the panel",
                     panel, x_key, min(ys), max(ys), x_key, lim)
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


def setup_x(ax, x_key):
    if x_key in LOG_X_SWEEPS:
        ax.set_xscale('log')
        pu.log_axis(ax.xaxis, X_TICKS.get(x_key))
    elif X_TICKS.get(x_key) is not None:
        ax.xaxis.set_major_locator(ticker.FixedLocator(X_TICKS[x_key]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pu.short_number))
    if X_LIMS.get(x_key):
        ax.set_xlim(*X_LIMS[x_key])


def draw_panel(ax, nested_data, x_key, panel):
    """One panel against one swept parameter: a line per dataset (and per metric, for a tuple)."""
    keys = metrics(panel)
    ys_drawn = []
    for dataset in g.DATASETS:
        dpb = FIXED_DPB if x_key == 'bs' else None   # the other parameter is pinned in globals
        runs = pu.bins_sweep(nested_data, dataset, x_key, K, vec=BINS_VEC, dpb=dpb)
        if MAX_DPB is not None:
            runs = [r for r in runs if r['dpb'] <= MAX_DPB]
        for y_key in keys:
            xs, ys = pu.xy(runs, x_key, y_key, ONLY_IMPROVING,
                           name=f'bins_ablation_grid/{x_key}/{y_key}/{dataset}')
            if not xs:
                continue
            ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset],
                    linestyle=METRIC_LINESTYLES.get(y_key, '-') if len(keys) > 1 else '-',
                    markersize=MARKER_SIZE,
                    clip_on=False)   # a point on the end tick would otherwise be cut in half
            ys_drawn += ys

    ax.grid(True, which='major')
    ax.tick_params(labelsize=TICK_SIZE)
    setup_x(ax, x_key)
    setup_y(ax, x_key, panel, ys_drawn)

    text = ax.set_ylabel(y_label(panel), fontsize=Y_LABEL_SIZE, labelpad=LABEL_PAD)
    if Y_LABEL_X is not None:
        ax.yaxis.set_label_coords(Y_LABEL_X, Y_LABEL_Y)
    if SHOW_ARROWS and len({g.METRICS.get(k, {}).get('better') for k in keys}) == 1:
        g.add_arrow(text, keys[0], 'y')   # a tuple only when its metrics agree on 'better'
    if X_LABEL_EACH:
        ax.set_xlabel(X_LABELS.get(x_key, g.label(x_key)), fontsize=X_LABEL_SIZE,
                      labelpad=LABEL_PAD)


def legend_handles(keys):
    """A coloured entry per dataset, then a black line per style when a panel mixes metrics."""
    handles = [Line2D([], [], color=g.DATASET_COLORS[d], marker=g.DATASET_MARKERS[d],
                      markersize=MARKER_SIZE, label=g.DATASET_LABELS[d]) for d in g.DATASETS]
    styled = []
    for panel in keys:
        if isinstance(panel, tuple):
            styled += [k for k in panel if k not in styled]
    handles += [Line2D([], [], color='black', linestyle=METRIC_LINESTYLES.get(k, '-'),
                       label=METRIC_LABELS.get(k, g.label(k, K))) for k in styled]
    return handles


def plot_grid(nested_data, x_key, name, title, keys, cols):
    n_cols = min(cols or len(keys), len(keys))
    n_rows = -(-len(keys) // n_cols)
    fig = plt.figure(figsize=(PANEL_SIZE[0] * (n_cols + PANEL_GAP * (n_cols - 1) + 0.25),
                              PANEL_SIZE[1] * (n_rows + ROW_GAP * (n_rows - 1)) + HEADROOM),
                     layout='constrained')
    fig.get_layout_engine().set(wspace=PANEL_GAP, hspace=ROW_GAP)
    axes = list(fig.subplots(n_rows, n_cols, squeeze=False).flat)

    for ax, panel in zip(axes, keys):
        draw_panel(ax, nested_data, x_key, panel)
    for ax in axes[len(keys):]:
        ax.set_axis_off()   # spare cell on a short last row

    sweep = X_LABELS.get(x_key, g.label(x_key))
    if not X_LABEL_EACH:
        fig.supxlabel(sweep, fontsize=X_LABEL_SIZE)
    if title:
        fig.suptitle(title.format(sweep=sweep.replace(' (log)', '')), fontsize=TITLE_SIZE)
    pu.legend_below(fig, legend_handles(keys), gap=LEGEND_GAP, style=LEGEND_SPACING)

    return pu.save_figure(fig, f'{name}_{x_key}')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        pu.log.info("bins ablation grid on %s: dpb=%s for the bs sweep, bs=%s for the dpb sweep, "
                    "k=%d, vec=%s", dataset, FIXED_DPB, g.BINS_FIXED_BS.get(dataset), K, BINS_VEC)
    return [plot_grid(nested_data, x_key, **figure) for figure in FIGURES for x_key in SWEEPS]


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
