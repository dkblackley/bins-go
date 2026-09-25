"""
The PILLAR-Bin ablation panels whose two datasets are too far apart to share an
axis (storage, preprocessing, quality), a dataset per column instead of a line
per dataset, so each gets its own y range. One figure per FIGURES entry and per
sweep, a row per entry of its 'keys', a column per dataset:

    figures/bins_storage_bs.pdf           Server / Client storage vs hash table size
    figures/bins_storage_dpb.pdf          Server / Client storage vs docs per bin
    figures/bins_preproc_quality_bs.pdf   Preproc. / Quality vs hash table size
    figures/bins_preproc_quality_dpb.pdf  Preproc. / Quality vs docs per bin

(SWEEPS_IN_ONE stacks the two sweeps into one figure per entry instead.) The
runs, colours, fonts and tick labels are those of plot_bins_ablation_grid.py,
which draws the costs that do share a range (WAN, LAN, Compute, Data Sent).

Every axis gets N_TICKS (y) / N_X_TICKS (x) ticks, the first and the last on
the ends of the axis, so every panel has a value in each corner. Limits can be
set per sweep, per panel and per dataset (Y_LIMS, X_LIMS).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_bins_ablation_grid as abl
import plot_pareto_grid as pg
import plot_utils as pu

# the same runs as plot_bins_ablation_grid.py, so the two figures agree
K = abl.K
# BINS_VEC = abl.BINS_VEC
BINS_VEC = 0
ONLY_IMPROVING = abl.ONLY_IMPROVING
FIXED_DPB = abl.FIXED_DPB   # docs per bin during the bin size sweep
MAX_DPB = 500        # runs with more docs per bin than this are left out (the ones past it
                     # broke); None keeps every run

# ==========================================
# WHAT GOES IN EACH FIGURE
# ==========================================

# A row is a run key, or a tuple of run keys sharing one axes (line style =
# metric, see METRIC_LINESTYLES). One figure per entry AND per sweep in SWEEPS:
# figures/<name>_<sweep>.pdf. '{sweep}' in the title becomes that sweep's
# X_LABELS name; None for no title.
FIGURES = [
    dict(name='bins_storage', title='Storage by {sweep}',
         keys=['db_size_mb', 'client_storage_mb']),
    dict(name='bins_preproc_quality', title='Preproc/Quality by {sweep}',
         keys=['maintenance_time', ('mrr', 'recall')]),
]

SKIP_TOP_ROW = True     # True leaves out each figure's first row (its first 'keys' entry), so
                         # Server and Preproc. go and Client and Quality stay; a figure can
                         # override it with skip_top_row=True/False in its FIGURES entry
SWEEPS = ['bs', 'dpb']   # a figure per sweep; ['dpb'] for the docs per bin figures only
SWEEPS_IN_ONE = False    # True: one figure per FIGURES entry, figures/<name>.pdf, with each
                         # sweep's rows stacked under the one before (a 4x2 for two keys)
DATASETS = g.DATASETS    # a column each, left to right

Y_LABELS = {             # anything not listed uses g.label(); the costs have no unit here,
    ('mrr', 'recall'): 'Quality',   # every tick carries its own (Y_FORMATS)
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

LOG_Y_KEYS = []   # e.g. ['db_size_mb', 'client_storage_mb']

Y_FORMATS = {        # y key -> (unit table, the unit the key is stored in): every tick
    'wan_time':          ('seconds', 's'),        # labelled in its own unit, '1.3ms', '2m'
    'lan_time':          ('seconds', 's'),
    'total_time':        ('seconds', 's'),
    'maintenance_time':  ('seconds', 's'),
    'comm_kb':           ('bytes', 'KB'),
    'db_size_mb':        ('bytes', 'MB'),
    'client_storage_mb': ('bytes', 'MB'),
}                    # anything else is a plain number of pg.TICK_SIG figures

# sweep -> {(dataset, row key) or row key -> (lo, hi)}, IN THE STORED UNIT (seconds
# for the times, MB for the storage, so 1 GB is (0, 1024) and 10 minutes (0, 600)).
# A (dataset, key) entry is that one panel; a bare key is every dataset's panel of
# that row. N_TICKS divides a pinned range evenly. A panel with no entry is fitted
# to its data, with round ticks (from 0 when Y_FROM_ZERO).
Y_LIMS = {
    'bs': {     # the hash table size figures
        ('msmarco', 'maintenance_time'):  (30, 45),
        ('scifact', 'maintenance_time'):  (0, 1.5),
        ('msmarco', 'db_size_mb'):        (0, 1024),
        ('scifact', 'db_size_mb'):        (0, 10),
        ('msmarco', 'client_storage_mb'): (0, 2540),
        ('scifact', 'client_storage_mb'): (0, 50),
        ('msmarco', ('mrr', 'recall')):   (0.0, 0.075),
        ('scifact', ('mrr', 'recall')):   (0.0, 0.6),
    },
    'dpb': {    # the docs per bin figures
        ('msmarco', 'maintenance_time'):  (0, 330),
        ('scifact', 'maintenance_time'):  (0, 120),
        ('msmarco', 'db_size_mb'):        (0, 5024),
        ('scifact', 'db_size_mb'):        (0, 10),
        ('msmarco', 'client_storage_mb'): (0, 15024),
        ('scifact', 'client_storage_mb'): (0, 120),
        ('msmarco', ('mrr', 'recall')):   (0.0, 0.3),
        ('scifact', ('mrr', 'recall')):   (0.25, 0.8),
    },
}
Y_FROM_ZERO = True   # a plain (not log) y axis with no Y_LIMS entry starts at 0 rather than
                     # at the lowest value, so the panels read as magnitudes
N_TICKS = abl.N_TICKS   # y ticks per panel, the first and the last on the ends of the axis

# ==========================================
# THE X AXES
# ==========================================

LOG_X_SWEEPS = ['bs']   # sweeps on a log x axis; the rest get a plain one
EVEN_X_TICKS = True   # every x axis gets N_X_TICKS ticks, evenly spaced (in log space on a log
                      # axis), the first and the last on the ends of the axis (like the y axes).
                      # False uses X_TICKS instead, where the ends of the axis may have no tick
N_X_TICKS = {'bs': 4, 'dpb': 4}   # per sweep, or one number for all; 3 on the plain dpb axis
                                  # gives round 0, 250, 500 (4 would give 167 and 333)
# sweep, or (sweep, dataset) for one column only -> (lo, hi). With EVEN_X_TICKS the ticks
# divide a pinned range; a range not listed is the data's, rounded out to pg.TICK_SIG figures
X_LIMS = {'dpb': (0, 500)}   # e.g. ('bs', 'scifact'): (100, 1e5)
# EVEN_X_TICKS off: sweep, or (sweep, dataset) -> tick positions; None = every power of 10
# on a log axis, matplotlib's own on a plain one
X_TICKS = {'bs': None, 'dpb': None}
X_LABELS = {'bs': 'Hash Table Size (log)', 'dpb': 'Docs per Bin'}   # else g.label()
X_LABEL_EACH = False   # False: one x label under the whole figure (all its rows share the
                       # sweep); True: one under every bottom panel. Always per panel when
                       # SWEEPS_IN_ONE puts two sweeps in one figure
SHARE_X = True         # a panel with the same sweep under it drops its x tick labels

# ==========================================
# THE FIGURE (sizes from plot_bins_ablation_grid.py, so the figures read alike)
# ==========================================

PANEL_SIZE = abl.PANEL_SIZE   # (width, height) in inches of one panel, before the gaps
PANEL_GAP = abl.PANEL_GAP     # space between two columns, as a fraction of a panel's width
ROW_GAP = abl.ROW_GAP         # space between two rows, as a fraction of a panel's height
HEADROOM = abl.HEADROOM       # inches added to the figure height for the titles

TITLE_SIZE = abl.TITLE_SIZE
COL_TITLES = True             # the dataset's name over each column
COL_TITLE_SIZE = pg.TITLE_SIZE
X_LABEL_SIZE = abl.X_LABEL_SIZE - 2
Y_LABEL_SIZE = abl.Y_LABEL_SIZE
LABEL_PAD = abl.LABEL_PAD
TICK_SIZE = abl.TICK_SIZE
Y_LABEL_EACH = False   # False: the metric name on the leftmost panel of a row only
Y_LABEL_X = abl.Y_LABEL_X   # x of every y label, as a fraction of its panel's width; None =
                            # off the widest tick label
Y_LABEL_Y = abl.Y_LABEL_Y   # height of a y label up its panel, 0 = bottom, 1 = top
SHOW_ARROWS = abl.SHOW_ARROWS   # the better-direction badge after each y label

MARKER_SIZE = abl.MARKER_SIZE
LEGEND_DATASETS = not COL_TITLES   # a coloured entry per dataset (the column titles already
                                   # say which is which); the metric line styles of a tuple
                                   # row are always listed
LEGEND_GAP = abl.LEGEND_GAP
LEGEND_SPACING = abl.LEGEND_SPACING


def metrics(panel):
    """The run keys one panel draws: itself, or each key of a tuple."""
    return list(panel) if isinstance(panel, tuple) else [panel]


def y_label(panel):
    default = ' / '.join(g.label(key, K) for key in metrics(panel))
    return Y_LABELS.get(panel, default).format(k=K)


def sweep_option(table, x_key, dataset):
    """table[(x_key, dataset)], else table[x_key], else None."""
    return table.get((x_key, dataset), table.get(x_key))


def setup_y(ax, x_key, dataset, panel, ys):
    """N_TICKS evenly spaced y ticks, the outer two on the ends of the axis."""
    key = metrics(panel)[0]
    log = key in LOG_Y_KEYS and ys and min(ys) > 0
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(ticker.NullLocator())   # set_yscale brings them back
    lims = Y_LIMS.get(x_key, {})
    lim = lims.get((dataset, panel), lims.get(panel))
    if not (lim or ys):
        return
    if lim and ys and (min(ys) < lim[0] or max(ys) > lim[1]):
        pu.warn_once("bins_ablation_split: %s on %s's %s sweep runs %.4g to %.4g, outside its "
                     "Y_LIMS range %s (in the stored unit), so part of it is off the panel",
                     panel, dataset, x_key, min(ys), max(ys), lim)
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


def setup_x(ax, x_key, dataset, xs):
    log = x_key in LOG_X_SWEEPS
    lim = sweep_option(X_LIMS, x_key, dataset)
    if log:
        ax.set_xscale('log')
    if not EVEN_X_TICKS:
        ticks = sweep_option(X_TICKS, x_key, dataset)
        if log:
            pu.log_axis(ax.xaxis, ticks)
        elif ticks is not None:
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(pu.short_number))
        if lim:
            ax.set_xlim(*lim)
        return
    xs = [x for x in xs if x > 0] if log else xs
    if not (lim or xs):
        return
    lo, hi = lim or (min(xs), max(xs))
    n = N_X_TICKS.get(x_key, 4) if isinstance(N_X_TICKS, dict) else N_X_TICKS
    ticks, (lo, hi) = pg.even_ticks(lo, hi, log, bool(lim), n=n)
    ax.set_xlim(lo, hi)
    # a pinned log range divides into uneven values (10 to 500 gives 36.8 and 136), so each
    # inner tick moves to the pg.TICK_SIG figures its label shows
    ticks = [ticks[0]] + [pg.round_sig(t) for t in ticks[1:-1]] + [ticks[-1]]
    formatter = lambda value, _pos=None: pu.short_number(float(pg.short(value)))
    if log:
        pu.log_axis(ax.xaxis, ticks, formatter=formatter)
    else:
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        ax.xaxis.set_minor_locator(ticker.NullLocator())


def draw_panel(ax, nested_data, x_key, dataset, panel):
    """One dataset's panel against one swept parameter: a line per metric of the row."""
    keys = metrics(panel)
    dpb = FIXED_DPB if x_key == 'bs' else None   # the other parameter is pinned in globals
    runs = pu.bins_sweep(nested_data, dataset, x_key, K, vec=BINS_VEC, dpb=dpb)
    if MAX_DPB is not None:
        runs = [r for r in runs if r['dpb'] <= MAX_DPB]
    xs_drawn, ys_drawn = [], []
    for y_key in keys:
        xs, ys = pu.xy(runs, x_key, y_key, ONLY_IMPROVING,
                       name=f'bins_ablation_split/{x_key}/{y_key}/{dataset}')
        if not xs:
            continue
        ax.plot(xs, ys, color=g.DATASET_COLORS[dataset], marker=g.DATASET_MARKERS[dataset],
                linestyle=METRIC_LINESTYLES.get(y_key, '-') if len(keys) > 1 else '-',
                markersize=MARKER_SIZE,
                clip_on=False)   # a point on the end tick would otherwise be cut in half
        xs_drawn += xs
        ys_drawn += ys

    ax.grid(True, which='major')
    ax.tick_params(labelsize=TICK_SIZE)
    setup_x(ax, x_key, dataset, xs_drawn)
    setup_y(ax, x_key, dataset, panel, ys_drawn)
    (x_lo, x_hi), (y_lo, y_hi) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    if any(not (x_lo <= x <= x_hi) for x in xs_drawn) or any(not (y_lo <= y <= y_hi) for y in ys_drawn):
        for line in ax.lines:   # past a pinned limit: unclipped, it would run off over the figure
            line.set_clip_on(True)


def label_y(ax, panel):
    keys = metrics(panel)
    text = ax.set_ylabel(y_label(panel), fontsize=Y_LABEL_SIZE, labelpad=LABEL_PAD)
    if Y_LABEL_X is not None:
        ax.yaxis.set_label_coords(Y_LABEL_X, Y_LABEL_Y)
    if SHOW_ARROWS and len({g.METRICS.get(k, {}).get('better') for k in keys}) == 1:
        g.add_arrow(text, keys[0], 'y')   # a tuple only when its metrics agree on 'better'


def sweep_label(x_key):
    return X_LABELS.get(x_key, g.label(x_key))


def legend_handles(panels):
    """A coloured entry per dataset (LEGEND_DATASETS), then a black line per metric style."""
    handles = [Line2D([], [], color=g.DATASET_COLORS[d], marker=g.DATASET_MARKERS[d],
                      markersize=MARKER_SIZE, label=g.DATASET_LABELS[d])
               for d in DATASETS] if LEGEND_DATASETS else []
    styled = []
    for panel in panels:
        if isinstance(panel, tuple):
            styled += [k for k in panel if k not in styled]
    handles += [Line2D([], [], color='black', linestyle=METRIC_LINESTYLES.get(k, '-'),
                       label=METRIC_LABELS.get(k, g.label(k, K))) for k in styled]
    return handles


def plot_grid(nested_data, rows, name, title):
    """rows: [(sweep, row key), ...] top to bottom, a column per dataset."""
    n_rows, n_cols = len(rows), len(DATASETS)
    fig = plt.figure(figsize=(PANEL_SIZE[0] * (n_cols + PANEL_GAP * (n_cols - 1)),
                              PANEL_SIZE[1] * (n_rows + ROW_GAP * (n_rows - 1)) + HEADROOM),
                     layout='constrained')
    fig.get_layout_engine().set(wspace=PANEL_GAP, hspace=ROW_GAP)
    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    sweeps = list(dict.fromkeys(x_key for x_key, _ in rows))
    for r, (x_key, panel) in enumerate(rows):
        last_of_sweep = r == n_rows - 1 or rows[r + 1][0] != x_key
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            draw_panel(ax, nested_data, x_key, dataset, panel)
            if c == 0 or Y_LABEL_EACH:
                label_y(ax, panel)
            if r == 0 and COL_TITLES:
                ax.set_title(g.DATASET_LABELS[dataset], fontsize=COL_TITLE_SIZE)
            if SHARE_X and not last_of_sweep:
                ax.tick_params(labelbottom=False)
            if last_of_sweep and (X_LABEL_EACH or len(sweeps) > 1):
                ax.set_xlabel(sweep_label(x_key), fontsize=X_LABEL_SIZE, labelpad=LABEL_PAD)

    names = [sweep_label(x_key).replace(' (log)', '') for x_key in sweeps]
    if len(sweeps) == 1 and not X_LABEL_EACH:
        fig.supxlabel(sweep_label(sweeps[0]), fontsize=X_LABEL_SIZE)
    if title:
        fig.suptitle(title.format(sweep=' and '.join(names)), fontsize=TITLE_SIZE)
    handles = legend_handles([panel for _, panel in rows])
    if handles:
        pu.legend_below(fig, handles, gap=LEGEND_GAP, style=LEGEND_SPACING)

    return pu.save_figure(fig, name)


def make_plots(nested_data):
    for dataset in DATASETS:
        pu.log.info("bins ablation split on %s: dpb=%s for the bs sweep, bs=%s for the dpb sweep, "
                    "k=%d, vec=%s", dataset, FIXED_DPB, g.BINS_FIXED_BS.get(dataset), K, BINS_VEC)
    paths = []
    for figure in FIGURES:
        keys = figure['keys'][1:] if figure.get('skip_top_row', SKIP_TOP_ROW) else figure['keys']
        if not keys:
            pu.warn_once("bins_ablation_split: %s has no rows left once its top one is "
                         "skipped, not drawn", figure['name'])
            continue
        if SWEEPS_IN_ONE:
            rows = [(x_key, panel) for x_key in SWEEPS for panel in keys]
            paths.append(plot_grid(nested_data, rows, figure['name'], figure.get('title')))
            continue
        for x_key in SWEEPS:
            rows = [(x_key, panel) for panel in keys]
            paths.append(plot_grid(nested_data, rows, f"{figure['name']}_{x_key}",
                                   figure.get('title')))
    return paths


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
