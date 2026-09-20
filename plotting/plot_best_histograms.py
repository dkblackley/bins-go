"""
Best config per method, bars grouped by dataset, one panel per entry in
PANELS, all in one grid PDF:

    figures/best_grid.pdf

For each (method, dataset) the BEST_N configs are picked by select_configs.py
(the same picks plot_metric_vs_latency.py draws). A method listed in PICK_BY
then uses the one pick that is best on that metric, in every panel. Any other
method shows whichever pick does best on *each* panel's metric, so its bars in
different panels can come from different runs. "Best" follows the metric's
'better' direction in g.METRICS (highest MRR, lowest latency, ...).
The chosen folder for every bar is logged, so

    python load_results.py <folder>

shows the full run behind it.
"""

import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.transforms import ScaledTranslation

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

# One dict per panel, left to right, top to bottom. 'key' is the run key; every
# other field is optional and falls back to PANEL_DEFAULTS (or UNIT_DEFAULTS
# when 'units' is set):
#   ylim   (lo, hi) in the units shown on the axis, or None to fit the data
#   step   gap between y ticks, or None for about N_TICKS auto-placed ticks
#   fmt    tick label format ('%.1f'), a function (value, pos) -> str, or None
#          for matplotlib's default
#   log    True for a log y axis (for values spanning several orders of magnitude)
#   units  'bytes' or 'seconds': rescale to the most readable unit in UNITS
#          (e.g. KB -> MB -> GB) and put it in the axis label
#   base   the unit the run key is stored in, e.g. 'KB' for comm_kb
#   label  axis label instead of g.label(key)
PANELS = [
    dict(key='mrr'),
    #dict(key='comm_kb', units='bytes', base='KB'),   # TotalByteSent per query
    # other ready-made panels, swap any of the above for these:
    # dict(key='comm_per_batch_kb', units='bytes', base='KB'),
    # dict(key='db_size_mb', units='bytes', base='MB'),
    dict(key='wan_time', units='seconds', base='s', ylim=(0, 350), step=350 / 4,
         fmt=lambda v, _: f'{round(v, -1):.0f}'),   # labels rounded to the nearest 10
    # dict(key='lan_time', units='seconds', base='s', log=True),
    # dict(key='recall'),
    dict(key='faithfulness', ylim=(0.35, 0.9), step=0.15),
    # both datasets share this panel, so the range covers SciFact (0.35-0.75)
    # and MS MARCO (0.5-0.9) together rather than either one on its own
    dict(key='answer_relevancy', ylim=(0.6, 1.0), step=0.1),
]
N_COLS = 2
K = g.K_MAIN

BEST_N = 5              # pick the bar from this many selected configs per method

PICK_BY = {'bins': 'wan_time', 'tree': 'wan_time', 'pacmann': 'wan_time'}
# PICK_BY = {}
# per method: the metric that picks ONE run out of its BEST_N, used for every
# panel (best MRR, lowest WAN latency, ...). A method left out or set to None
# instead shows its best run on each panel's own metric, so its bars can come
# from different runs in different panels
BINS_FILTER = {}        # e.g. {'vec': 1} to only use single-DB bins runs

BAR_GROUP_WIDTH = 0.7   # share of each dataset slot filled by its bars
EDGE_DARKEN = 0.6       # bar outline = fill colour with RGB scaled by this (0 = black, 1 = same)
SHOW_VALUES = False     # print each bar's height above it

GRID_FIG_SIZE = (2.5 * g.FIG_SIZE[0], 1.5 * g.FIG_SIZE[0])   # same footprint as the latency grid

PANEL_DEFAULTS = dict(ylim=(0.0, 0.7), step=0.7 / 4, fmt='%.1f', log=False,
                      units=None, base=None, label=None)
# 0.0 to 0.7 in 4 equal steps, labels rounded to 1 dp, right for MRR / Recall / RAGAS scores
UNIT_DEFAULTS = dict(ylim=None, step=None, fmt=None)
# costs have no natural range, so unit panels fit the data unless told otherwise
N_TICKS = 5             # roughly how many y ticks an auto-placed ('step': None) axis gets

UNITS = {               # smallest to largest, each as a multiple of the first
    'bytes':   {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4},
    'seconds': {'µs': 1e-6, 'ms': 1e-3, 's': 1, 'min': 60},
}

MISSING_VALUE = 2.0     # a method with no value gets a bar this many times the panel's
                        # height, so it runs off the top: clearly a placeholder

X_LABEL_SIZE = 11       # dataset names under each group of bars
Y_LABEL_SIZE = 17       # metric name on the left of each panel
Y_LABEL_PAD = 2        # gap between the metric name and the y tick labels, in points
Y_TICK_SIZE = 14        # y tick labels (the numbers)
X_LABEL_SHIFT = {'scifact': 2}   # nudge a dataset name right by this many points (negative = left)

RIGHT_COL_ON_RIGHT = True   # True puts the right column's y label and tick labels on the right edge
SHOW_ARROWS = True          # add the better-direction arrow to each y label, as in the latency grid

COL_SPACE = 0.01        # extra gap between columns, as a fraction of the figure width
COL_PAD = 0.01          # padding around each panel, in inches (constrained layout's w_pad)


def darker(color, factor=EDGE_DARKEN):
    """The same hue, darker: each RGB channel scaled by factor."""
    r, gr, b = mcolors.to_rgb(color)
    return (r * factor, gr * factor, b * factor)


def best_run(runs, y_key):
    """The run with the best y_key value, or None if none of them have one."""
    good = [r for r in runs if pu.is_number(r.get(y_key))]
    if not good:
        return None
    lower = g.METRICS.get(y_key, {}).get('better') == 'lower'
    return (min if lower else max)(good, key=lambda r: r[y_key])


def candidates(nested_data):
    """{(method, dataset): [the BEST_N selected runs]}, computed once for every panel."""
    picks = {}
    for method in g.METHOD_ORDER:
        fixed = BINS_FILTER if method == 'bins' else {}
        for dataset in g.DATASETS:
            runs = pu.get_runs(nested_data, method, dataset, k=K, **fixed)
            picks[method, dataset] = sc.select_configs(runs, n=BEST_N, name=f'{method}/{dataset}')
    return picks


def panel_spec(panel):
    """A PANELS entry with every unset field filled in from the defaults."""
    return {**PANEL_DEFAULTS, **(UNIT_DEFAULTS if panel.get('units') else {}), **panel}


def pick_unit(values, units, base):
    """
    (unit name, multiplier) that shows `values` (stored in `base`) most readably:
    the largest unit the biggest value still reaches at least 1 of, so
    2,500,000 KB comes out as 2.4 GB rather than 2441 MB.
    """
    table = UNITS[units]
    top = max((abs(v) * table[base] for v in values if pu.is_number(v)), default=0)
    name = next(iter(table))
    for candidate, size in table.items():
        if top >= size:
            name = candidate
    return name, table[base] / table[name]


def axis_label(spec):
    if spec['label']:
        return spec['label']
    text = g.label(spec['key'], K)
    if spec['units']:
        text = re.sub(r'\s*\([^)]*\)$', '', text)   # drop the stored unit, '(KB)'
    return text


def draw_panel(ax, picks, panel):
    """One metric: a slot per dataset, a bar per method. The legend is set by the caller."""
    spec = panel_spec(panel)
    y_key = spec['key']
    slots = np.arange(len(g.DATASETS))
    bar_width = BAR_GROUP_WIDTH / len(g.METHOD_ORDER)

    heights = {}   # method -> [value per dataset], NaN where missing
    for method in g.METHOD_ORDER:
        heights[method] = []
        for dataset in g.DATASETS:
            pick_key = PICK_BY.get(method) or y_key
            run = best_run(picks[method, dataset], pick_key)
            if run is None or not pu.is_number(run.get(y_key)):
                pu.warn_once("best_grid: %s/%s has no %s (run picked by best %s)",
                             method, dataset, y_key, pick_key)
                heights[method].append(np.nan)
                continue
            pu.log.info("%-16s %-8s %-8s %.4g  (best %s)  %s", y_key, method, dataset,
                        run[y_key], pick_key, run['folder'])
            heights[method].append(run[y_key])

    label = axis_label(spec)
    if spec['units']:
        unit, scale = pick_unit([v for hs in heights.values() for v in hs],
                                spec['units'], spec['base'])
        heights = {m: [v * scale for v in hs] for m, hs in heights.items()}
        label += f' ({unit})'

    def bars(method, xs, ys, **kwargs):
        color = g.METHOD_COLORS[method]
        return ax.bar(xs, ys, width=bar_width * 0.8, color=color,
                      edgecolor=darker(color), linewidth=1.0, **kwargs)

    offsets = {method: (i - (len(g.METHOD_ORDER) - 1) / 2) * bar_width
               for i, method in enumerate(g.METHOD_ORDER)}
    for method in g.METHOD_ORDER:
        drawn = bars(method, slots + offsets[method], heights[method],
                     label=g.METHOD_LABELS[method])   # NaN bars draw nothing
        if SHOW_VALUES:
            ax.bar_label(drawn, fmt='%.2f', padding=1, fontsize=g.LEGEND_SIZE)

    if spec['log']:
        ax.set_yscale('log')
        ax.minorticks_off()
    if spec['ylim']:
        ax.set_ylim(*spec['ylim'])
    else:
        ax.set_ylim(*ax.get_ylim())   # freeze the fitted range before the placeholders go in
    if spec['step']:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(spec['step']))
    elif not spec['log']:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=N_TICKS))
    if callable(spec['fmt']):
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(spec['fmt']))
    elif spec['fmt']:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(spec['fmt']))

    # placeholders for missing values, drawn after the limits are fixed so they
    # run off the top instead of stretching the axis
    top = ax.get_ylim()[1]
    for method in g.METHOD_ORDER:
        missing = [x for x, v in zip(slots + offsets[method], heights[method]) if not pu.is_number(v)]
        if missing:
            bars(method, missing, [top * MISSING_VALUE] * len(missing))

    ax.set_xticks(slots)
    ax.set_xticklabels([g.DATASET_LABELS[d] for d in g.DATASETS], fontsize=X_LABEL_SIZE + 4, y=-0.04)
    ax.tick_params(axis='x', length=0)   # no tick marks under the dataset names
    for tick, dataset in zip(ax.get_xticklabels(), g.DATASETS):
        shift = X_LABEL_SHIFT.get(dataset, 0)
        if shift:   # moves only the text; the bars and axes stay put
            tick.set_transform(tick.get_transform() + ScaledTranslation(
                shift / 72, 0, ax.figure.dpi_scale_trans))
    ax.grid(False, axis='x')
    ax.tick_params(axis='y', labelsize=Y_TICK_SIZE)
    text = ax.set_ylabel(label, fontsize=Y_LABEL_SIZE, y=0.35, labelpad=Y_LABEL_PAD)
    if SHOW_ARROWS:
        g.add_arrow(text, y_key, 'y')   # follows the label if it moves to the right edge


def plot_best_histograms(nested_data):
    n_rows = -(-len(PANELS) // N_COLS)   # ceiling division
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=GRID_FIG_SIZE,
                             squeeze=False, layout='constrained')
    fig.get_layout_engine().set(wspace=COL_SPACE, w_pad=COL_PAD)

    picks = candidates(nested_data)
    for ax, panel in zip(axes.flat, PANELS):
        draw_panel(ax, picks, panel)
    if RIGHT_COL_ON_RIGHT and N_COLS > 1:
        for ax in axes[:, -1]:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
            ax.spines['left'].set_visible(False)   # move the axis line too, not just the ticks
            ax.spines['right'].set_visible(True)
    for ax in axes.flat[len(PANELS):]:
        ax.set_visible(False)   # spare cell when PANELS doesn't fill the grid

    pu.method_legend(fig)   # styled and placed in globals (METHOD_LEGEND_*)

    return pu.save_figure(fig, 'best_grid')


def make_plots(nested_data):
    plot_best_histograms(nested_data)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())