"""
Small helpers shared by the plot files: picking runs, turning them into x/y
lists, the "only improving" filter, the dataset/line-style legend, and saving.
"""

import logging
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g

log = logging.getLogger('plot')
_warned = set()


def warn_once(message, *args):
    """Same warning is only printed once per session (the plots reuse the same runs a lot)."""
    text = message % args
    if text not in _warned:
        _warned.add(text)
        log.warning(text)


# ==========================================
# PICKING RUNS
# ==========================================

def get_runs(nested_data, method, dataset, k=None, **fixed):
    """
    Runs for one method/dataset, optionally only those with k == k and with
    every run[key] == value in `fixed`, e.g. get_runs(d, 'bins', 'msmarco', k=10, vec=1, dpb=1500).
    """
    runs = nested_data.get(method, {}).get(dataset, [])
    runs = [r for r in runs
            if (k is None or r['k'] == k)
            and all(r.get(key) == value for key, value in fixed.items())]
    if not runs:
        warn_once("no %s runs on %s with k=%s %s", method, dataset, k, fixed or '(no other filter)')
    return runs


def bins_sweep(nested_data, dataset, x_key, k, vec=None, bs=None, dpb=None):
    """
    Bins runs where only x_key ('bs' or 'dpb') changes. The other one is
    pinned to `bs`/`dpb` if given, else to g.BINS_FIXED_BS / g.BINS_FIXED_DPB.
    """
    if x_key == 'bs':
        fixed = {'dpb': dpb if dpb is not None else g.BINS_FIXED_DPB}
    else:
        fixed = {'bs': bs if bs is not None else g.BINS_FIXED_BS.get(dataset)}
    if vec is not None:
        fixed['vec'] = vec
    return get_runs(nested_data, 'bins', dataset, k=k, **fixed)


# ==========================================
# RUNS -> X/Y
# ==========================================

def is_number(value):
    return isinstance(value, (int, float)) and not math.isnan(value)


def keep_improving(runs, y_key):
    """
    Runs must already be sorted by x. Keeps a run only if y_key is strictly
    better than every run before it, using METRICS[y_key]['better'], so a
    slower config that does worse is dropped.
    """
    higher = g.METRICS.get(y_key, {}).get('better', 'higher') == 'higher'
    kept, best = [], None
    for run in runs:
        y = run[y_key]
        if best is None or (y > best if higher else y < best):
            kept.append(run)
            best = y
    return kept


def xy(runs, x_key, y_key, only_improving=False, name=''):
    """Sorted x and y lists. Runs missing either value are dropped (and logged)."""
    good = [r for r in runs if is_number(r.get(x_key)) and is_number(r.get(y_key))]
    if runs and len(good) < len(runs):
        warn_once("%s: %d of %d runs are missing %s or %s (NaN), dropped",
                  name, len(runs) - len(good), len(runs), x_key, y_key)
    good.sort(key=lambda r: r[x_key])
    if only_improving:
        before = len(good)
        good = keep_improving(good, y_key)
        log.debug("%s: kept %d of %d improving runs", name, len(good), before)
    return [r[x_key] for r in good], [r[y_key] for r in good]


# ==========================================
# FIGURE HELPERS
# ==========================================

def new_figure(size=g.FIG_SIZE):
    fig, ax = plt.subplots(figsize=size)
    ax.grid(True, which='major')
    ax.minorticks_off()   # log axes otherwise get unlabelled minor ticks
    return fig, ax


# ==========================================
# TICKS (the look of plot_bins_ablation_stacked: short labels, never 10^x)
# ==========================================

def short_number(value, _pos=None):
    """0 -> '0', 50 -> '50', 400 -> '0.4K', 6000 -> '6K', 200000 -> '0.2M', 1e6 -> '1M'.
    Under 1000 the K form would need two decimals ('0.05K'), so those stay plain."""
    if value == 0:
        return '0'
    if abs(value) >= 1e6:
        return f'{value / 1e6:g}M'
    if abs(value) < 1000:
        return f'{value:g}'
    return f'{value / 1e3:g}K'


def plain_number(value, _pos=None):
    """A log tick as a plain decimal ('0.01', '1', '250'), never '10^-2'. Values
    at or above 1000 fall back to short_number, so a byte axis reads '10K'."""
    if abs(value) >= 1000:
        return short_number(value)
    return f'{value:g}'


def log_axis(axis, ticks=None, formatter=short_number):
    """
    Turn one axis (ax.xaxis / ax.yaxis) into a log axis labelled in plain
    numbers. `ticks` fixes the labelled positions; None labels every power of
    10 (numticks is set explicitly because the default skips every other power
    on a short axis). The caller sets the scale with set_xscale / set_yscale.
    """
    if ticks is None:
        axis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    else:
        axis.set_major_locator(ticker.FixedLocator(list(ticks)))
    axis.set_major_formatter(ticker.FuncFormatter(formatter))
    axis.set_minor_locator(ticker.NullLocator())   # set_?scale turns minor ticks back on


# ==========================================
# LEGENDS AND LABELS UNDER A FIGURE
# ==========================================

def legend_below(fig, handles, gap=-6, ncol=None, style=None):
    """
    One legend centred under everything else in the figure (the look of
    plot_bins_ablation_stacked): all entries on one line unless `ncol` says
    otherwise. `gap` is the space between the figure and the legend in points,
    `style` extra keyword arguments on top of g.LEGEND_STYLE.
    """
    fig.draw_without_rendering()   # lay out the figure so the legend knows where the bottom is
    bottom = fig.get_tightbbox().y0 / fig.get_figheight()
    offset = gap / 72 / fig.get_figheight()   # points -> figure fraction
    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, bottom - offset),
                      ncol=ncol or len(handles), **{**g.LEGEND_STYLE, **(style or {})})


def legend_rows(fig, rows, gap=-6, line_gap=0, style=None):
    """
    legend_below with one centred line per entry of `rows` (each a list of
    handles), e.g. the datasets on one line and the line styles under them. One
    legend can't centre a short line, so each line is its own legend, placed
    under the one before. `gap` is the space under the figure and `line_gap`
    the space between two lines, in points; empty rows are skipped.
    """
    style = {**g.LEGEND_STYLE, **(style or {})}
    fig.draw_without_rendering()   # lay out the figure so the legend knows where the bottom is
    top = fig.get_tightbbox().y0 / fig.get_figheight() - gap / 72 / fig.get_figheight()
    legends = []
    for handles in rows:
        if not handles:
            continue
        legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, top),
                            ncol=len(handles), **style)
        legends.append(legend)
        fig.draw_without_rendering()
        top = (legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
               - line_gap / 72 / fig.get_figheight())
    return legends


def row_xlabel(fig, axes, text, gap=4, centre_on=None, **kwargs):
    """
    One x label under a row of panels, `gap` points below the lowest of them
    (its tick labels included), for a grid where every panel in a row shares an
    x axis but the rows do not. It is centred on `axes`, or on `centre_on`
    instead when the row is short and the label should still line up with the
    full grid above it. Returns the Text.
    """
    fig.draw_without_rendering()   # tick labels have to be laid out before they can be measured

    def boxes(of):
        return [ax.get_tightbbox().transformed(fig.transFigure.inverted()) for ax in of]

    across = boxes(axes if centre_on is None else centre_on)
    centre = (min(b.x0 for b in across) + max(b.x1 for b in across)) / 2
    bottom = min(b.y0 for b in boxes(axes)) - gap / 72 / fig.get_figheight()
    return fig.text(centre, bottom, text, ha='center', va='top', **kwargs)


def grid_size(panel, n_cols, n_rows, wspace, hspace):
    """
    figsize for a grid of `panel`-sized (inches) panels with matplotlib's
    wspace/hspace between them. Those are fractions of a panel, and
    subplots_adjust divides a FIXED figure between panels and gaps, so the gaps
    have to be added here or they eat the panels instead.
    """
    return (panel[0] * (n_cols + wspace * (n_cols - 1)),
            panel[1] * (n_rows + hspace * (n_rows - 1)))


def dataset_legend(ax, datasets, styles=(), **position):
    """
    Legend with one coloured entry per dataset plus one black entry per line
    style, for plots where colour = dataset and line style = something else.
    styles: [(label, linestyle), ...]
    """
    handles = [Line2D([], [], color=g.DATASET_COLORS[d], marker=g.DATASET_MARKERS[d],
                      label=g.DATASET_LABELS[d]) for d in datasets]
    handles += [Line2D([], [], color='black', linestyle=ls, label=text) for text, ls in styles]
    return ax.legend(handles=handles, **position, **g.LEGEND_STYLE)


def method_legend(fig):
    """
    The method legend, centred under everything else in the figure, one line
    per entry of g.METHOD_LEGEND_ROWS ('Bin  Tree' then 'PACMANN'). One legend
    can't centre a lone entry, so each line is its own legend, placed under the
    one before. Handles come from any axes that drew a method (by its
    g.METHOD_LABELS label); methods nobody drew are left out.
    """
    drawn = {}
    for ax in fig.axes:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            drawn.setdefault(text, handle)
    style = {**g.LEGEND_STYLE, **g.METHOD_LEGEND_STYLE}
    gap = g.METHOD_LEGEND_LINE_GAP / 72 / fig.get_figheight()   # points -> figure fraction

    fig.draw_without_rendering()   # lay out the figure so each line knows where to go
    top = fig.get_tightbbox().y0 / fig.get_figheight()   # bottom of the axes, labels, ...
    legends = []
    for row in g.METHOD_LEGEND_ROWS:
        methods = [m for m in row if g.METHOD_LABELS[m] in drawn]
        if not methods:
            continue
        legend = fig.legend([drawn[g.METHOD_LABELS[m]] for m in methods],
                            [g.METHOD_LEGEND_LABELS[m] for m in methods],
                            loc='upper center', bbox_to_anchor=(0.5, top - gap),
                            ncol=len(methods), **style)
        legends.append(legend)
        fig.draw_without_rendering()
        top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
    return legends


def save_figure(fig, name):
    os.makedirs(g.FIGURE_DIR, exist_ok=True)
    path = os.path.join(g.FIGURE_DIR, name + '.pdf')
    if not any(ax.has_data() for ax in fig.axes):
        log.warning("%s: nothing was plotted, saving an empty figure", name)
    fig.savefig(path)
    plt.close(fig)
    log.info("saved %s", path)
    return path
