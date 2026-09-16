"""
Small helpers shared by the plot files: picking runs, turning them into x/y
lists, the "only improving" filter, the dataset/line-style legend, and saving.
"""

import logging
import math
import os

import matplotlib.pyplot as plt
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


def save_figure(fig, name):
    os.makedirs(g.FIGURE_DIR, exist_ok=True)
    path = os.path.join(g.FIGURE_DIR, name + '.pdf')
    if not any(ax.has_data() for ax in fig.axes):
        log.warning("%s: nothing was plotted, saving an empty figure", name)
    fig.savefig(path)
    plt.close(fig)
    log.info("saved %s", path)
    return path
