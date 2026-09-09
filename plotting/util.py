"""
Global aesthetics for the paper figures. Tweak here, every plot module follows.

Sizing model
------------
Sizes are given in *final* points, i.e. what lands on the compiled page, then
multiplied by SCALE. Figures are drawn SCALE times too big and shrunk back by

    \\includegraphics[width=\\textwidth]{fig.pdf}

so FONT_PT is literally the point size in the paper. Set TEXT_WIDTH_IN to your
\\textwidth (\\the\\textwidth in the log) and nothing else needs touching.
Do not wrap the result in \\resizebox: that rescales the fonts a second time.

FONT_SIZE is kept as an alias for the matplotlib-facing number, so the
`FONT_SIZE + 10` idiom from the other project still works.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================
# THE KNOBS
# ==========================================

TEXT_WIDTH_IN = 5.5   # \textwidth of the paper (5.5in = ICLR one-column)
SCALE = 4.0           # draw-large-then-shrink factor; invisible in the output
FONT_PT = 16.0         # tick/legend size in the compiled paper

FONT_SIZE = FONT_PT * SCALE   # matplotlib-facing size

# ==========================================
# PALETTE
# ==========================================

# Okabe-Ito, colour-blind safe. Hatches give a second, greyscale-safe channel.
COLORS = {
    'pacmann': '#0072B2',  # Blue
    'bins':    '#D55E00',  # Vermillion
    'tree':    '#CC79A7',  # Reddish purple
}

HATCHES = {
    'pacmann': '',
    'bins':    '///',
    'tree':    'xxx',
}

MARKERS = {
    'pacmann': 'o',
    'bins':    's',
    'tree':    '^',
}

LABELS = {
    'pacmann': 'PACMANN',
    'bins':    'BM25-Bin',
    'tree':    'BM25-Tree',
}

METHOD_ORDER = ['pacmann', 'bins', 'tree']

# Datasets. 'short' is used where a panel is ~1in wide and the full name
# collides with its neighbour.
DATASETS = [
    {'key': 'scifact', 'label': 'SciFact',  'short': 'SciFact'},
    {'key': 'msmarco', 'label': 'MS MARCO', 'short': 'MS\nMARCO'},
]

DATASET_COLORS = {
    'msmarco': '#0072B2',
    'scifact': '#D55E00',
}

DATASET_MARKERS = {
    'msmarco': 'o',
    'scifact': 's',
}

# Sequential-ish but still distinguishable, for the round breakdown.
ROUND_COLORS = ['#0072B2', '#56B4E9', '#E69F00']
ROUND_HATCHES = ['', '///', 'xxx']


def setup_sleek_style(font_pt=FONT_PT, scale=SCALE):
    """Flat, high-contrast, large-type. Call once from main.py."""
    plt.rcParams.update({
        'font.size':          font_pt * scale,
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'Nimbus Roman',
                               'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset':   'stix',
        'axes.titlesize':     (font_pt + 1.0) * scale,
        'axes.labelsize':     (font_pt - 0.5) * scale,
        'xtick.labelsize':    (font_pt - 2.5) * scale,
        'ytick.labelsize':    (font_pt - 2.5) * scale,
        'legend.fontsize':    (font_pt + 0.5) * scale,
        'axes.linewidth':     0.7 * scale,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.titlepad':      3.0 * scale,
        'axes.labelpad':      1.5 * scale,
        'xtick.major.size':   2.0 * scale,
        'ytick.major.size':   2.0 * scale,
        'xtick.major.width':  0.7 * scale,
        'ytick.major.width':  0.7 * scale,
        'xtick.major.pad':    1.5 * scale,
        'ytick.major.pad':    1.5 * scale,
        'lines.linewidth':    1.5 * scale,
        'lines.markersize':   3.4 * scale,
        'lines.markeredgewidth': 0.0,
        'grid.color':         '#CCCCCC',
        'grid.linestyle':     '-',
        'grid.linewidth':     0.5 * scale,
        'grid.alpha':         0.9,
        'hatch.linewidth':    0.6 * scale,
        'figure.autolayout':  False,   # manual subplots_adjust everywhere
        'pdf.fonttype':       42,      # embed TrueType, text stays selectable
        'ps.fonttype':        42,
    })


# ==========================================
# SHARED HELPERS
# ==========================================

def new_figure(nrows, ncols, height_in, width_in=None, scale=SCALE):
    """Figure whose on-page size is (width_in x height_in) after shrinking."""
    width_in = TEXT_WIDTH_IN if width_in is None else width_in
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_in * scale, height_in * scale))
    fig.set_layout_engine(None)
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]
    return fig, axes


def style_axes(ax, title=None, ylim=None, nyticks=3, yfmt='{:g}',
               xlabel=None, ylabel=None, ygrid=True, xgrid=False):
    """Uniform panel furniture: 3 y-ticks, light y-grid, no top/right spines."""
    import numpy as np
    if ylim is not None:
        ax.set_ylim(*ylim)
    lo, hi = ax.get_ylim()
    ax.set_yticks(np.linspace(lo, hi, nyticks))
    ax.set_yticklabels([yfmt.format(t) for t in np.linspace(lo, hi, nyticks)])
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.grid(ygrid)
    ax.xaxis.grid(xgrid)
    ax.set_axisbelow(True)
    return ax


def log_xticks(ax, values, labels=None, base=10, xlim=None, margin=0.12):
    """
    Log x-axis showing exactly the ticks asked for, no minor tick clutter.
    Pass `xlim` (all sampled points) when the ticks are a subset, so the limits
    still frame the data.
    """
    values = [float(v) for v in values]
    span = [float(v) for v in (xlim if xlim is not None else values)]
    lo, hi = min(span), max(span)

    ax.set_xscale('log', base=base)
    if lo > 0:
        pad = (hi / lo) ** margin
        ax.set_xlim(lo / pad, hi * pad)

    ax.set_xticks(values)
    if labels is not None and len(labels) != len(values):
        print(f"[WARN] {len(labels)} tick labels for {len(values)} ticks; "
              f"falling back to the values themselves")
        labels = None
    ax.set_xticklabels(labels if labels is not None
                       else [f'{v:g}' for v in values])
    ax.minorticks_off()
    ax.tick_params(axis='x', which='minor', length=0)

def patch_handles(keys, colors=COLORS, hatches=HATCHES, labels=LABELS,
                  scale=SCALE):
    """Legend swatches for bar charts."""
    return [Patch(facecolor=colors[k], edgecolor='#333333',
                  hatch=hatches.get(k, ''), linewidth=0.55 * scale,
                  label=labels.get(k, k)) for k in keys]


def shared_legend(fig, handles, ncol, y=1.0, labels=None):
    """One legend in the strip above the panels: costs one text line, not a row."""
    return fig.legend(
        handles=handles, labels=labels, loc='upper center',
        bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
        handlelength=1.2, handleheight=1.0, handletextpad=0.45,
        columnspacing=1.8, borderaxespad=0.0,
    )


def save(fig, output_dir, filename, scale=SCALE):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format='pdf', bbox_inches='tight',
                pad_inches=0.01 * scale)
    plt.close(fig)
    print(f"Saved {path}")
    return path

# (metric key in run_info, panel title, short title for narrow panels, ylim)
METRICS = [
    {'key': 'mrr',              'title': 'MRR@10',
     'short': 'MRR@10',       'ylim': (0.0, 0.70)},
    {'key': 'faithfulness',     'title': 'Faithfulness',
     'short': 'Faithfulness', 'ylim': (0.0, 0.85)},
    {'key': 'answer_relevancy', 'title': 'Relevancy',
     'short': 'Relevancy',    'ylim': (0.60, 0.85)},
    {'key': 'comm_cost',        'title': 'Comm. Cost (MB)',
     'short': 'Comm. (MB)',   'ylim': (0.0, 0.65)},
]

# Relevancy keeps the non-zero baseline of the original figure: all three
# methods sit in [0.74, 0.81], so a zero baseline flattens the panel into three
# identical bars. Change the ylim above to (0.0, 0.85) if a reviewer objects.

GROUP_FRAC = 0.86   # share of the unit slot occupied by one group of bars

