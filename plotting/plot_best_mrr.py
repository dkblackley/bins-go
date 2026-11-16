"""
Best MRR@10 of every private method on each dataset, as horizontal bars, with the non-private references
(BM25, exact dense, fused hybrid) as dotted vertical lines:

    figures/best_mrr_<dataset>.pdf     one per dataset in DATASETS
    figures/best_mrr_grid.pdf          every dataset in one figure (GRID = True)

The numbers are rebuttal_final.best_scores(), i.e. the "Best MRR@10" table of figures/rebuttal_final.md, and every
value drawn is printed (PRINT_VALUES) so it can be pasted next to the figure.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_utils as pu
import rebuttal_final as rf
import frozen_values as fv

DATASETS = g.REBUTTAL_DATASETS
METHODS = ['bins', 'tree', 'pacmann', 'tiptoe', 'coeus']   # top to bottom
X_KEY = 'mrr'

# one dotted line per non-private reference: key in baselines.json -> style (label comes from rebuttal_final.NONPRIVATE)
BASELINE_STYLES = {
    'BM25':      dict(color='#555555', linestyle=':', linewidth=1.6),
    'Dense':     dict(color='#882255', linestyle=':', linewidth=1.6),
    'HybridRRF': dict(color='#117733', linestyle=':', linewidth=1.6),
}

FIG_SIZE = (3.2, 2.0)          # one dataset
GRID = True                    # also every dataset in one figure
GRID_COLS = 3
GRID_PANEL = (2.8, 1.9)        # (width, height) of one grid panel, inches
XLIM = None                    # (lo, hi) for every panel, or None: 0 .. XLIM_PAD x the largest value (bars and lines)
XLIM_PAD = 1.15
BAR_HEIGHT = 0.7
EDGE_DARKEN = 0.6
SHOW_VALUES = True             # the value at the end of each bar
VALUE_SIZE = 8
NA_TEXT = 'n/a'                # written where a method has no runs
TITLE = '{dataset}'            # per figure / panel, None for none
TITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 10
LEGEND_SIZE = 10
PRINT_VALUES = True


def _darker(color):
    import matplotlib.colors as mc
    r, gr, b = mc.to_rgb(color)
    return (r * EDGE_DARKEN, gr * EDGE_DARKEN, b * EDGE_DARKEN)


def draw(ax, data, dataset, show_ylabels=True):
    best, nonpriv = data['best'], data['nonprivate']
    ys = list(range(len(METHODS)))[::-1]   # first method at the top
    vals = [best[(m, dataset)][X_KEY] if best.get((m, dataset)) else None for m in METHODS]
    lines = {k: nonpriv.get((k, dataset)) for k, _ in data['nonprivate_labels']}
    top = max([v for v in vals if v is not None] + [v for v in lines.values() if v is not None] + [0.01])
    lo, hi = XLIM or (0, top * XLIM_PAD)
    for y, m, v in zip(ys, METHODS, vals):
        if v is None:
            ax.text(lo + (hi - lo) * 0.01, y, NA_TEXT, va='center', ha='left', fontsize=VALUE_SIZE, color='#666666')
            continue
        c = g.METHOD_COLORS[m]
        ax.barh(y, v, height=BAR_HEIGHT, color=c, edgecolor=_darker(c), linewidth=1.0)
        if SHOW_VALUES:
            ax.text(v + (hi - lo) * 0.01, y, fv.r2(v), va='center', ha='left', fontsize=VALUE_SIZE)
    for key, label in data['nonprivate_labels']:
        if lines.get(key) is not None:
            ax.axvline(lines[key], zorder=3, **BASELINE_STYLES.get(key, dict(color='black', linestyle=':')))
    ax.set_xlim(lo, hi)
    ax.set_yticks(ys)
    ax.set_yticklabels([g.METHOD_LABELS[m] for m in METHODS] if show_ylabels else [], fontsize=TICK_SIZE)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.grid(False, axis='y')
    text = ax.set_xlabel(g.label(X_KEY, g.K_MAIN), fontsize=LABEL_SIZE)
    g.add_arrow(text, X_KEY, 'x')
    if TITLE:
        ax.set_title(TITLE.format(dataset=g.DATASET_LABELS.get(dataset, dataset)), fontsize=TITLE_SIZE)


def legend(fig, data):
    handles = [Line2D([], [], label=label, **BASELINE_STYLES.get(key, dict(color='black', linestyle=':')))
               for key, label in data['nonprivate_labels']]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               **{**g.LEGEND_STYLE, 'fontsize': LEGEND_SIZE, 'handlelength': 1.6})


def print_values(data):
    print('\nBest MRR@10 drawn (method rows, dataset columns; non-private references as dotted lines)')
    head = ['Method'] + [g.DATASET_LABELS.get(d, d) for d in DATASETS]
    rows = [[label] + [fv.r2(data['nonprivate'][(k, d)]) if data['nonprivate'].get((k, d)) else 'n/a' for d in DATASETS]
            for k, label in data['nonprivate_labels']]
    rows += [[g.METHOD_LABELS[m]] + [fv.r2(data['best'][(m, d)][X_KEY]) if data['best'].get((m, d)) else 'n/a'
                                     for d in DATASETS] for m in METHODS]
    print('\n'.join(rf.md(head, rows)) + '\n')


def make_plots(nested_data):
    R = rf.get_runs(nested_data)
    data = rf.best_scores(R, DATASETS, METHODS).data
    paths = []
    for d in DATASETS:
        fig, ax = plt.subplots(figsize=FIG_SIZE, layout='constrained')
        draw(ax, data, d)
        legend(fig, data)
        paths.append(pu.save_figure(fig, f'best_mrr_{d}'))
    if GRID:
        rows = -(-len(DATASETS) // GRID_COLS)
        fig, axes = plt.subplots(rows, GRID_COLS, figsize=(GRID_PANEL[0] * GRID_COLS, GRID_PANEL[1] * rows),
                                 squeeze=False, layout='constrained')
        for i, (ax, d) in enumerate(zip(axes.flat, DATASETS)):
            draw(ax, data, d, show_ylabels=i % GRID_COLS == 0)
        for ax in list(axes.flat)[len(DATASETS):]:
            ax.set_visible(False)
        legend(fig, data)
        paths.append(pu.save_figure(fig, 'best_mrr_grid'))
    if PRINT_VALUES:
        print_values(data)
    return paths


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
