"""
Query-term ablation: PILLAR-Tree's MRR@10 (Y_KEYS) against the number of query terms it keeps (q*), a line per
dataset, in the look of the PILLAR-Bin ablation line plots (plot_bins_ablation_grid.py: colour and marker per
dataset, evenly spaced y ticks):

    figures/query_terms.pdf

The runs are rebuttal_final.query_terms(): per dataset the (b, r, s, L, c) family swept over the most q* values,
everything but q* fixed (the "query terms" table of figures/rebuttal_final.md). Every point drawn is printed.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import globals as g
import load_results
import plot_pareto_grid as pg
import plot_utils as pu
import rebuttal_final as rf

Y_KEYS = ['mrr']              # one panel each, e.g. ['mrr', 'recall']
DATASETS = None               # None = every dataset rebuttal_final.query_terms() keeps (QT_DATASETS with >= 3 q* values)
PANEL_SIZE = (3.0, 2.0)
LOG_X = False                 # q* runs 1 .. 12 (24 on SciFact); True spreads the small values out
X_TICKS = None                # e.g. [1, 2, 4, 8, 12]; None = every q* that was run
N_TICKS = 5                   # y ticks (plot_pareto_grid.even_ticks), the outer two on the ends of the axis
Y_FROM_ZERO = True
MARKER_SIZE = g.MARKER_SIZE
LABEL_SIZE = 12
TICK_SIZE = 10
X_LABEL = 'Query Terms Used (q*)'
LEGEND_COLS = 3
PRINT_VALUES = True


def draw(ax, data, y_key, datasets):
    ys_drawn = []
    for d in datasets:
        byq = data['runs'][d]
        qs = sorted(byq)
        ys = [byq[q][y_key] for q in qs]
        ax.plot(qs, ys, color=g.DATASET_COLORS.get(d, '#333333'), marker=g.DATASET_MARKERS.get(d, 'o'),
                markersize=MARKER_SIZE, clip_on=False, label=g.DATASET_LABELS.get(d, d))
        ys_drawn += ys
    if LOG_X:
        ax.set_xscale('log', base=2)
    ticks = X_TICKS or data['q_values']
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p=None: f'{v:g}'))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    lo, hi = (0 if Y_FROM_ZERO else min(ys_drawn)), max(ys_drawn)
    yt, (lo, hi) = pg.even_ticks(lo, hi, False, False, n=N_TICKS)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(ticker.FixedLocator(yt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p=None: pg.short(v)))
    ax.grid(True, which='major')
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    text = ax.set_ylabel(g.label(y_key, g.K_MAIN), fontsize=LABEL_SIZE)
    g.add_arrow(text, y_key, 'y')


def make_plots(nested_data):
    R = rf.get_runs(nested_data)
    res = rf.query_terms(R.nd)
    data = res.data
    datasets = [d for d in (DATASETS or data['datasets']) if d in data['runs']]
    if not datasets:
        pu.log.warning('query_terms: no dataset with a q* sweep, nothing to plot')
        return None
    fig, axes = plt.subplots(1, len(Y_KEYS), figsize=(PANEL_SIZE[0] * len(Y_KEYS), PANEL_SIZE[1]),
                             squeeze=False, layout='constrained')
    for ax, y_key in zip(axes.flat, Y_KEYS):
        draw(ax, data, y_key, datasets)
    handles = [Line2D([], [], color=g.DATASET_COLORS.get(d, '#333333'), marker=g.DATASET_MARKERS.get(d, 'o'),
                      markersize=MARKER_SIZE, label=g.DATASET_LABELS.get(d, d)) for d in datasets]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=LEGEND_COLS,
               **{**g.LEGEND_STYLE, 'fontsize': 10, 'handlelength': 1.6})
    path = pu.save_figure(fig, 'query_terms')
    if PRINT_VALUES:
        print(f'\n{res.title} (points of {path})')
        for y_key in Y_KEYS:
            head = ['q*'] + [g.DATASET_LABELS.get(d, d) for d in datasets]
            rows = [[q] + [f"{data['runs'][d][q][y_key]:.3f}" if q in data['runs'][d] else '' for d in datasets]
                    for q in data['q_values']]
            print(f'{g.label(y_key, g.K_MAIN)}:')
            print('\n'.join(rf.md(head, rows)) + '\n')
    return path


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
