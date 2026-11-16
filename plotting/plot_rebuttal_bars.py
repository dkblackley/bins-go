"""
Bar charts of the fixed-value comparisons, the same numbers as the tables in figures/rebuttal_final.md
(rebuttal_final.py does the selection; this file only draws it):

    figures/fixed_mrr_bin.pdf        WAN per dataset at MRR@10 = 0.10 (PILLAR-Bin, PACMANN, Tiptoe, Coeus)
    figures/fixed_mrr_tree.pdf       WAN per dataset at MRR@10 = 0.15 (PILLAR-Tree, PACMANN, Tiptoe, Coeus)
    figures/scale.pdf                WAN and client storage per corpus size at MRR@10 = 0.05
    figures/fixed_storage_<S>gb.pdf  WAN and MRR@10 of PILLAR-Bin and PACMANN at client storage S (one per S)

Targets, methods and datasets are rebuttal_final.FIXED_MRR / STORAGE_*; FIXED_MRR_FIGS says which get a figure
(add 'fixed_mrr_hybrid' for the Tiptoe+Coeus table, left out by default). Bars are grouped by dataset, a bar per
method; the y axis is the cost compared, with the better-direction arrow. A method that never reaches the target
gets a hatched bar (UNREACHED_HATCH), a method with no runs an 'n/a'. Every value drawn is printed.
"""

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import globals as g
import load_results
import plot_utils as pu
import rebuttal_final as rf
import frozen_values as fv

FIXED_MRR_FIGS = ['fixed_mrr_bin', 'fixed_mrr_tree', 'scale']   # + 'fixed_mrr_hybrid'
STORAGE_FIGS = True            # one figure per rebuttal_final.STORAGE_BUDGETS_GB
STORAGE_PANELS = ['wan_time', 'mrr']

PANEL_SIZE = (3.4, 2.2)        # one panel, inches
GROUP_WIDTH = 0.8              # share of a dataset slot filled by its bars
EDGE_DARKEN = 0.6
LOG = {'wan_time': True}       # log y per key (Coeus is ~100x slower than PILLAR-Bin); others plain
YLIM = {}                      # key -> (lo, hi) in the unit shown (GB for client storage), else fitted
SHOW_VALUES = True
VALUE_SIZE = 7
UNREACHED_HATCH = '///'
NA_TEXT = 'n/a'
SCALE = {'client_storage_mb': (1 / 1024, 'Client Storage (GB)')}   # key -> (factor, axis label)
TITLE = {'fixed_mrr': 'MRR@10 = {target:.2f}', 'storage': 'Client storage {s:g} GB'}
TITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 9
XTICK_SIZE = 9
LEGEND = True
PRINT_VALUES = True


def _darker(color):
    r, gr, b = mcolors.to_rgb(color)
    return (r * EDGE_DARKEN, gr * EDGE_DARKEN, b * EDGE_DARKEN)


def _fmt(key, v):
    if key in ('mrr', 'recall'):
        return fv.r2(v)
    if key == 'client_storage_mb':
        return f'{v:.2f}'
    return rf.wr.f_s(v)


def bars(ax, groups, methods, value, key, hatched=lambda m, gk: False):
    """grouped bars: a slot per group, a bar per method; value(method, group) -> number in the unit shown or None"""
    width = GROUP_WIDTH / len(methods)
    drawn = []
    for i, m in enumerate(methods):
        for j, gk in enumerate(groups):
            x = j + (i - (len(methods) - 1) / 2) * width
            v = value(m, gk)
            if v is None or not pu.is_number(v):
                drawn.append((x, None))
                continue
            c = g.METHOD_COLORS.get(m, '#777777')
            ax.bar(x, v, width=width * 0.9, color=c, edgecolor=_darker(c), linewidth=0.8,
                   hatch=UNREACHED_HATCH if hatched(m, gk) else None,
                   label=g.METHOD_LABELS.get(m, rf.L(m)) if j == 0 else None)
            drawn.append((x, v))
    log = LOG.get(key) and all(v > 0 for _, v in drawn if v is not None)
    if log:
        ax.set_yscale('log')
    if key in YLIM:
        ax.set_ylim(*YLIM[key])
    lo, hi = ax.get_ylim()
    for x, v in drawn:
        if v is None:
            ax.text(x, lo if not log else lo * 1.1, NA_TEXT, ha='center', va='bottom', fontsize=VALUE_SIZE,
                    rotation=90, color='#666666')
        elif SHOW_VALUES:
            ax.text(x, v, _fmt(key, v), ha='center', va='bottom', fontsize=VALUE_SIZE)   # v is in the unit shown
    ax.set_xticks(range(len(groups)))
    ax.tick_params(axis='x', length=0, labelsize=XTICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    if log:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p=None: f'{v:g}'))
    ax.grid(False, axis='x')
    label = SCALE[key][1] if key in SCALE else g.label(key, g.K_MAIN)
    text = ax.set_ylabel(label, fontsize=LABEL_SIZE)
    g.add_arrow(text, key, 'y')


def _legend(fig, methods):
    handles = {}
    for ax in fig.axes:
        for h, t in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(t, h)
    labels = [g.METHOD_LABELS.get(m, rf.L(m)) for m in methods if g.METHOD_LABELS.get(m, rf.L(m)) in handles]
    fig.legend([handles[t] for t in labels], labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), **g.LEGEND_STYLE)


def fixed_mrr_figure(R, key):
    res = rf.fixed_mrr(R, key)
    d = res.data
    picks, ms, ds, costs = d['picks'], d['methods'], d['datasets'], d['costs']
    fig, axes = plt.subplots(1, len(costs), figsize=(PANEL_SIZE[0] * len(costs), PANEL_SIZE[1]),
                             squeeze=False, layout='constrained')
    for ax, c in zip(axes.flat, costs):
        f = SCALE.get(c, (1,))[0]
        val = lambda m, dd: picks[(m, dd)][0][c] * f if picks[(m, dd)][0] and pu.is_number(picks[(m, dd)][0].get(c)) else None
        bars(ax, ds, ms, val, c, hatched=lambda m, dd: picks[(m, dd)][1] == '*')
        ax.set_xticklabels([g.DATASET_LABELS.get(x, x) for x in ds])
    fig.suptitle(TITLE['fixed_mrr'].format(target=d['target']), fontsize=TITLE_SIZE)
    if LEGEND:
        _legend(fig, ms)
    path = pu.save_figure(fig, key)
    if PRINT_VALUES:
        print(f"\n{res.title} (bars of {path})")
        print('\n'.join(res.clean) + '\n')
    return path


def storage_figures(R):
    res = rf.fixed_storage(R)
    d = res.data
    paths = []
    for s in d['budgets']:
        fig, axes = plt.subplots(1, len(STORAGE_PANELS), figsize=(PANEL_SIZE[0] * 0.7 * len(STORAGE_PANELS), PANEL_SIZE[1]),
                                 squeeze=False, layout='constrained')
        for ax, key in zip(axes.flat, STORAGE_PANELS):
            val = lambda m, _g: d['picks'][(s, m)][key] if d['picks'][(s, m)] else None
            bars(ax, [None], d['methods'], val, key)
            ax.set_xticks([])
        fig.suptitle(TITLE['storage'].format(s=s), fontsize=TITLE_SIZE)
        if LEGEND:
            _legend(fig, d['methods'])
        paths.append(pu.save_figure(fig, f'fixed_storage_{s:g}gb'.replace('.', 'p')))
    if PRINT_VALUES:
        print(f'\n{res.title} (bars of {", ".join(paths)})')
        print('\n'.join(res.clean) + '\n')
    return paths


def make_plots(nested_data):
    R = rf.get_runs(nested_data)
    paths = [fixed_mrr_figure(R, key) for key in FIXED_MRR_FIGS]
    if STORAGE_FIGS:
        paths += storage_figures(R)
    return paths


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
