"""
Where BM25-Tree's latency goes: one bar per selected config, split into the
three PIR stages. One PDF per dataset.

    figures/tree_stages_<dataset>.pdf

The configs come from select_configs.py, so they are the same five the other
tree plots use. Set CONFIGS below to pin specific ones by hand instead.
"""

import numpy as np
from matplotlib.patches import Patch

import globals as g
import load_results
import plot_utils as pu
import select_configs as sc

Y_KEY = 'lan_time'    # stacked as lan_time_s1/_s2/_s3; 'wan_time' and 'comm_kb' also split
K = g.K_ABLATION
N_CONFIGS = 5

BAR_WIDTH = 0.65
SHOW_TOTALS = False   # print the total above each bar

# {dataset: [config, ...]} to choose the bars by hand; empty = use select_configs.
CONFIGS = {}


def stage_values(run, y_key):
    """[stage 1, stage 2, stage 3] for this run, missing stages as 0."""
    return [run.get(f'{y_key}_{stage}', np.nan) for stage in g.STAGE_ORDER]


def plot_tree_stages(nested_data, dataset, y_key=Y_KEY):
    runs = pu.get_runs(nested_data, 'tree', dataset, k=K)
    if CONFIGS.get(dataset):
        by_config = {r['config']: r for r in runs}
        runs = [by_config[c] for c in CONFIGS[dataset] if c in by_config]
        missing = [c for c in CONFIGS[dataset] if c not in by_config]
        if missing:
            pu.log.warning("tree/%s: no run for config(s) %s", dataset, ', '.join(missing))
    else:
        runs = sc.select_configs(runs, n=N_CONFIGS, name=f'tree/{dataset}')
    if not runs:
        pu.log.warning("tree/%s: nothing to plot", dataset)

    fig, ax = pu.new_figure()
    slots = np.arange(len(runs), dtype=float)
    bottoms = np.zeros(len(runs))

    for stage in g.STAGE_ORDER:
        heights = np.array([run.get(f'{y_key}_{stage}', np.nan) for run in runs])
        if np.all(np.isnan(heights)) and runs:
            pu.log.warning("tree/%s: %s missing from every run", dataset, stage)
        heights = np.nan_to_num(heights)
        ax.bar(slots, heights, bottom=bottoms, width=BAR_WIDTH,
               color=g.STAGE_COLORS[stage], hatch=g.STAGE_HATCHES[stage],
               edgecolor='#333333', linewidth=0.6)
        bottoms += heights

    if SHOW_TOTALS:
        for x, total in zip(slots, bottoms):
            ax.text(x, total, f'{total:.3g}', ha='center', va='bottom', fontsize=g.LEGEND_SIZE - 1)

    ax.set_xticks(slots)
    ax.set_xticklabels([run['config'] for run in runs], rotation=30, ha='right')
    # rotation/ha keep the b.._r.._s.._L.. labels from colliding
    ax.tick_params(axis='x', length=0)
    ax.grid(False, axis='x')
    ax.set_ylabel(g.label(y_key, K))
    ax.set_title(g.DATASET_LABELS[dataset], pad=18)
    # pad pushes the title up to leave room for the legend

    handles = [Patch(facecolor=g.STAGE_COLORS[s], hatch=g.STAGE_HATCHES[s],
                     edgecolor='#333333', linewidth=0.6, label=g.STAGE_LABELS[s])
               for s in g.STAGE_ORDER]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3,
              **g.LEGEND_STYLE)
    # bbox_to_anchor moves the legend: (0.5, 0.98) = centred, just above the axes

    return pu.save_figure(fig, f'tree_stages_{dataset}')


def make_plots(nested_data):
    for dataset in g.DATASETS:
        plot_tree_stages(nested_data, dataset)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results())
