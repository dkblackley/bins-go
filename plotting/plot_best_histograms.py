"""
Hand-picked "best balance" config per method, bars grouped by dataset. One PDF
per metric so each can sit next to the matching line plot:

    figures/best_<y_key>.pdf

Values are typed into HAND_PICKED below, not read from disk. Every value is a
1.0 placeholder; fill them in from
    python load_results.py <folder>
which prints every key of that run.
"""

import numpy as np

import globals as g
import plot_utils as pu

Y_KEYS = ['mrr', 'comm_kb', 'faithfulness', 'answer_relevancy', 'lan_time']
K = g.K_MAIN   # only used for the MRR@k axis label

BAR_GROUP_WIDTH = 0.8   # share of each dataset slot filled by its bars
SHOW_VALUES = False     # print each bar's height above it

HAND_PICKED = {
    'bins': {
        'msmarco': {'folder': 'TODO', 'mrr': 1.0, 'comm_kb': 1.0, 'faithfulness': 1.0,
                    'answer_relevancy': 1.0, 'lan_time': 1.0},
        'scifact': {'folder': 'TODO', 'mrr': 1.0, 'comm_kb': 1.0, 'faithfulness': 1.0,
                    'answer_relevancy': 1.0, 'lan_time': 1.0},
    },
    'tree': {
        'msmarco': {'folder': 'TODO', 'mrr': 1.0, 'comm_kb': 1.0, 'faithfulness': 1.0,
                    'answer_relevancy': 1.0, 'lan_time': 1.0},
        'scifact': {'folder': 'TODO', 'mrr': 1.0, 'comm_kb': 1.0, 'faithfulness': 1.0,
                    'answer_relevancy': 1.0, 'lan_time': 1.0},
    },
    'pacmann': {
        'msmarco': {'folder': 'TODO', 'mrr': 1.0, 'comm_kb': 1.0, 'faithfulness': 1.0,
                    'answer_relevancy': 1.0, 'lan_time': 1.0},
        'scifact': {'folder': 'TODO', 'mrr': 1.0, 'comm_kb': 1.0, 'faithfulness': 1.0,
                    'answer_relevancy': 1.0, 'lan_time': 1.0},
    },
}


def plot_best_histogram(y_key, values=HAND_PICKED):
    fig, ax = pu.new_figure()
    slots = np.arange(len(g.DATASETS))
    bar_width = BAR_GROUP_WIDTH / len(g.METHOD_ORDER)

    for i, method in enumerate(g.METHOD_ORDER):
        heights = [values.get(method, {}).get(d, {}).get(y_key, np.nan) for d in g.DATASETS]
        if any(np.isnan(heights)):
            pu.log.warning("best_%s: %s has no value for some datasets", y_key, method)
        offset = (i - (len(g.METHOD_ORDER) - 1) / 2) * bar_width
        bars = ax.bar(slots + offset, heights, width=bar_width * 0.92,
                      label=g.METHOD_LABELS[method], color=g.METHOD_COLORS[method],
                      hatch=g.METHOD_HATCHES[method], edgecolor='#333333', linewidth=0.6)
        if SHOW_VALUES:
            ax.bar_label(bars, fmt='%.2f', padding=1, fontsize=g.LEGEND_SIZE - 1)

    ax.set_xticks(slots)
    ax.set_xticklabels([g.DATASET_LABELS[d] for d in g.DATASETS])
    ax.tick_params(axis='x', length=0)   # no tick marks under the dataset names
    ax.grid(False, axis='x')
    ax.set_ylabel(g.label(y_key, K))

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3, **g.LEGEND_STYLE)
    # bbox_to_anchor moves the legend: (0.5, 0.98) = centred, just above the axes

    return pu.save_figure(fig, f'best_{y_key}')


def make_plots():
    for y_key in Y_KEYS:
        plot_best_histogram(y_key)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots()
