"""
BM25-Bin ablation: docs-per-bin sweep (top row, bin size fixed at 1.0) and
bin-size sweep (bottom row, docs/bin fixed at 1000), against MRR@10, runtime
and communication cost. Replaces the 3-by-2 `groupplot`.

    plot_bins_ablation(data, OUTPUT_DIR)

`data[sweep]['x']` plus `data[sweep][dataset_key][metric] = [...]`. Build it
from your loader with data_from_nested(nested_data), or pass HARDCODED.
"""

import util as ps
import numpy as np

SWEEPS = [
    # xticks: which sampled points get a label. Labelling all six dpb values
    # collides at this panel width, so label one per decade.
    {'key': 'dpb', 'xlabel': 'Docs per bin', 'base': 10,
     'xticks': [10, 100, 1000], 'ticklabels': ['10', '100', '1000'],
     'ylims': {'mrr': (0.0, 0.6), 'time': (0.0, 0.2), 'comm': (0.0, 25)}},
    {'key': 'bs',  'xlabel': 'Bin size', 'base': 10,
     'xticks': [0.01, 0.1, 1], 'ticklabels': ['0.01', '0.1', '1'],
     'ylims': {'mrr': (0.0, 0.6), 'time': (0.0, 3.0), 'comm': (0.0, 100)}},
]

METRICS = [
    {'key': 'mrr',  'title': 'MRR@10'},
    {'key': 'time', 'title': 'Runtime (s)'},
    {'key': 'comm', 'title': 'Comm. (MB)'},
]

# Run keys in your nested_data, mapped onto the metric keys above.
RUN_KEYS = {'mrr': 'mrr', 'time': 'total_time', 'comm': 'comm_cost'}


# ==========================================
# DATA
# ==========================================

def _num(value, default=np.nan):
    """metadata.json returns some fields as strings ('11601'); coerce or drop."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def data_from_nested(nested_data, target_k=100, method='bins',
                     datasets=ps.DATASETS, fixed_bs=1.0, fixed_dpb=1000):
    """
    Pulls both sweeps out of the loader's nested_data. The bin-size sweep is
    plotted against the multiplier `bs`, not the measured RealBinSize: the
    latter is an absolute width, so the same multiplier lands on a different
    number for every dataset and the two lines would not share an axis.

    Each dataset carries its own 'x', since a config can be missing on one
    dataset and present on the other.
    """
    data = {'dpb': {}, 'bs': {}}

    for sweep in ('dpb', 'bs'):
        for ds in datasets:
            runs = [r for r in nested_data.get(method, {}).get(ds['key'], [])
                    if r['k'] == target_k]

            if sweep == 'dpb':
                runs = [r for r in runs
                        if r['dpb'] is not None and _num(r['bs']) == fixed_bs]
                runs.sort(key=lambda r: _num(r['dpb']))
                xs = [_num(r['dpb']) for r in runs]
            else:
                runs = [r for r in runs
                        if r['bs'] is not None and _num(r['dpb']) == fixed_dpb]
                runs.sort(key=lambda r: _num(r['bs']))
                xs = [_num(r['bs']) for r in runs]

            if not runs:
                print(f"[WARN] no {method} runs for {ds['key']} {sweep} sweep")
                continue

            series = {'x': xs}
            series.update({m: [_num(r[RUN_KEYS[m]]) for r in runs]
                           for m in RUN_KEYS})
            data[sweep][ds['key']] = series

            # Widest grid across datasets, used only for tick placement.
            if len(xs) > len(data[sweep].get('x', [])):
                data[sweep]['x'] = xs

    return data

# Numbers from the pgfplots version, so this module runs standalone.
HARDCODED = {
    'dpb': {
        'x': [10, 50, 100, 500, 1000, 5000],
        'msmarco': {
            'mrr':  [0.0514, 0.0887, 0.1143, 0.1725, 0.1923, 0.2347],
            'time': [0.0074, 0.0146, 0.0239, 0.0877, 0.1765, 0.5205],
            'comm': [0.0075, 0.0235, 0.0409, 0.1759, 0.3306, 1.1993],
        },
        'scifact': {
            'mrr':  [0.4206, 0.5272, 0.5351, 0.5653, 0.5619, 0.5631],
            'time': [0.0048569, 0.01547038, 0.02522669, 0.1055198, 0.1372965,
                     0.56254626],
            'comm': [0.00831507, 0.0311282, 0.05180926, 0.20723704, 0.26970665,
                     1.06624738],
        },
    },
    'bs': {
        'x': [0.01, 0.1, 1.0],
        'msmarco': {
            'mrr':  [0.1850, 0.1848, 0.1923, 0.1919],
            'time': [0.9204, 0.3156, 0.1765, 0.1361],
            'comm': [1.2845, 0.4966, 0.3306, 0.3275],
        },
        'scifact': {
            'mrr':  [0.3310, 0.5479, 0.5619, 0.5614],
            'time': [2.7055, 0.2278, 0.1373, 0.1660],
            'comm': [0.8820, 0.4074, 0.2697, 0.2565],
        },
    },
}


# ==========================================
# PLOTTING
# ==========================================

def plot_bins_ablation(data, output_dir, filename='fig_bins_ablation.pdf',
                       sweeps=SWEEPS, metrics=METRICS, datasets=ps.DATASETS,
                       colors=ps.DATASET_COLORS, markers=ps.DATASET_MARKERS,
                       height_in=3.20, scale=ps.SCALE):
    """One figure, both sweeps, shared legend, row-level titles and global X/Y labels."""
    # Slightly taller height_in (2.80 -> 3.20) to make room for the new row titles
    fig, axes = ps.new_figure(len(sweeps), len(metrics), height_in, scale=scale)

    for row, sweep in enumerate(sweeps):
        block = data.get(sweep['key'], {})
        xs = block.get('x', [])

        # 1. Define the big subtitle for this specific row
        row_subtitle = 'Changing Documents Per Bin' if sweep['key'] == 'dpb' else 'Changing Bin Size'

        for col, metric in enumerate(metrics):
            ax = axes[row * len(metrics) + col]

            for ds in datasets:
                series = block.get(ds['key'], {})
                ys = series.get(metric['key'])
                if not ys:
                    continue
                ax.plot(series.get('x', block.get('x', [])), ys,
                        marker=markers.get(ds['key'], 'o'), linestyle='-',
                        color=colors.get(ds['key'], '#000000'),
                        label=ds['label'], zorder=3)

            ps.log_xticks(ax, sweep.get('xticks') or xs, sweep['ticklabels'],
                          base=sweep['base'], xlim=xs)

            # 2. Update styling: Y-axis labeled, per-plot title removed, per-plot xlabel removed
            ps.style_axes(ax,
                          title=None,
                          ylabel=None,
                          ylim=sweep['ylims'].get(metric['key']),
                          xlabel=None,
                          xgrid=True)

            ax.set_ylabel(metric['title'], labelpad=0, y=0.35)

            # 3. Apply the big row subtitle and big X-axis label to the center plot ONLY
            if col == 1:
                ax.set_title(row_subtitle, fontsize=(ps.FONT_PT + 2) * scale, pad=16 * scale, fontweight='bold')
                ax.set_xlabel(sweep['xlabel'], fontsize=(ps.FONT_PT + 1) * scale, labelpad=0 * scale)

    handles, labels = axes[0].get_legend_handles_labels()
    ps.shared_legend(fig, handles, ncol=len(datasets), labels=labels, y=1.05)

    # 4. Tweak spacing to accommodate the new text locations
    # wspace increased for y-labels; hspace adjusted for the row titles
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.12,
                        wspace=0.6, hspace=1.25)

    return ps.save(fig, output_dir, filename, scale)

if __name__ == "__main__":
    ps.setup_sleek_style()
    plot_bins_ablation(HARDCODED, './plots')