"""
Experiment 2: per-method comparison at each method's best-MRR operating point.
Replaces the 4-by-1 `groupplot` of ybars.

    plot_exp2_comparison(values, OUTPUT_DIR, layout='row')

`values[metric_key][method][dataset_key] = float`. Build it from your loader
with values_from_nested(nested_data), or pass HARDCODED to check the layout.
"""

import numpy as np

import util as ps

# (metric key in run_info, panel title, short title for narrow panels, ylim)
METRICS = [
    {'key': 'mrr',              'title': 'MRR@10',
     'short': 'MRR@10',       'ylim': (0.0, 0.70)},
    {'key': 'faithfulness',     'title': 'Faithfulness',
     'short': 'Faithfulness', 'ylim': (0.0, 0.85)},
    {'key': 'answer_relevancy', 'title': 'Relevancy',
     'short': 'Relevancy',    'ylim': (0.60, 0.85)},
    {'key': 'comm_cost',        'title': 'Comm. Cost (MB)',
     'short': 'Comm. (MB)',   'ylim': (0.0, 40.65)},
]

# Relevancy keeps the non-zero baseline of the original figure: all three
# methods sit in [0.74, 0.81], so a zero baseline flattens the panel into three
# identical bars. Change the ylim above to (0.0, 0.85) if a reviewer objects.

GROUP_FRAC = 0.86   # share of the unit slot occupied by one group of bars


# ==========================================
# DATA
# ==========================================

# ==========================================
# BALANCED CONFIG SELECTION
# ==========================================

# Which metrics enter the balance score, and which direction is better.
# Add 'total_time': 'low' if runtime should count as well.
BALANCE_METRICS = {
    'mrr':              'low',
    'faithfulness':     'low',
    'answer_relevancy': 'low',
    'comm_cost':        'high',
}

# Metric -> weight, missing entries default to 1.0. Halve a metric here rather
# than dropping it if you want it to break ties without driving the choice.
BALANCE_WEIGHTS = {}


def _normalised_costs(runs, metrics, scheme='minmax'):
    """
    Map each metric onto a cost where 0 is the best run in the pool, so the
    directions can be summed. Returns one {metric: cost} dict per run.
    """
    costs = [dict() for _ in runs]
    for key, direction in metrics.items():
        arr = np.array([float(r.get(key, np.nan)) for r in runs], dtype=float)
        if np.all(np.isnan(arr)):
            print(f"[WARN] metric {key!r} missing from every candidate run")
            continue

        if scheme == 'zscore':
            sd = np.nanstd(arr)
            c = np.zeros_like(arr) if sd == 0 else (arr - np.nanmean(arr)) / sd
            if direction == 'high':
                c = -c
        else:
            lo, hi = np.nanmin(arr), np.nanmax(arr)
            if hi - lo < 1e-12:
                c = np.zeros_like(arr)   # metric doesn't separate these runs
            else:
                c = (arr - lo) / (hi - lo)
                if direction == 'high':
                    c = 1.0 - c

        for i, v in enumerate(c):
            costs[i][key] = 0.5 if np.isnan(v) else float(v)   # missing = mid
    return costs


def _score(cost, weights):
    return sum(weights.get(k, 1.0) * v for k, v in cost.items())


def select_best_runs(nested_data, criterion='balanced', k_val=100,
                     method_order=ps.METHOD_ORDER, datasets=ps.DATASETS,
                     balance_metrics=BALANCE_METRICS, weights=None,
                     normalise='minmax', pool='method'):
    """
    Pick one configuration per method per dataset.

    criterion='balanced' normalises every metric in balance_metrics onto a
    common [0, 1] cost (0 = best), sums them, and takes the lowest total, so a
    config only wins by doing well everywhere. Any other value ('mrr',
    'faithfulness', ...) falls back to maximising that single metric.

    pool='method'  normalises within each method's own candidate configs. Use
                   this to answer "which of my configs is the best compromise".
    pool='dataset' normalises across every method's runs on that dataset, so
                   the costs share one scale and the printed scores are
                   comparable between methods.

    normalise='minmax' is scale-free but sensitive to outliers: one config with
    a huge comm cost squashes the rest of the sweep towards 0. 'zscore' spreads
    them by standard deviation instead.

    Prints its pick with the per-metric breakdown, so a surprising bar in the
    figure can be traced back to the config and the term that drove it.
    """
    weights = weights or BALANCE_WEIGHTS
    best = {m: {} for m in method_order}

    for ds in datasets:
        candidates = {
            m: [r for r in nested_data.get(m, {}).get(ds['key'], [])
                if k_val is None or r['k'] == k_val]
            for m in method_order
        }

        lookup = None
        if criterion == 'balanced' and pool == 'dataset':
            flat = [r for m in method_order for r in candidates[m]]
            lookup = {id(r): c for r, c in
                      zip(flat, _normalised_costs(flat, balance_metrics, normalise))}

        for method in method_order:
            runs = candidates[method]
            if not runs:
                print(f"[WARN] no runs for {method}/{ds['key']} (k={k_val})")
                continue

            if criterion != 'balanced':
                top, score, cost = max(runs, key=lambda r: r[criterion]), None, None
            else:
                costs = ([lookup[id(r)] for r in runs] if lookup is not None
                         else _normalised_costs(runs, balance_metrics, normalise))
                score, idx = min((_score(c, weights), i) for i, c in enumerate(costs))
                top, cost = runs[idx], costs[idx]

            best[method][ds['key']] = top
            detail = '' if cost is None else '  ' + ' '.join(
                f'{k}={cost[k]:.2f}' for k in balance_metrics if k in cost)
            print(f"[PICK] {method:<8} {ds['key']:<9} config={top['config']:<24} "
                  + (f"score={score:.3f}{detail}" if score is not None
                     else f"{criterion}={top[criterion]:.4f}"))

    return best

def values_from_nested(nested_data, criterion='mrr', k_val=100,
                       method_order=ps.METHOD_ORDER, datasets=ps.DATASETS,
                       metrics=METRICS):
    best = select_best_runs(nested_data, criterion, k_val, method_order, datasets)
    values = {}
    for metric in metrics:
        key = metric['key']
        values[key] = {
            m: {ds['key']: best.get(m, {}).get(ds['key'], {}).get(key, np.nan)
                for ds in datasets}
            for m in method_order
        }
    return values


# Numbers from the pgfplots version, so this module runs standalone.
HARDCODED = {
    'mrr': {
        'pacmann': {'scifact': 0.5393,     'msmarco': 0.2427},
        'bins':    {'scifact': 0.5653,     'msmarco': 0.1923},
        'tree':    {'scifact': 0.6183,     'msmarco': 0.3231},
    },
    'faithfulness': {
        'pacmann': {'scifact': 0.37333333, 'msmarco': 0.5923},
        'bins':    {'scifact': 0.42483333, 'msmarco': 0.6388},
        'tree':    {'scifact': 0.40166667, 'msmarco': 0.7667},
    },
    'answer_relevancy': {
        'pacmann': {'scifact': 0.7748981,  'msmarco': 0.7695},
        'bins':    {'scifact': 0.74800144, 'msmarco': 0.7490},
        'tree':    {'scifact': 0.79527016, 'msmarco': 0.8039},
    },
    'comm_cost': {
        'pacmann': {'scifact': 0.01220534, 'msmarco': 0.2962},
        'bins':    {'scifact': 0.20723704, 'msmarco': 0.2306},
        'tree':    {'scifact': 0.1827175,  'msmarco': 0.5805},
    },
}


# ==========================================
# PLOTTING
# ==========================================

def _draw_panel(ax, values, metric, method_order, datasets, colors, hatches,
                labels, use_short, show_values, scale):
    key = metric['key']
    n_m = len(method_order)
    centres = np.arange(len(datasets), dtype=float)
    bar_w = GROUP_FRAC / n_m

    for i, method in enumerate(method_order):
        offs = (i - (n_m - 1) / 2.0) * bar_w
        heights = [values[key][method].get(ds['key'], np.nan) for ds in datasets]
        bars = ax.bar(centres + offs, heights, width=bar_w * 0.92,
                      color=colors[method], edgecolor='#333333',
                      linewidth=0.55 * scale, hatch=hatches.get(method, ''),
                      label=labels.get(method, method), zorder=3)
        if show_values:
            ax.bar_label(bars, fmt='%.2f', padding=1.0 * scale,
                         fontsize=(ps.FONT_PT - 2.0) * scale,
                         color='#444444', zorder=4)

    ylim = metric.get('ylim')
    if ylim is None:
        top = np.nanmax([values[key][m][d['key']]
                         for m in method_order for d in datasets])
        ylim = (0.0, top * (1.30 if show_values else 1.15))

    ps.style_axes(ax, title=metric['short'] if use_short else metric['title'],
                  ylim=ylim)

    ax.set_xticks(centres)
    ax.set_xticklabels([d['short'] if use_short else d['label'] for d in datasets])
    pad = 0.5 * GROUP_FRAC + 0.15
    ax.set_xlim(-pad, len(datasets) - 1 + pad)
    ax.tick_params(axis='x', length=0)


def plot_exp2_comparison(values, output_dir, filename=None, layout='row',
                         method_order=ps.METHOD_ORDER, datasets=ps.DATASETS,
                         metrics=METRICS, colors=ps.COLORS, hatches=ps.HATCHES,
                         labels=ps.LABELS, show_values=False, scale=ps.SCALE):
    """
    layout='row'  -> 1x4 strip, ~1.4in tall on the page. The compact option.
    layout='grid' -> 2x2 block, ~2.9in tall, taller bars and room for numbers.

    show_values prints each bar's height above it. Readable in 'grid'; in 'row'
    the panels are too narrow and adjacent numbers touch, so leave it off there.
    """
    if layout == 'row':
        nrows, ncols, height_in = 1, len(metrics), 1.55
        wspace, hspace, top, bottom, use_short = 0.40, 0.0, 0.78, 0.28, True
    elif layout == 'grid':
        nrows, ncols, height_in = 2, 2, 3.10
        wspace, hspace, top, bottom, use_short = 0.34, 0.62, 0.87, 0.10, False
    else:
        raise ValueError(f"unknown layout: {layout!r}")

    fig, axes = ps.new_figure(nrows, ncols, height_in, scale=scale)

    for ax, metric in zip(axes, metrics):
        _draw_panel(ax, values, metric, method_order, datasets, colors,
                    hatches, labels, use_short, show_values, scale)
    for ax in axes[len(metrics):]:
        ax.axis('off')

    ps.shared_legend(fig, ps.patch_handles(method_order, colors, hatches,
                                           labels, scale),
                     ncol=len(method_order))

    fig.subplots_adjust(left=0.055, right=0.985, top=top, bottom=bottom,
                        wspace=wspace, hspace=hspace)

    return ps.save(fig, output_dir,
                   filename or f'fig_exp2_comparison_{layout}.pdf', scale)


if __name__ == "__main__":
    ps.setup_sleek_style()
    plot_exp2_comparison(HARDCODED, './plots', layout='row')
    plot_exp2_comparison(HARDCODED, './plots', layout='grid')