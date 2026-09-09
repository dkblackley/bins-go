"""
BM25-Tree ablation: MRR, runtime and communication cost against the PIR tree
branch factor r, one line per candidate count k, plus the share of per-query
latency spent in each protocol round. Replaces the 2-by-2 `groupplot`.

    plot_tree_ablation(sweep, rounds, OUTPUT_DIR)

`sweep['x']` plus `sweep[series_label][metric] = [...]`; `rounds` is a list of
{'label', 'shares'} with one share per round. Build the sweep from your loader
with sweep_from_nested(nested_data), or pass the HARDCODED constants.
"""

import numpy as np

import util as ps

METRICS = [
    # Limits are round numbers so the three tick labels are exact rather than
    # rounded versions of an arbitrary bound.
    {'key': 'mrr',  'title': 'MRR@10',      'ylim': (0.28, 0.34)},
    {'key': 'time', 'title': 'Runtime (s)', 'ylim': (0.25, 0.75)},
    {'key': 'comm', 'title': 'Comm. (KB)',  'ylim': (0.10, 10.50)},
]

RUN_KEYS = {'mrr': 'mrr', 'time': 'total_time', 'comm': 'comm_cost'}

# One colour + marker per k. Okabe-Ito again, ordered so adjacent k values are
# easy to tell apart.
SERIES_COLORS = ['#0072B2', '#CC79A7', '#009E73', '#E69F00']
SERIES_MARKERS = ['o', 's', '^', 'D']

ROUND_LABELS = ['Round 1', 'Round 2', 'Round 3']


# ==========================================
# DATA
# ==========================================

def sweep_from_nested(nested_data, dataset='msmarco', method='tree',
                      k_values=(32, 64, 128, 256), target_k=10):
    """
    Branch factor sweep out of the loader's nested_data. Config strings look
    like 'b64_r128', so b is the candidate count and r the branch factor.
    """
    import re
    runs = [r for r in nested_data.get(method, {}).get(dataset, [])
            if target_k is None or r['k'] == target_k]

    parsed = []
    for r in runs:
        b = re.search(r'b(\d+)', r['config'])
        rr = re.search(r'r(\d+)', r['config'])
        if not (b and rr):
            print(f"[WARN] cannot parse b/r from config {r['config']!r}")
            continue
        parsed.append((int(b.group(1)), int(rr.group(1)), r))

    sweep = {}
    x_ref = None
    for b in k_values:
        rows = sorted([p for p in parsed if p[0] == b], key=lambda p: p[1])
        if not rows:
            print(f"[WARN] no {method} runs on {dataset} with b={b}")
            continue
        xs = [p[1] for p in rows]
        x_ref = x_ref or xs
        sweep['x'] = xs
        sweep[f'k={b} (docs={b * 8})'] = {
            m: [p[2][RUN_KEYS[m]] for p in rows] for m in RUN_KEYS
        }
    return sweep


# Numbers from the pgfplots version, so this module runs standalone.
HARDCODED_SWEEP = {
    'x': [8, 32, 128],
    'k=32 (docs=256)':   {'mrr': [0.2894, 0.2892, 0.2887],
                          'time': [0.6746, 0.3854, 0.2903],
                          'comm': [0.16, 0.18, 0.25]},
    'k=64 (docs=512)':   {'mrr': [0.3015, 0.3025, 0.3021],
                          'time': [0.6889, 0.3978, 0.3118],
                          'comm': [0.25, 0.25, 0.32]},
    'k=128 (docs=1024)': {'mrr': [0.3078, 0.3105, 0.3100],
                          'time': [0.6907, 0.4068, 0.3125],
                          'comm': [0.35, 0.36, 0.40]},
    'k=256 (docs=2048)': {'mrr': [0.3105, 0.3124, 0.3119],
                          'time': [0.7055, 0.4212, 0.3350],
                          'comm': [0.45, 0.44, 0.48]},
}

# 'group' is drawn once under each run of consecutive bars that share it, so
# the dataset name gets the width of a whole pair instead of one 0.6in column.
HARDCODED_ROUNDS = [
    {'group': 'MS MARCO', 'label': 'b16r8',   'shares': [75.6, 19.5, 4.9]},
    {'group': 'MS MARCO', 'label': 'b32r128', 'shares': [32.2, 54.9, 13.0]},
    {'group': 'SciFact',  'label': 'b16r8',   'shares': [16.8, 57.3, 25.9]},
    {'group': 'SciFact',  'label': 'b32r128', 'shares': [19.8, 51.8, 28.4]},
]


# ==========================================
# PLOTTING
# ==========================================

def _draw_grouped_rounds(ax, rounds, colors, hatches, scale, num_stages=3):
    """Grouped bar chart for stage latencies, stages on X-axis, configs in legend."""
    stages = [f'Stage {i + 1}' for i in range(num_stages)]
    xs = np.arange(num_stages, dtype=float)

    n_configs = len(rounds)
    bar_width = 0.8 / n_configs

    for i, r in enumerate(rounds):
        shares = r['shares']
        if num_stages == 2:
            # Merge round 1 and 2 into stage 1, round 3 becomes stage 2
            plot_shares = [shares[0] + shares[1], shares[2]]
        else:
            plot_shares = shares[:3]

        # Offset bars based on index to group them around the X-axis tick
        offset = (i - n_configs / 2 + 0.5) * bar_width
        label = f"{r.get('group', '')} {r['label']}".strip()

        ax.bar(xs + offset, plot_shares, width=bar_width, label=label,
               color=colors[i % len(colors)], edgecolor='#333333',
               linewidth=0.55 * scale, hatch=hatches[i % len(hatches)],
               zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(stages, fontsize=(ps.FONT_PT) * scale)
    ax.set_ylabel('Latency share (%)', labelpad=4 * scale)

    ps.style_axes(ax, title=None, ylim=(0, 100), xgrid=False)



def plot_tree_ablation(sweep, rounds, output_dir, num_stages=3,
                       filename=None, metrics=METRICS,
                       series_colors=SERIES_COLORS, series_markers=SERIES_MARKERS,
                       round_colors=ps.ROUND_COLORS, round_hatches=ps.ROUND_HATCHES,
                       xlabel='Branch factor $r$', height_in=3.20, scale=ps.SCALE):
    """
    2x2: three branch-factor panels plus the round breakdown.
    Supports toggle between 2-stage (merges stage 1+2, renames to Block size=doc size)
    and 3-stage (renames to Block size=k).
    """
    import re
    if filename is None:
        filename = f'fig_tree_ablation_{num_stages}stage.pdf'

    # 1. Format the sweep keys dynamically based on num_stages
    formatted_sweep = {'x': sweep.get('x', [])}
    for k, v in sweep.items():
        if k == 'x':
            continue

        match = re.search(r'k=(\d+)\s*\(docs=(\d+)\)', k)
        if match:
            k_val, doc_val = match.groups()
            new_key = f"Block size={doc_val}" if num_stages == 2 else f"Block size={k_val}"
            formatted_sweep[new_key] = v
        else:
            formatted_sweep[k] = v

    fig, axes = ps.new_figure(2, 2, height_in, scale=scale)
    xs = formatted_sweep['x']
    series = [k for k in formatted_sweep if k != 'x']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for j, name in enumerate(series):
            ys = formatted_sweep[name].get(metric['key'])
            if not ys:
                continue
            ax.plot(xs, ys, marker=series_markers[j % len(series_markers)],
                    linestyle='-', color=series_colors[j % len(series_colors)],
                    label=name, zorder=3)

        ps.log_xticks(ax, xs, base=2)

        # 2. Update styling: Y-axis manually labeled, per-plot title removed
        ps.style_axes(ax, title=None, ylabel=None, ylim=metric['ylim'],
                      xlabel=xlabel if idx >= 2 else None, xgrid=True, yfmt='{:.2f}')
        ax.set_ylabel(metric['title'], labelpad=4 * scale)

    # 3. Draw the grouped round latencies
    _draw_grouped_rounds(axes[3], rounds, round_colors, round_hatches, scale, num_stages=num_stages)

    # 4. Shared Legend Placement
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=line_handles, labels=line_labels, loc='upper center',
               bbox_to_anchor=(0.28, 1.05), ncol=2, frameon=False,
               handlelength=1.6, handletextpad=0.45, columnspacing=1.4)

    bar_handles, bar_labels = axes[3].get_legend_handles_labels()
    fig.legend(handles=bar_handles, labels=bar_labels, loc='upper center',
               bbox_to_anchor=(0.80, 1.05), ncol=2, frameon=False,
               handlelength=1.2, handleheight=1.0, handletextpad=0.45,
               columnspacing=1.2)

    # 5. Tweak spacing to accommodate the new text locations
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.12,
                        wspace=0.40, hspace=0.40)

    return ps.save(fig, output_dir, filename, scale)

if __name__ == "__main__":
    ps.setup_sleek_style()
    plot_tree_ablation(HARDCODED_SWEEP, HARDCODED_ROUNDS, './plots')