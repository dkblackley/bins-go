import os
import matplotlib.pyplot as plt
from util import enforce_monotonic_increasing


def plot_metric_vs_lan_time(nested_data, output_dir, dataset, method_order, colors, target_configs, metric_key='mrr',
                            metric_label='MRR Score', use_log_scale=False, enforce_monotonic=False):
    """
    Plots a specific metric (e.g., MRR) vs LAN Time for a single dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Reverted to default figure size to maintain your original aspect ratio
    fig, ax = plt.subplots()

    for method in method_order:
        if method not in nested_data or dataset not in nested_data[method]:
            continue

        method_runs = [
            run for run in nested_data[method][dataset]
            if run['config'] == target_configs.get(method, run['config'])
        ]

        if not method_runs:
            continue

        method_runs.sort(key=lambda x: x['lan_time'])

        if enforce_monotonic:
            filtered_lan = enforce_monotonic_increasing(method_runs, 'lan_time', metric_key)
        else:
            filtered_lan = method_runs

        if filtered_lan:
            x_lan = [r['lan_time'] for r in filtered_lan]
            y_lan = [r[metric_key] for r in filtered_lan]
            ax.plot(x_lan, y_lan, marker='o', linestyle='-',
                    label=method.capitalize(), color=colors.get(method), zorder=3)

    if use_log_scale:
        ax.set_xscale('log')

    ax.set_xlabel('Per-Query LAN Time (seconds)')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} vs LAN Time ({dataset.capitalize()})')
    ax.grid(True, zorder=0)

    # 1. Run tight_layout FIRST so it perfectly formats the graph area and labels
    plt.tight_layout()

    # 2. Place the legend completely BELOW the X-axis label
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),  # Negative Y-value pushes it down
        ncol=3,
        fontsize='small',
        frameon=False
    )

    # 3. bbox_inches='tight' expands the PDF canvas downwards to capture the legend
    filename = f'fig_{dataset}_{metric_key}_vs_lantime.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_metric_vs_wan_time(nested_data, output_dir, dataset, method_order, colors, target_configs, metric_key='mrr',
                            metric_label='MRR Score', use_log_scale=False, enforce_monotonic=False):
    """
    Plots a specific metric (e.g., MRR) vs WAN Time for a single dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    for method in method_order:
        if method not in nested_data or dataset not in nested_data[method]:
            continue

        method_runs = [
            run for run in nested_data[method][dataset]
            if run['config'] == target_configs.get(method, run['config'])
        ]

        if not method_runs:
            continue

        method_runs.sort(key=lambda x: x['wan_time'])

        if enforce_monotonic:
            filtered_wan = enforce_monotonic_increasing(method_runs, 'wan_time', metric_key)
        else:
            filtered_wan = method_runs

        if filtered_wan:
            x_wan = [r['wan_time'] for r in filtered_wan]
            y_wan = [r[metric_key] for r in filtered_wan]
            ax.plot(x_wan, y_wan, marker='o', linestyle='-',
                    label=method.capitalize(), color=colors.get(method), zorder=3)

    if use_log_scale:
        ax.set_xscale('log')

    ax.set_xlabel('Per-Query WAN Time (seconds)')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} vs WAN Time ({dataset.capitalize()})')
    ax.grid(True, zorder=0)

    plt.tight_layout()

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        fontsize='small',
        frameon=False
    )

    filename = f'fig_{dataset}_{metric_key}_vs_wantime.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_metric_vs_total_time(nested_data, output_dir, dataset, method_order, colors, target_configs, metric_key='mrr',
                              metric_label='MRR Score', use_log_scale=False, enforce_monotonic=False):
    """
    Plots a specific metric (e.g., MRR) vs Total Answer Time for a single dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    for method in method_order:
        if method not in nested_data or dataset not in nested_data[method]:
            continue

        method_runs = [
            run for run in nested_data[method][dataset]
            if run['config'] == target_configs.get(method, run['config'])
        ]

        if not method_runs:
            continue

        method_runs.sort(key=lambda x: x['total_time'])

        if enforce_monotonic:
            filtered_total = enforce_monotonic_increasing(method_runs, 'total_time', metric_key)
        else:
            filtered_total = method_runs

        if filtered_total:
            x_total = [r['total_time'] for r in filtered_total]
            y_total = [r[metric_key] for r in filtered_total]
            ax.plot(x_total, y_total, marker='o', linestyle='-',
                    label=method.capitalize(), color=colors.get(method), zorder=3)

    if use_log_scale:
        ax.set_xscale('log')

    ax.set_xlabel('Per-Query Total Time (seconds)')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} vs Total Time ({dataset.capitalize()})')
    ax.grid(True, zorder=0)

    plt.tight_layout()

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        fontsize='small',
        frameon=False
    )

    filename = f'fig_{dataset}_{metric_key}_vs_totaltime.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_quality_vs_time(nested_data, output_dir, dataset, method_order, colors, target_configs, time_key='total_time',
                         time_label='Per-Query Total Time (s)', use_log_scale=False, enforce_monotonic=False):
    """
    Plots Faithfulness (Solid) and Answer Relevancy (Dashed) vs a chosen time metric for a single dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    for method in method_order:
        if method not in nested_data or dataset not in nested_data[method]:
            continue

        method_runs = [
            run for run in nested_data[method][dataset]
            if run['config'] == target_configs.get(method, run['config'])
        ]

        if not method_runs:
            continue

        method_runs.sort(key=lambda x: x[time_key])

        # --- Faithfulness Line (Solid) ---
        if enforce_monotonic:
            filtered_faith = enforce_monotonic_increasing(method_runs, time_key, 'faithfulness')
        else:
            filtered_faith = method_runs

        if filtered_faith:
            x_faith = [r[time_key] for r in filtered_faith]
            y_faith = [r['faithfulness'] for r in filtered_faith]
            ax.plot(x_faith, y_faith, marker='o', linestyle='-',
                    label=f"{method.capitalize()} (Faith.)", color=colors.get(method), zorder=3)

        # --- Answer Relevancy Line (Dashed) ---
        if enforce_monotonic:
            filtered_rel = enforce_monotonic_increasing(method_runs, time_key, 'answer_relevancy')
        else:
            filtered_rel = method_runs

        if filtered_rel:
            x_rel = [r[time_key] for r in filtered_rel]
            y_rel = [r['answer_relevancy'] for r in filtered_rel]
            ax.plot(x_rel, y_rel, marker='^', linestyle='--',
                    label=f"{method.capitalize()} (Ans. Rel.)", color=colors.get(method), zorder=3)

    if use_log_scale:
        ax.set_xscale('log')

    ax.set_xlabel(time_label)
    ax.set_ylabel('Score')
    ax.set_title(f'Quality Metrics vs Time ({dataset.capitalize()})')
    ax.grid(True, zorder=0)

    plt.tight_layout()

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=2, # THIS IS CAUSING THE ISSUE
        fontsize='small',
        frameon=False
    )

    filename = f'fig_{dataset}_quality_vs_{time_key}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)