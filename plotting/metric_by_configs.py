import os
import matplotlib.pyplot as plt
from util import enforce_monotonic_increasing


def plot_metric_vs_network_time(all_runs, output_dir, dataset, method_order, colors, target_configs, metric_key='mrr',
                                metric_label='MRR Score'):
    """
    Plots a specific metric (e.g., MRR) vs Network Time for a single dataset.
    Solid line = LAN Time. Dashed line = WAN Time.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    for method in method_order:
        method_runs = [
            run for run in all_runs
            if run['method'] == method
               and run['dataset'] == dataset
               and run['config'] == target_configs.get(method, run['config'])
        ]

        if not method_runs:
            continue

        # --- LAN Line (Solid) ---
        # filtered_lan = enforce_monotonic_increasing(method_runs, 'lan_time', metric_key)
        filtered_lan = method_runs
        if filtered_lan:
            x_lan = [r['lan_time'] for r in filtered_lan]
            y_lan = [r[metric_key] for r in filtered_lan]
            ax.plot(x_lan, y_lan, marker='o', linestyle='-',
                    label=f"{method.capitalize()} (LAN)", color=colors.get(method), zorder=3)

        # --- WAN Line (Dashed) ---
        filtered_wan = enforce_monotonic_increasing(method_runs, 'wan_time', metric_key)
        if filtered_wan:
            x_wan = [r['wan_time'] for r in filtered_wan]
            y_wan = [r[metric_key] for r in filtered_wan]
            ax.plot(x_wan, y_wan, marker='s', linestyle='--',
                    label=f"{method.capitalize()} (WAN)", color=colors.get(method), zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel('Per-Query Network Time (seconds)')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} vs Network Time ({dataset.capitalize()})')
    ax.grid(True, zorder=0)

    # Position legend carefully since there are multiple lines
    ax.legend(fontsize='small', loc='best')

    plt.tight_layout()
    filename = f'fig_{dataset}_{metric_key}_vs_networktime.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf')
    plt.close(fig)


def plot_quality_vs_time(all_runs, output_dir, dataset, method_order, colors, target_configs, time_key='total_time',
                         time_label='Per-Query Total Time (s)'):
    """
    Plots Faithfulness (Solid) and Answer Relevancy (Dashed) vs a chosen time metric for a single dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    for method in method_order:
        method_runs = [
            run for run in all_runs
            if run['method'] == method
               and run['dataset'] == dataset
               and run['config'] == target_configs.get(method, run['config'])
        ]

        if not method_runs:
            continue

        # --- Faithfulness Line (Solid) ---
        filtered_faith = enforce_monotonic_increasing(method_runs, time_key, 'faithfulness')
        if filtered_faith:
            x_faith = [r[time_key] for r in filtered_faith]
            y_faith = [r['faithfulness'] for r in filtered_faith]
            ax.plot(x_faith, y_faith, marker='o', linestyle='-',
                    label=f"{method.capitalize()} (Faithfulness)", color=colors.get(method), zorder=3)

        # --- Answer Relevancy Line (Dashed) ---
        filtered_rel = enforce_monotonic_increasing(method_runs, time_key, 'answer_relevancy')
        if filtered_rel:
            x_rel = [r[time_key] for r in filtered_rel]
            y_rel = [r['answer_relevancy'] for r in filtered_rel]
            ax.plot(x_rel, y_rel, marker='^', linestyle='--',
                    label=f"{method.capitalize()} (Ans. Rel.)", color=colors.get(method), zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel(time_label)
    ax.set_ylabel('Score')
    ax.set_title(f'Quality Metrics vs Time ({dataset.capitalize()})')
    ax.grid(True, zorder=0)

    ax.legend(fontsize='small', loc='best')

    plt.tight_layout()
    filename = f'fig_{dataset}_quality_vs_{time_key}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf')
    plt.close(fig)