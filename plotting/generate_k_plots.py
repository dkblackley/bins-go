import matplotlib.pyplot as plt
import os
from util import enforce_monotonic_increasing


def generate_k_plots(extended_data, output_dir, method_order, colors, target_configs):
    """
    Generates three distinct figures comparing the methods across Network Time (LAN/WAN),
    ensuring all plotted lines are monotonically increasing.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ('mrr', 'MRR Score', 'fig1_mrr_vs_networktime.pdf'),
        ('faithfulness', 'Faithfulness Score', 'fig2_faithfulness_vs_networktime.pdf'),
        ('answer_relevancy', 'Answer Relevancy Score', 'fig3_ansrel_vs_networktime.pdf')
    ]

    for y_key, y_label, filename in metrics:
        fig, ax = plt.subplots()

        for method in method_order:
            method_runs = [
                run for run in extended_data
                if run['method'] == method
                   and run['config'] == target_configs.get(method, run['config'])
                   and run['dataset'] == "msmarco"
            ]

            if not method_runs:
                continue

            # Check for missing/zero data before filtering
            for run in method_runs:
                if run[y_key] == 0:
                    print(
                        f"[DEBUG - Zero Value] {method.upper()} on msmarco (k={run['k']}, config={run['config']}) has 0.0 for {y_key}.")
                if run['wan_time'] == 0 or run['lan_time'] == 0:
                    print(
                        f"[DEBUG - Zero Network] {method.upper()} on msmarco (k={run['k']}, config={run['config']}) has 0.0 WAN/LAN Time.")

            # Apply monotonic filter using LAN time
            filtered_lan = enforce_monotonic_increasing(method_runs, 'lan_time', y_key)
            if filtered_lan:
                x_lan = [run['lan_time'] for run in filtered_lan]
                y_lan = [run[y_key] for run in filtered_lan]
                ax.plot(x_lan, y_lan, marker='s', linestyle=':',
                        label=f"{method.capitalize()} (LAN)", color=colors[method], zorder=3)

            # Apply monotonic filter using WAN time
            filtered_wan = enforce_monotonic_increasing(method_runs, 'wan_time', y_key)
            if filtered_wan:
                x_wan = [run['wan_time'] for run in filtered_wan]
                y_wan = [run[y_key] for run in filtered_wan]
                ax.plot(x_wan, y_wan, marker='o', linestyle='--',
                        label=f"{method.capitalize()} (WAN)", color=colors[method], zorder=3)

        # Log scale is required since LAN and WAN times are an order of magnitude apart
        ax.set_xscale('log')
        ax.set_xlabel('Per-Query Network Time (seconds)')
        ax.set_ylabel(y_label)
        ax.grid(True, zorder=0)

        # Position legend carefully since there are now 2x the lines
        ax.legend(fontsize='small')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), format='pdf')
        plt.close(fig)