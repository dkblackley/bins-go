import os
import matplotlib.pyplot as plt

def plot_mrr_vs_time(all_runs, output_dir, colors):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    # Filter purely for MSMARCO
    msmarco_runs = [r for r in all_runs if r['dataset'] == 'msmarco']

    for method in ['tree', 'bins', 'pacmann']:
        method_runs = [r for r in msmarco_runs if r['method'] == method]
        if not method_runs:
            continue

        # Sorting by total time traces the configuration curve smoothly
        method_runs.sort(key=lambda x: x['total_time'])

        x_vals = [r['total_time'] for r in method_runs]
        y_vals = [r['mrr'] for r in method_runs]

        # Use scatter + line alpha to keep visual noise low while showing the curve
        ax.plot(x_vals, y_vals, marker='o', label=method.capitalize(), color=colors.get(method), zorder=3, alpha=0.8)

    ax.set_xlabel('Total Answer Time (seconds)')
    ax.set_ylabel('MRR Score')
    ax.set_title('Config Tradeoffs on MSMARCO: MRR vs Time')
    ax.grid(True, zorder=0)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_config_tradeoffs.pdf'), format='pdf')
    plt.close(fig)