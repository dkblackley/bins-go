import os
import matplotlib.pyplot as plt
from main import SUBPLOT_SIZE
from util import format_ticks

def plot_bins_parameters(all_runs, output_dir, param='dpb'):
    """
    Plots a 1x3 subplot grid for 'bins' method on MSMARCO.
    """
    os.makedirs(output_dir, exist_ok=True)

    target_runs = [r for r in all_runs if r['method'] == 'bins' and r['dataset'] == 'msmarco' and r['k'] == 100]

    if param == 'dpb':
        target_runs = [r for r in target_runs if r['bs'] == 1.0 and r['dpb'] is not None]
        target_runs.sort(key=lambda x: x['dpb'])
        x_vals = [r['dpb'] for r in target_runs]
        xlabel = 'Documents Per Bin (dpb)'
        filename = 'fig_bins_dpb_analysis.pdf'
    else:
        target_runs = [r for r in target_runs if r['dpb'] == 1000 and r['bs'] is not None]
        target_runs.sort(key=lambda x: x['bs'])
        # Multiply ratio by standard doc count to show actual DB capacity
        x_vals = [r['bs'] * 8841823 for r in target_runs]
        xlabel = 'DB Capacity (Number of Documents)'
        filename = 'fig_bins_bs_analysis.pdf'

    if not target_runs: return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=SUBPLOT_SIZE)
    fig.suptitle(f"Changing {xlabel} Effect on MSMarco")

    # Only show ticks for the actual data points to prevent squishing
    for ax in [ax1, ax2, ax3]:
        # ax.set_xscale('log')
        ax.set_xticks(x_vals)
        ax.set_xticklabels([format_ticks(x) for x in x_vals])
        ax.minorticks_off()

    # Graph 1: MRR
    ax1.plot(x_vals, [r['mrr'] for r in target_runs], marker='o', color='#009E73', linewidth=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('MRR')
    ax1.grid(True)

    # Graph 2: Faithfulness and Answer Relevancy
    ax2.plot(x_vals, [r['faithfulness'] for r in target_runs], marker='s', label='Faithfulness', color='#56B4E9')
    ax2.plot(x_vals, [r['answer_relevancy'] for r in target_runs], marker='^', label='Answer Relevancy', color='#E69F00')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)

    # Graph 3: WAN/LAN Time vs Total Time
    ax3.plot(x_vals, [r['wan_time'] for r in target_runs], marker='v', label='WAN Time (s)', color='red', linestyle='--')
    ax3.plot(x_vals, [r['lan_time'] for r in target_runs], marker='v', label='LAN Time (s)', color='brown', linestyle='--')
    ax3.plot(x_vals, [r['total_time'] for r in target_runs], marker='o', label='Total Time (s)', color='black')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('Per-Query Time (s)')
    ax3.grid(True)

    lines, labels = ax3.get_legend_handles_labels()
    ax3.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), format='pdf')
    plt.close(fig)