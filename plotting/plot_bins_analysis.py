import os
import matplotlib.pyplot as plt

from main import SUBPLOT_SIZE
from util import format_ticks


def plot_bins_parameters(all_runs, output_dir, param='dpb'):
    """
    Plots a 1x3 subplot grid for 'bins' method on MSMARCO (k=50).
    param can be 'dpb' (Docs Per Bin) or 'bs' (Bin Size).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Isolate bins, msmarco, k=50 (the baseline standard found in your results)
    target_runs = [r for r in all_runs if r['method'] == 'bins' and r['dataset'] == 'msmarco' and r['k'] == 50]

    if param == 'dpb':
        target_runs = [r for r in target_runs if r['bs'] == 0.1 and r['dpb'] is not None]
        target_runs.sort(key=lambda x: x['dpb'])
        x_vals = [r['dpb']*8841823 for r in target_runs]
        xlabel = 'Documents Per Bin (dpb)'
        filename = 'fig_bins_dpb_analysis.pdf'
    else:
        target_runs = [r for r in target_runs if r['dpb'] == 1000 and r['bs'] is not None]
        target_runs.sort(key=lambda x: x['bs'])
        x_vals = [int(r['bs']*8841823) for r in target_runs]
        xlabel = 'DB Size'
        filename = 'fig_bins_bs_analysis.pdf'

    if not target_runs: return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=SUBPLOT_SIZE)
    fig.suptitle("Changing Bin Size Effect on MSMarco")

    labels = [format_ticks(k) for k in x_vals]

    for ax in [ax1, ax2, ax3]:
        ax.set_xscale('log')
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels)
        ax.minorticks_off()

    # Graph 1: MRR
    ax1.plot(x_vals, [r['mrr'] for r in target_runs], marker='o', color='#009E73', linewidth=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('MRR')
    #ax1.set_title('MRR vs ' + xlabel)
    ax1.grid(True)

    # Graph 2: Faithfulness and Answer Relevancy
    ax2.plot(x_vals, [r['faithfulness'] for r in target_runs], marker='s', label='Faithfulness', color='#56B4E9')
    ax2.plot(x_vals, [r['answer_relevancy'] for r in target_runs], marker='^', label='Answer Relevancy', color='#E69F00')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Score')
    #ax2.set_title('Quality vs ' + xlabel)
    ax2.legend()
    ax2.grid(True)

    # Graph 3: Communication Cost & WAN/LAN Time (Using TwinX for distinct scales)
    ax3.plot(x_vals, [r['wan_time'] for r in target_runs], marker='v', label='WAN Time', color='red', linestyle='--')
    ax3.plot(x_vals, [r['lan_time'] for r in target_runs], marker='v', label='LAN Time', color='brown', linestyle='--')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('Per-Query Time (s)')
    ax3.grid(True)

    # ax3_twin = ax3.twinx()
    # ax3_twin.plot(x_vals, [r['comm_cost'] for r in target_runs], marker='D', label='Comm Cost', color='purple')
    # ax3_twin.set_ylabel('Comm Cost (KB)', color='purple')
    # ax3_twin.tick_params(axis='y', labelcolor='purple')
    #
    # # Merge Legends securely
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    # lines_2, labels_2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_1 , labels_1, loc='upper left')
    #ax3.set_title('Overheads vs ' + xlabel)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), format='pdf')
    plt.close(fig)