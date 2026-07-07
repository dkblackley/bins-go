import os
import matplotlib.pyplot as plt
from util import enforce_monotonic_increasing

def plot_bins_ablations(nested_data, output_dir, param_to_vary='bs', target_k=100):
    """
    Generates three separate figures for the Bins method on MSMARCO:
    1. LAN Time vs param
    2. MRR & Recall vs param
    3. Faithfulness & Answer Relevancy vs param
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = 'msmarco'
    method = 'bins'

    if method not in nested_data or dataset not in nested_data[method]:
        print(f"[DEBUG] No {method} data found for {dataset}.")
        return

    # Filter for the specific K-value so we don't mix k=10, 100, 5000 lines together
    runs = [r for r in nested_data[method][dataset] if r['k'] == target_k]

    # Setup the fixed baseline parameter vs the varying parameter
    if param_to_vary == 'bs':
        valid_runs = [r for r in runs if r['dpb'] == 1000 and r['bs'] is not None]
        valid_runs.sort(key=lambda x: x['bs'])
        x_vals = [r['bin_size'] for r in valid_runs]
        xlabel = 'Bin Size Multiplier (bs)'
        file_prefix = f'fig_bins_bs_ablation_k{target_k}'
    elif param_to_vary == 'dpb':
        valid_runs = [r for r in runs if r['bs'] == 1.0 and r['dpb'] is not None]
        valid_runs.sort(key=lambda x: x['dpb'])
        x_vals = [r['dpb'] for r in valid_runs]
        xlabel = 'Documents Per Bin (dpb)'
        file_prefix = f'fig_bins_dpb_ablation_k{target_k}'
    else:
        return

    if not valid_runs:
        print(f"[DEBUG] No valid runs found for Bins {param_to_vary} ablation.")
        return

    # ==========================================
    # Plot 1: LAN Time
    # ==========================================
    fig1, ax1 = plt.subplots()
    y_lan = [r['lan_time'] for r in valid_runs]

    ax1.plot(x_vals, y_lan, marker='o', linestyle='-', color='#009E73', label='LAN Time')
    # ax1.set_xscale('log')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Per-Query LAN Time (seconds)')
    ax1.set_title(f'LAN Time vs {xlabel}')
    ax1.grid(True, zorder=0)

    plt.tight_layout()
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize='small', frameon=False)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_lantime.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig1)

    # ==========================================
    # Plot 2: MRR and Recall
    # ==========================================
    fig2, ax2 = plt.subplots()
    y_mrr = [r['mrr'] for r in valid_runs]
    y_recall = [r['recall'] for r in valid_runs]

    ax2.plot(x_vals, y_mrr, marker='o', linestyle='-', color='#009E73', label='MRR')
    ax2.plot(x_vals, y_recall, marker='^', linestyle='--', color='#E69F00', label='Recall')

    # ax2.set_xscale('log')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Score')
    ax2.set_title(f'MRR & Recall vs {xlabel}')
    ax2.grid(True, zorder=0)

    plt.tight_layout()
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize='small', frameon=False)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_mrr_recall.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig2)

    # ==========================================
    # Plot 3: Faithfulness and Answer Relevancy
    # ==========================================
    fig3, ax3 = plt.subplots()
    y_faith = [r['faithfulness'] for r in valid_runs]
    y_rel = [r['answer_relevancy'] for r in valid_runs]

    ax3.plot(x_vals, y_faith, marker='s', linestyle='-', color='#56B4E9', label='Faithfulness')
    ax3.plot(x_vals, y_rel, marker='D', linestyle='--', color='#CC79A7', label='Answer Relevancy')

    # ax3.set_xscale('log')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('Score')
    ax3.set_title(f'Quality Metrics vs {xlabel}')
    ax3.grid(True, zorder=0)

    plt.tight_layout()
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize='small', frameon=False)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_quality.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig3)