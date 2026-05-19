import matplotlib.pyplot as plt
import os


def generate_k_plots(data_store, output_dir, method_order, colors):
    """
    Generates three distinct figures comparing the methods across different k values.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Figure 1: MRR vs k
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots()

    for method in method_order:
        method_data = data_store.get(method, {})
        if not method_data:
            continue

        # Sort by k so lines draw sequentially left to right
        sorted_k = sorted(method_data.keys())
        y_vals = [method_data[k]['mrr'] for k in sorted_k]

        ax1.plot(sorted_k, y_vals, marker='o', label=method.capitalize(), color=colors[method], zorder=3)

    ax1.set_xlabel('Retrieved Documents ($k$)')
    ax1.set_ylabel('MRR Score')
    ax1.set_title('Effectiveness vs. Retrieval Depth')
    ax1.grid(True, zorder=0)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_mrr_vs_k.pdf'), format='pdf')
    plt.close(fig1)

    # ---------------------------------------------------------
    # Figure 2: Communication Cost vs k
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots()

    for method in method_order:
        method_data = data_store.get(method, {})
        if not method_data:
            continue

        sorted_k = sorted(method_data.keys())
        y_vals = [method_data[k]['comm_cost'] for k in sorted_k]

        ax2.plot(sorted_k, y_vals, marker='s', label=method.capitalize(), color=colors[method], zorder=3)

    ax2.set_xlabel('Retrieved Documents ($k$)')
    ax2.set_ylabel('Comm Cost per Batch (KB)')
    ax2.set_title('Bandwidth Efficiency vs. Retrieval Depth')
    ax2.grid(True, zorder=0)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_commcost_vs_k.pdf'), format='pdf')
    plt.close(fig2)

    # ---------------------------------------------------------
    # Figure 3: Time Per Query vs k
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots()

    for method in method_order:
        method_data = data_store.get(method, {})
        if not method_data:
            continue

        sorted_k = sorted(method_data.keys())
        y_vals = [method_data[k]['time_per_query'] for k in sorted_k]

        ax3.plot(sorted_k, y_vals, marker='D', label=method.capitalize(), color=colors[method], zorder=3)

    ax3.set_xlabel('Retrieved Documents ($k$)')
    ax3.set_ylabel('Avg. Time per Query (seconds)')
    ax3.set_title('Computational Latency vs. Retrieval Depth')
    ax3.grid(True, zorder=0)
    ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_latency_vs_k.pdf'), format='pdf')
    plt.close(fig3)