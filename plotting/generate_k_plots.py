import matplotlib.pyplot as plt
import os


def generate_k_plots(extended_data, output_dir, method_order, colors, target_configs):
    """
    Generates three distinct figures comparing the methods across different k values.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Figure 1: MRR vs k
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots()

    for method in method_order:
        # Filter for the specific method and its target config
        method_runs = [
            run for run in extended_data
            if run['method'] == method and run['config'] == target_configs.get(method, run['config']) and run['dataset'] == "msmarco"
        ]
        if not method_runs:
            continue

        # Sort by k so lines draw sequentially left to right
        method_runs.sort(key=lambda x: x['k'])

        sorted_k = [run['k'] for run in method_runs]
        y_vals = [run['mrr'] for run in method_runs]

        ax1.plot(sorted_k, y_vals, marker='o', label=method.capitalize(), color=colors[method], zorder=3)


    ax1.set_xscale('log')
    ax1.set_xlabel('Retrieved Documents ($k$)')
    all_k = sorted(list(set(run['k'] for run in extended_data)))
    ax1.set_xticks(all_k)
    ax1.set_xticklabels([str(k) for k in all_k])
    ax1.minorticks_off()

    ax1.set_ylabel('MRR Score')
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
        # Filter for the specific method and its target config
        method_runs = [
            run for run in extended_data
            if run['method'] == method and run['config'] == target_configs.get(method, run['config']) and run['dataset'] == "msmarco"
        ]
        if not method_runs:
            continue

        # Sort by k so lines draw sequentially left to right
        method_runs.sort(key=lambda x: x['k'])

        sorted_k = [run['k'] for run in method_runs]
        y_vals = [run['comm_cost'] for run in method_runs]

        ax2.plot(sorted_k, y_vals, marker='o', label=method.capitalize(), color=colors[method], zorder=3)

    ax2.set_xscale('log')
    ax2.set_xlabel('Retrieved Documents ($k$)')
    ax2.set_xticks(all_k)
    ax2.set_xticklabels([str(k) for k in all_k])
    ax2.minorticks_off()
    ax2.set_ylabel('Total data sent (MB)')
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
        # Filter for the specific method and its target config
        method_runs = [
            run for run in extended_data
            if run['method'] == method and run['config'] == target_configs.get(method, run['config']) and run['dataset'] == "msmarco"
        ]
        if not method_runs:
            continue

        # Sort by k so lines draw sequentially left to right
        method_runs.sort(key=lambda x: x['k'])

        sorted_k = [run['k'] for run in method_runs]
        y_vals = [run['total_time'] for run in method_runs]

        ax3.plot(sorted_k, y_vals, marker='o', label=method.capitalize(), color=colors[method], zorder=3)

    ax3.set_xscale('log')
    ax3.set_xlabel('Retrieved Documents ($k$)')
    ax3.set_xticks(all_k)
    ax3.set_xticklabels([str(k) for k in all_k])
    ax3.minorticks_off()
    ax3.set_ylabel('Avg. Time per Query (seconds)')
    ax3.grid(True, zorder=0)
    ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_latency_vs_k.pdf'), format='pdf')
    plt.close(fig3)