import os
import matplotlib.pyplot as plt

def plot_wan_lan_vs_k(all_runs, output_dir, target_configs, colors, methods):
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax_wan, ax_lan) = plt.subplots(1, 2)

    for method in methods:
        target_config = target_configs.get(method)
        # Filter for baseline config strictly on MSMARCO
        method_runs = [r for r in all_runs if r['method'] == method and r['dataset'] == 'msmarco' and r['config'] == target_config]
        if not method_runs: continue

        method_runs.sort(key=lambda x: x['k'])
        k_vals = [r['k'] for r in method_runs]
        wan_vals = [r['wan_time'] for r in method_runs]
        lan_vals = [r['lan_time'] for r in method_runs]

        ax_wan.plot(k_vals, wan_vals, marker='o', label=method.capitalize(), color=colors.get(method))
        ax_lan.plot(k_vals, lan_vals, marker='s', label=method.capitalize(), color=colors.get(method))

    ax_wan.set_xlabel('Retrieved Documents ($k$)')
    ax_wan.set_ylabel('Per-Query WAN Time (s)')
    ax_wan.set_title('WAN Overhead vs. Retrieval Depth')
    ax_wan.grid(True)
    ax_wan.legend()

    ax_lan.set_xlabel('Retrieved Documents ($k$)')
    ax_lan.set_ylabel('Per-Query LAN Time (s)')
    ax_lan.set_title('LAN Overhead vs. Retrieval Depth')
    ax_lan.grid(True)
    ax_lan.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_network_vs_k.pdf'), format='pdf')
    plt.close(fig)