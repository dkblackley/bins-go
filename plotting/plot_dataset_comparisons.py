import os
import numpy as np
import matplotlib.pyplot as plt

def plot_cross_dataset_metrics(all_runs, output_dir, target_configs, colors, methods):
    os.makedirs(output_dir, exist_ok=True)
    datasets = ['scifact', 'trec-covid', 'msmarco']

    # Filter for standard k=50 baseline targets
    baseline_runs = [r for r in all_runs if r['k'] == 100 and r['config'] == target_configs.get(r['method']) and r['dataset'] in datasets]

    def get_matrix(metric_key):
        matrix = []
        for m in methods:
            row = []
            for d in datasets:
                val = next((r[metric_key] for r in baseline_runs if r['method'] == m and r['dataset'] == d), 0)
                row.append(val)
            matrix.append(row)
        return matrix

    x = np.arange(len(datasets))
    width = 0.25

    # ---------------------------------------------------------
    # Figure 1: Quality Metrics (1x3 Grid)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3,  sharey=True)
    metrics = [('mrr', 'MRR'), ('faithfulness', 'Faithfulness'), ('answer_relevancy', 'Answer Relevancy')]

    for idx, (metric_key, title) in enumerate(metrics):
        ax = axes1[idx]
        matrix = get_matrix(metric_key)

        for m_idx, m in enumerate(methods):
            offset = width * m_idx - width
            ax.bar(x + offset, matrix[m_idx], width, label=m.capitalize(), color=colors.get(m))

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in datasets])
        ax.set_title(title)
        ax.grid(axis='y', zorder=0)
        if idx == 0:
            ax.set_ylabel('Score')
            ax.legend()

    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'fig_dataset_quality.pdf'), format='pdf')
    plt.close(fig1)

    # ---------------------------------------------------------
    # Figure 2: Runtimes and Network (1x2 Grid)
    # ---------------------------------------------------------
    fig2, (ax_rt, ax_net) = plt.subplots(1, 2)

    # Total Answer Time
    matrix_time = get_matrix('total_time')
    for m_idx, m in enumerate(methods):
        ax_rt.bar(x + (width * m_idx - width), matrix_time[m_idx], width, label=m.capitalize(), color=colors.get(m))
    ax_rt.set_xticks(x)
    ax_rt.set_xticklabels([d.capitalize() for d in datasets])
    ax_rt.set_title('Total Answer Time by Dataset')
    ax_rt.set_ylabel('Time (s)')
    ax_rt.grid(axis='y')
    ax_rt.legend()

    # Total WAN Time
    matrix_wan = get_matrix('wan_time')
    for m_idx, m in enumerate(methods):
        ax_net.bar(x + (width * m_idx - width), matrix_wan[m_idx], width, label=m.capitalize(), color=colors.get(m))
    ax_net.set_xticks(x)
    ax_net.set_xticklabels([d.capitalize() for d in datasets])
    ax_net.set_title('Avg WAN Time per Query by Dataset')
    ax_net.set_ylabel('Time (s)')
    ax_net.grid(axis='y')

    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'fig_dataset_runtime.pdf'), format='pdf')
    plt.close(fig2)