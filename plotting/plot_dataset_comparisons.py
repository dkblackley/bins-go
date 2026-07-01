import os
import matplotlib.pyplot as plt


def plot_cross_dataset_metrics(all_runs, output_dir, target_configs, colors, methods):
    os.makedirs(output_dir, exist_ok=True)
    datasets = ['scifact', 'trec-covid', 'msmarco']

    baseline_runs = [r for r in all_runs if
                     r['k'] == 100 and r['config'] == target_configs.get(r['method']) and r['dataset'] in datasets]

    metrics = [
        ('mrr', 'MRR Score', 'fig_dataset_mrr.pdf'),
        ('faithfulness', 'Faithfulness Score', 'fig_dataset_faithfulness.pdf'),
        ('answer_relevancy', 'Answer Relevancy Score', 'fig_dataset_ansrel.pdf'),
        ('total_time', 'Total Answer Time (s)', 'fig_dataset_total_time.pdf'),
        ('wan_time', 'Avg WAN Time (s)', 'fig_dataset_wan_time.pdf')
    ]

    for metric_key, y_label, filename in metrics:
        fig, ax = plt.subplots()

        for method in methods:
            method_runs = [r for r in baseline_runs if r['method'] == method]

            # Sort by database size to draw a clean line from smallest DB to largest DB
            method_runs.sort(key=lambda x: x['db_size_mb'])

            if not method_runs:
                continue

            x_vals = [r['db_size_mb'] for r in method_runs]
            y_vals = [r[metric_key] for r in method_runs]

            # Log missing data to terminal
            for run in method_runs:
                if run[metric_key] == 0:
                    print(f"[DEBUG - Missing Data] {method.upper()} on {run['dataset']} has 0.0 for {metric_key}.")

            ax.plot(x_vals, y_vals, marker='o', label=method.capitalize(), color=colors.get(method), zorder=3)

            # Annotate the points with the dataset names so you still know which is which
            for r in method_runs:
                ax.annotate(r['dataset'].capitalize(),
                            (r['db_size_mb'], r[metric_key]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=10)

        ax.set_xscale('log')
        ax.set_xlabel('Database Size (MB)')
        ax.set_ylabel(y_label)
        ax.grid(True, zorder=0)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), format='pdf')
        plt.close(fig)



def plot_cross_dataset_metrics_hardcoded(all_runs, output_dir, target_configs, colors, methods):
    os.makedirs(output_dir, exist_ok=True)
    datasets = ['scifact', 'trec-covid', 'msmarco']

    # Using known dataset document counts since JSON NumDocs is null
    dataset_sizes = {
        'scifact': 5183,
        'trec-covid': 171332,
        'msmarco': 8841823
    }

    baseline_runs = [r for r in all_runs if
                     r['k'] == 100 and r['config'] == target_configs.get(r['method']) and r['dataset'] in datasets]

    metrics = [
        ('mrr', 'MRR Score', 'fig_dataset_mrr.pdf'),
        ('faithfulness', 'Faithfulness Score', 'fig_dataset_faithfulness.pdf'),
        ('answer_relevancy', 'Answer Relevancy Score', 'fig_dataset_ansrel.pdf'),
        ('total_time', 'Per-Query Answer Time (s)', 'fig_dataset_total_time.pdf'),
        ('wan_time', 'Per-Query WAN Time (s)', 'fig_dataset_wan_time.pdf')
    ]

    for metric_key, y_label, filename in metrics:
        fig, ax = plt.subplots()

        for method in methods:
            method_runs = [r for r in baseline_runs if r['method'] == method]

            # Map the document count to the run and sort by it
            for r in method_runs:
                r['num_docs'] = dataset_sizes[r['dataset']]
            method_runs.sort(key=lambda x: x['num_docs'])

            if not method_runs:
                continue

            x_vals = [r['num_docs'] for r in method_runs]
            y_vals = [r[metric_key] for r in method_runs]

            ax.plot(x_vals, y_vals, marker='o', label=method.capitalize(), color=colors.get(method), zorder=3)

            for r in method_runs:
                ax.annotate(r['dataset'].capitalize(),
                            (r['num_docs'], r[metric_key]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=10)

        ax.set_xscale('log')

        # Use log scale for time plots so fast queries don't look like 0
        if 'time' in metric_key:
            ax.set_yscale('log')

        ax.set_xlabel('Number of Documents in Corpus')
        ax.set_ylabel(y_label)
        ax.grid(True, zorder=0)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), format='pdf')
        plt.close(fig)