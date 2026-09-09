import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
METHODS = ['bins', 'tree', 'pacmann']
K_VALUES = [10, 50, 100, 500, 1000]
# Assuming script is in 'bins-go', and results are in '../results'
BASE_DIR = "../../../datasets/results"


def parse_go_duration(duration_str):
    """
    Parses a Go duration string (e.g., '5h16m25.964s') into total seconds.
    """
    # Regex updated to include an optional 'hours' group at the start
    regex = r'(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?P<seconds>[\d.]+)s'

    match = re.match(regex, duration_str)

    if not match:
        print(f"Warning: Could not parse time '{duration_str}'. Returning 0.")
        return 0.0

    groups = match.groupdict()

    # Safely convert matches to float, defaulting to 0.0 if the group is None
    hours = float(groups['hours']) if groups['hours'] else 0.0
    minutes = float(groups['minutes']) if groups['minutes'] else 0.0
    seconds = float(groups['seconds'])

    return (hours * 3600) + (minutes * 60) + seconds

def load_data(methods, k, base_dir, amortized=False):
    data = {
        'methods': [],
        'mrr_pre': [],
        'mrr_post': [],
        'time_seconds': [],
        'data_sent_gb': [],
        'data_sent_mb': [],
        'time_seconds_stages': [],
        'faithfulness': [],
        'answer_relevancy': [],
        'k': k
    }

    for method in methods:
        # Construct the directory and filename based on your structure
        # Example: ../results/bins_10/bins_10_metadata.json
        folder_name = f"{method}_{k}"
        file_name = f"{method}_{k}_metadata.json"
        file_path = os.path.join(base_dir, folder_name, file_name)

        if not os.path.exists(file_path):
            print(f"Skipping {method}: File not found at {file_path}")
            continue

        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)

                total_queries = int(json_data.get('NumQueries', 0))

                # 1. Parse MRRs
                mrr_post = float(json_data.get('MRR', 0))
                mrr_pre = float(json_data.get('MRRPreReRank', 0))

                faithfulness = float(json_data.get('faithfulness', 0.0))
                answer_relevancy = float(json_data.get('answer_relevancy', 0.0))

                # 2. Parse Time
                time_str = json_data.get('TotalAnswerTime', "0s")
                total_seconds = parse_go_duration(time_str)

                # 3. Parse Data Sent (Uint64 count -> Bytes -> GB)
                # 1 Uint64 = 8 bytes
                uint64_count = float(json_data.get('TotalUint64Sent', 0))

                if uint64_count == 0:  # assume it's the tree method
                    uint64_count1 = float(json_data.get('TotalUint64Sent_stage1', 0))
                    uint64_count2 = float(json_data.get('TotalUint64Sent_stage2', 0))
                    uint64_count3 = float(json_data.get('TotalUint64Sent_stage3', 0))
                    uint64_count = uint64_count1 + uint64_count2 + uint64_count3

                    # grab the time spent in each stage while we're at it:
                    data['time_seconds_stages'].append(parse_go_duration(json_data.get('AnswerTime_stage1', "0s")))
                    data['time_seconds_stages'].append(parse_go_duration(json_data.get('AnswerTime_stage2', "0s")))
                    data['time_seconds_stages'].append(parse_go_duration(json_data.get('AnswerTime_stage3', "0s")))

                bytes_sent = uint64_count * 8
                gb_sent = bytes_sent / (1024**3) # Convert to Gigabytes
                mb_sent = bytes_sent / (1024 ** 2)  # Convert to Megabytes

                # Append to lists
                data['methods'].append(method)
                data['mrr_pre'].append(mrr_pre)
                data['mrr_post'].append(mrr_post)
                data['faithfulness'].append(faithfulness)
                data['answer_relevancy'].append(answer_relevancy)
                if amortized:
                    data['time_seconds'].append(total_seconds/total_queries)
                    data['data_sent_gb'].append(gb_sent/total_queries)
                    data['data_sent_mb'].append(mb_sent/total_queries)
                else:
                    data['time_seconds'].append(total_seconds)
                    data['data_sent_gb'].append(gb_sent)
                    data['data_sent_mb'].append(mb_sent)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return data


def plot_stage_breakdown(data):
    """
    Plots the time spent in Stage 1, 2, and 3 for the specific method
    that has stage data.
    """
    stages = data['time_seconds_stages'] # Expecting [s1, s2, s3]
    stage_labels = ['Stage 1', 'Stage 2', 'Stage 3']
    colors = ['#FFD700', '#FF8C00', '#FF4500'] # Gold, DarkOrange, OrangeRed

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bars
    bars = ax.bar(stage_labels, stages, color=colors, edgecolor='black', alpha=0.8)

    # Add labels and title
    ax.set_ylabel('Time (Seconds)')
    ax.set_title('Detailed Time Breakdown by Stage')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"figures/stage_breakdown_k{data['k']}.png")

def plot_performance(data, gb=True, amortized=False, llm_judge=False):
    # Set up a figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    title_prefix = "Amortized " if amortized else ""
    metric_type = "LLM Judge Eval" if llm_judge else "MRR"
    fig.suptitle(f'{title_prefix}Method Performance: {metric_type} (k={data["k"]})', fontsize=16)

    # Common variables
    methods = data['methods']
    x = np.arange(len(methods))

    # ---------------------------------------------------------
    # --- Plot 1: Score Comparison (Grouped Bar Chart) ---
    # ---------------------------------------------------------
    ax1 = axes[0]
    width = 0.35

    if llm_judge:
        # Compare Faithfulness vs Answer Relevancy
        rects1 = ax1.bar(x - width / 2, data['faithfulness'], width, label='Faithfulness', color='#2ca02c')  # Green
        rects2 = ax1.bar(x + width / 2, data['answer_relevancy'], width, label='Ans. Relevancy',
                         color='#1f77b4')  # Blue
        ax1.set_ylabel('Score (0-1)')
        ax1.set_title('LLM Metrics: Faithfulness vs Relevancy')
    else:
        # Original MRR Comparison
        rects1 = ax1.bar(x - width / 2, data['mrr_pre'], width, label='Pre-Rerank', color='#A0C4FF')
        rects2 = ax1.bar(x + width / 2, data['mrr_post'], width, label='Post-Rerank', color='#0055D4')
        ax1.set_ylabel('MRR Score')
        ax1.set_title('MRR Improvement (Pre vs Post)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # ---------------------------------------------------------
    # --- Helper to draw sorted dotted lines ---
    # ---------------------------------------------------------
    def plot_trend_line(ax, x_vals, y_vals, color):
        # We must sort points by X to draw a clean line, otherwise it zig-zags
        sorted_indices = np.argsort(x_vals)
        sorted_x = np.array(x_vals)[sorted_indices]
        sorted_y = np.array(y_vals)[sorted_indices]
        ax.plot(sorted_x, sorted_y, linestyle=':', color=color, alpha=0.6, zorder=1)

    # ---------------------------------------------------------
    # --- Plot 2: Efficiency Frontier (Time vs Score) ---
    # ---------------------------------------------------------
    ax2 = axes[1]
    time_data = data['time_seconds']

    if llm_judge:
        # Plot Faithfulness
        ax2.scatter(time_data, data['faithfulness'], s=100, marker='o', label='Faithfulness', c='#2ca02c', zorder=5)
        plot_trend_line(ax2, time_data, data['faithfulness'], '#2ca02c')

        # Plot Relevancy
        ax2.scatter(time_data, data['answer_relevancy'], s=100, marker='^', label='Relevancy', c='#1f77b4', zorder=5)
        plot_trend_line(ax2, time_data, data['answer_relevancy'], '#1f77b4')

        ax2.set_ylabel('LLM Score')
        ax2.legend()
    else:
        # Plot MRR
        ax2.scatter(time_data, data['mrr_post'], s=150, c='crimson', zorder=5)
        plot_trend_line(ax2, time_data, data['mrr_post'], 'crimson')
        ax2.set_ylabel('MRR (Post-Rerank)')

    # Annotate points (Only need to do this once per method)
    y_annotate = data['faithfulness'] if llm_judge else data['mrr_post']
    for i, txt in enumerate(methods):
        ax2.annotate(txt, (time_data[i], y_annotate[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Total Answer Time (Seconds)')
    ax2.set_title('Efficiency Frontier: Time vs Quality')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(left=0)

    # ---------------------------------------------------------
    # --- Plot 3: Bandwidth Cost (Data Sent vs Score) ---
    # ---------------------------------------------------------
    ax3 = axes[2]

    # Determine X data based on GB flag
    data_x = data['data_sent_gb'] if gb else data['data_sent_mb']
    xlabel = 'Total Data Sent (GB)' if gb else 'Total Data Sent (MB)'

    if llm_judge:
        # Plot Faithfulness
        ax3.scatter(data_x, data['faithfulness'], s=100, marker='o', label='Faithfulness', c='#2ca02c', zorder=5)
        plot_trend_line(ax3, data_x, data['faithfulness'], '#2ca02c')

        # Plot Relevancy
        ax3.scatter(data_x, data['answer_relevancy'], s=100, marker='^', label='Relevancy', c='#1f77b4', zorder=5)
        plot_trend_line(ax3, data_x, data['answer_relevancy'], '#1f77b4')

        ax3.set_ylabel('LLM Score')
        ax3.legend()
    else:
        # Plot MRR
        ax3.scatter(data_x, data['mrr_post'], s=150, c='forestgreen', zorder=5)
        plot_trend_line(ax3, data_x, data['mrr_post'], 'forestgreen')
        ax3.set_ylabel('MRR (Post-Rerank)')

    # Annotate points
    for i, txt in enumerate(methods):
        # We annotate based on the highest point or just the first metric to avoid clutter
        ax3.annotate(txt, (data_x[i], y_annotate[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

    ax3.set_xlabel(xlabel)
    ax3.set_title('Bandwidth Cost: Data vs Quality')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_xlim(left=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f"figures/perf_compare_k{data['k']}.jpg")


# --- Main Execution ---
if __name__ == "__main__":

    amortized = True
    gb = False
    llm_judge = True

    for k in K_VALUES:

        print(f"Loading data for k={k}...")

        results = load_data(METHODS, k, BASE_DIR, amortized=amortized)

        if not results['methods']:
            print("No data loaded. Check your paths and file names.")
        else:
            print(f"Loaded data for: {results['methods']}")
            plot_performance(results, gb=gb, amortized=amortized, llm_judge=llm_judge)
            plot_stage_breakdown(results)