import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
METHODS = ['bins', 'tree', 'pacmann']
K_VALUE = 10
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
        'data_sent_mb': []
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

                bytes_sent = uint64_count * 8
                gb_sent = bytes_sent / (1024**3) # Convert to Gigabytes
                mb_sent = bytes_sent / (1024 ** 2)  # Convert to Megabytes

                # Append to lists
                data['methods'].append(method)
                data['mrr_pre'].append(mrr_pre)
                data['mrr_post'].append(mrr_post)
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

def plot_performance(data, gb=True, amortized=False):
    # Set up a figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if amortized:
        fig.suptitle(f'Amortized Method Performance Comparison (k={K_VALUE})', fontsize=16)
    else:
        fig.suptitle(f'Method Performance Comparison (k={K_VALUE})', fontsize=16)

    # --- Plot 1: MRR Pre vs Post (Grouped Bar Chart) ---
    x = np.arange(len(data['methods']))
    width = 0.35

    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, data['mrr_pre'], width, label='Pre-Rerank', color='#A0C4FF')
    rects2 = ax1.bar(x + width/2, data['mrr_post'], width, label='Post-Rerank', color='#0055D4')

    ax1.set_ylabel('MRR Score')
    ax1.set_title('MRR Improvement (Pre vs Post)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data['methods'])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: MRR vs Running Time (Trade-off) ---
    ax2 = axes[1]
    # Scatter plot
    ax2.scatter(data['time_seconds'], data['mrr_post'], s=150, c='crimson', zorder=5)

    # Annotate points
    for i, txt in enumerate(data['methods']):
        ax2.annotate(txt, (data['time_seconds'][i], data['mrr_post'][i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')

    ax2.set_xlabel('Total Answer Time (Seconds)')
    ax2.set_ylabel('MRR (Post-Rerank)')
    ax2.set_title('Efficiency Frontier: Time vs Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.5)
    # Invert x axis if you prefer "faster is better" on the right,
    # but standard is 0 on left. 0 on left is usually fine.

    # --- Plot 3: MRR vs Data Sent (Bandwidth Cost) ---
    ax3 = axes[2]

    if gb:
        ax3.scatter(data['data_sent_gb'], data['mrr_post'], s=150, c='forestgreen', zorder=5)
        ax3.set_xlabel('Total Data Sent (GB)')
    else:
        ax3.scatter(data['data_sent_mb'], data['mrr_post'], s=150, c='forestgreen', zorder=5)
        ax3.set_xlabel('Total Data Sent (MB)')

    for i, txt in enumerate(data['methods']):
        ax3.annotate(txt, (data['data_sent_gb'][i], data['mrr_post'][i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')

    ax3.set_ylabel('MRR (Post-Rerank)')
    ax3.set_title('Bandwidth Cost: Data Sent vs Accuracy')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":

    amortized = True
    gb = False

    results = load_data(METHODS, K_VALUE, BASE_DIR, amortized=amortized)

    if not results['methods']:
        print("No data loaded. Check your paths and file names.")
    else:
        print(f"Loaded data for: {results['methods']}")
        plot_performance(results, gb=gb, amortized=amortized)