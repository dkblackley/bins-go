import os
import re
import json
import matplotlib.pyplot as plt
import generate_k_plots

# ==========================================
# GLOBAL CONSTANTS & AESTHETICS
# ==========================================

# Okabe-Ito colorblind-friendly palette
COLORS = {
    'bins': '#009E73',  # Bluish Green
    'pacmann': '#E69F00',  # Orange
    'tree': '#56B4E9',  # Sky Blue
}

METHOD_ORDER = ['tree',  'bins', 'pacmann']
FONT_SIZE = 16

# The target internal parameters that should remain constant across k-values
TARGET_CONFIGS = {
    'bins': 'bs0.1_dpb1000',
    'pacmann': 'steps15_neighb32',
    'tree': 'b32_r128'
}


def setup_sleek_style():
    """Applies a modern, flat, and highly readable style to all matplotlib plots."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
        'legend.fontsize': FONT_SIZE - 2,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '#E5E5E5',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.figsize': (8, 6),
        'lines.linewidth': 2.5,
        'lines.markersize': 9
    })


# ==========================================
# DATA PARSING
# ==========================================

def parse_duration_to_seconds(duration_str):
    """Parses a Go-style duration string (e.g., '3m40.76s') into total seconds."""
    total_seconds = 0.0

    hours_match = re.search(r'([\d\.]+)h', duration_str)
    mins_match = re.search(r'([\d\.]+)m', duration_str)
    secs_match = re.search(r'([\d\.]+)s', duration_str)

    if hours_match:
        total_seconds += float(hours_match.group(1)) * 3600
    if mins_match:
        total_seconds += float(mins_match.group(1)) * 60
    if secs_match:
        total_seconds += float(secs_match.group(1))

    return total_seconds


def load_all_data(results_dir):
    """Scans the results directory and builds a structured dictionary of the metrics."""
    # Structure: { method_name: { k_value: { 'mrr': float, 'comm_cost': float, 'time_per_query': float } } }
    data_store = {m: {} for m in METHOD_ORDER}

    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return data_store

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Identify the method
        method = folder.split('_')[0]
        if method not in METHOD_ORDER:
            continue

        # Ensure it matches the constant internal parameters for this method
        if TARGET_CONFIGS[method] not in folder:
            continue

        # Extract k value
        k_match = re.search(r'_k(\d+)', folder)
        if not k_match:
            continue
        k_val = int(k_match.group(1))

        # Load metadata
        meta_path = os.path.join(folder_path, 'metadata.json')
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # 1. MRR logic (Pre-rerank for pacmann, Post-rerank for others)
        if method == 'pacmann':
            mrr = float(meta.get('MRRPreReRank', 0))
        else:
            mrr = float(meta.get('MRR', 0))

        # 2. Communication Cost
        comm_cost = float(meta.get('CommCostPerBatchKB', 0))

        # 3. Latency logic
        total_time_str = meta.get('TotalAnswerTime', '0s')
        total_time_sec = parse_duration_to_seconds(total_time_str)
        num_queries = float(meta.get('NumQueries', 1))
        time_per_query = total_time_sec / num_queries

        # Store
        data_store[method][k_val] = {
            'mrr': mrr,
            'comm_cost': comm_cost,
            'time_per_query': time_per_query
        }

    return data_store


if __name__ == "__main__":
    RESULTS_DIR = "../results"
    OUTPUT_DIR = "../plots"

    # 1. Apply global style
    setup_sleek_style()

    # 2. Parse logs
    print("Loading data from experiments...")
    data = load_all_data(RESULTS_DIR)

    # 3. Generate figures
    print("Generating plots...")
    generate_k_plots(data, OUTPUT_DIR, METHOD_ORDER, COLORS)
    print(f"Done! Plots saved to {OUTPUT_DIR}/")