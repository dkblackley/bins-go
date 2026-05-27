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
        'lines.markersize': 9,
        'axes.xmargin': 0.01,
        'axes.ymargin': 0.05
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

def load_extended_data(results_dir, method_order):
    """
    Builds a flat list of metadata across all datasets and configs.
    This provides maximum flexibility for multi-dimensional filtering.
    """
    all_runs = []
    if not os.path.exists(results_dir):
        return all_runs

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        parts = folder.split('_')
        if len(parts) < 4: continue

        method, dataset = parts[0], parts[1]
        if method not in method_order: continue

        k_match = re.search(r'_k(\d+)', folder)
        k_val = int(k_match.group(1)) if k_match else 0

        # Isolate the configuration string
        config_str = folder.replace(f"{method}_{dataset}_", "").replace(f"k{k_val}", "").strip('_')
        if not config_str: config_str = "default"

        meta_path = os.path.join(folder_path, 'metadata.json')
        if not os.path.exists(meta_path): continue

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Basic extracted params
        mrr = float(meta.get('MRRPreReRank', 0)) if method == 'pacmann' else float(meta.get('MRR', 0))
        num_queries = float(meta.get('NumQueries', 1))

        # Bins specific config parsing for the Bins analysis plot
        dpb, bs = None, None
        if method == 'bins':
            dpb_match = re.search(r'dpb(\d+)', config_str)
            bs_match = re.search(r'bs([\d\.]+)', config_str)
            if dpb_match: dpb = int(dpb_match.group(1))
            if bs_match: bs = float(bs_match.group(1))

        # Note: TotalWANTime is significantly larger than AnswerTime, suggesting
        # it is an aggregated threaded metric in seconds.
        run_info = {
            'method': method,
            'dataset': dataset,
            'k': k_val,
            'config': config_str,
            'dpb': dpb,
            'bs': bs,
            'mrr': mrr,
            'comm_cost': float(meta.get('CommCostPerBatchKB', 0)),
            'total_time': parse_duration_to_seconds(meta.get('TotalAnswerTime', '0s')),
            'wan_time': float(meta.get('TotalWANTime', 0)) / num_queries, # Normalized per query
            'lan_time': float(meta.get('TotalLANTime', 0)) / num_queries, # Normalized per query
            'faithfulness': float(meta.get('Faithfulness', 0)),
            'answer_relevancy': float(meta.get('AnswerRelevancy', 0)),
            'db_size_mb': float(meta.get('DBSizeInBytesMB', 0))
        }
        all_runs.append(run_info)

    return all_runs


import plot_config_tradeoffs
import plot_bins_analysis
import plot_dataset_comparisons
import plot_network_vs_k

if __name__ == "__main__":
    RESULTS_DIR = "../../../../datasets/results"
    OUTPUT_DIR = "./plots"

    setup_sleek_style()

    print("Loading classic filtered data for original plots...")
    filtered_data = load_all_data(RESULTS_DIR)

    print("Loading extended flat data for multi-dimensional plots...")
    extended_data = load_extended_data(RESULTS_DIR, METHOD_ORDER)

    print("Generating classic k-plots...")
    generate_k_plots.generate_k_plots(filtered_data, OUTPUT_DIR, METHOD_ORDER, COLORS)

    print("Generating new analytical plots...")
    plot_config_tradeoffs.plot_mrr_vs_time(extended_data, OUTPUT_DIR, COLORS)
    plot_bins_analysis.plot_bins_parameters(extended_data, OUTPUT_DIR, param='dpb')
    plot_bins_analysis.plot_bins_parameters(extended_data, OUTPUT_DIR, param='bs')
    plot_dataset_comparisons.plot_cross_dataset_metrics(extended_data, OUTPUT_DIR, TARGET_CONFIGS, COLORS, METHOD_ORDER)
    plot_network_vs_k.plot_wan_lan_vs_k(extended_data, OUTPUT_DIR, TARGET_CONFIGS, COLORS, METHOD_ORDER)

    print(f"Done! Plots saved to {OUTPUT_DIR}/")