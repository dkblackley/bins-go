import logging
import os
import re
import json
import matplotlib.pyplot as plt
import metric_by_configs
import bins_ablation

# ==========================================
# GLOBAL CONSTANTS & AESTHETICS
# ==========================================

# Okabe-Ito colorblind-friendly palette
COLORS = {
    'bins': '#009E73',  # Bluish Green
    'pacmann': '#E69F00',  # Orange
    'tree': '#56B4E9',  # Sky Blue
}

METHOD_ORDER = ['tree', 'bins', 'pacmann']
FONT_SIZE = 60

# The target internal parameters that should remain constant across k-values
TARGET_CONFIGS = {
    'bins': 'bs1.0_dpb1000',
    'pacmann': 'steps15_neighb32',
    'tree': 'b64_r128'
}

SUBPLOT_SIZE = (20, 5)


def setup_sleek_style():
    """Applies a modern, flat, and highly readable style to all matplotlib plots."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 8,
        'ytick.labelsize': FONT_SIZE - 8,
        'legend.fontsize': FONT_SIZE - 4,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.autolayout': True,
        'figure.constrained_layout.use': False,
        'grid.color': '#E5E5E5',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.figsize': (17, 11),
        'axes.titlepad': 50,
        'axes.labelpad': 0,
        'lines.linewidth': 8.0,
        'lines.markersize': 25,
        'figure.subplot.right': 0.99,
    })


# ==========================================
# DATA PARSING
# ==========================================

def parse_duration_to_seconds(duration_str):
    """
    Robustly parses a Go-style duration string into total seconds.
    Can correctly handle '3m40.76s', '101.38ms', and '1.2µs'.
    """
    total_seconds = 0.0

    # Match hours
    h_match = re.search(r'([\d\.]+)h', duration_str)
    if h_match:
        total_seconds += float(h_match.group(1)) * 3600

    # Match minutes (must be 'm' not followed by 's', to avoid catching 'ms')
    m_match = re.search(r'([\d\.]+)m(?!s)', duration_str)
    if m_match:
        total_seconds += float(m_match.group(1)) * 60

    # Match seconds (must be 's' not preceded by 'm', 'µ', or 'u')
    s_match = re.search(r'(?<!m)(?<!µ)(?<!u)([\d\.]+)s', duration_str)
    if s_match:
        total_seconds += float(s_match.group(1))

    # Match milliseconds
    ms_match = re.search(r'([\d\.]+)ms', duration_str)
    if ms_match:
        total_seconds += float(ms_match.group(1)) / 1000.0

    # Match microseconds
    us_match = re.search(r'([\d\.]+)[µu]s', duration_str)
    if us_match:
        total_seconds += float(us_match.group(1)) / 1000000.0

    return total_seconds


def load_extended_data(results_dir, method_order):
    """
    Builds a structured nested map of metadata using explicit substring matching.
    nested_data[method][dataset] = [ list of config runs ]
    Includes robust error logging for fast debugging.
    """
    # Initialize dictionary hierarchy
    nested_data = {method: {} for method in method_order}

    if not os.path.exists(results_dir):
        logging.error(f"[ERROR] Results directory does not exist: {results_dir}")
        return nested_data

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 1. Determine Method via direct matching
        if 'bins' in folder:
            method = 'bins'
        elif 'pacmann' in folder:
            method = 'pacmann'
        elif 'tree' in folder:
            method = 'tree'
        else:
            logging.warning(f"[SKIP] Unknown method in folder name: {folder}")
            continue

        if method not in method_order:
            continue

        # 2. Determine Dataset via direct matching
        if 'msmarco' in folder:
            dataset = 'msmarco'
        elif 'scifact' in folder:
            dataset = 'scifact'
        elif 'trec-covid' in folder:
            dataset = 'trec-covid'
        else:
            logging.warning(f"[SKIP] Unknown dataset in folder name: {folder}")
            continue

        # Ensure dataset key exists for this method
        if dataset not in nested_data[method]:
            nested_data[method][dataset] = []

        # 3. Determine K-value
        k_match = re.search(r'_k(\d+)', folder)
        if not k_match:
            logging.error(f"[ERROR] Missing k-value (e.g., '_k100') in folder: {folder}")
            continue
        k_val = int(k_match.group(1))

        # 4. Isolate Config String (Strip out the parts we already know)
        # Replacing them exactly once avoids breaking configs that might share names
        config_str = folder.replace(f"{method}_", "", 1).replace(f"{dataset}_", "", 1).replace(f"k{k_val}", "", 1)
        config_str = config_str.replace("__", "_").strip('_')
        if not config_str:
            config_str = "default"

        # 5. Load and Validate JSON
        meta_path = os.path.join(folder_path, 'metadata.json')
        if not os.path.exists(meta_path):
            logging.error(f"[ERROR] No metadata.json found in: {folder}")
            continue

        with open(meta_path, 'r') as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                logging.error(f"[ERROR] Corrupted JSON in: {meta_path}")
                continue

        # Bins specific config parsing
        dpb, bs = None, None
        if method == 'bins':
            dpb_match = re.search(r'dpb(\d+)', config_str)
            bs_match = re.search(r'bs([\d\.]+)', config_str)
            if dpb_match: dpb = int(dpb_match.group(1))
            if bs_match: bs = float(bs_match.group(1))

        recall = meta.get('Recall')
        if recall is None:
            logging.error(f"[ERROR] Missing Recall field for {method} in {folder}")
            recall = 0.0
        else:
            recall = float(recall)
        # 6. Extract Metrics with Error Catching
        mrr_val = meta.get('MRRPreReRank') if method == 'pacmann' else meta.get('MRR')
        if mrr_val is None:
            logging.error(f"[ERROR] Missing MRR field for {method} in {folder}")
            mrr = 0.0
        else:
            mrr = float(mrr_val)

        if mrr == 0.0:
            logging.warning(f"[WARNING] {method} had 0.0 MRR with config: {config_str} on {dataset}")

        num_queries = float(meta.get('NumQueries', 1))
        if num_queries <= 1:
            logging.error(
                f"[DEBUG] Only {num_queries} query detected for {method} on {dataset} with config {config_str}")
            if dataset == "msmarco":
                num_queries = 6980  # Hard fallback based on prior runs

        total_uint64 = float(meta.get('TotalUint64Sent', 0))
        if total_uint64 == 0:
            logging.error(f"[ERROR] {method} had 0 total uint sent with config: {config_str} on {dataset}")

        comm_cost_mb = (total_uint64 * 8) / (1024 * 1024) / num_queries

        # 7. Final Dictionary Assembly (Keys kept strictly identical for downstream plots)
        run_info = {
            'method': method,
            'dataset': dataset,
            'k': k_val,
            'config': config_str,
            'dpb': dpb,
            'bs': bs,
            'mrr': mrr,
            'recall': recall,
            'comm_cost': comm_cost_mb,
            'total_time': parse_duration_to_seconds(meta.get('TotalAnswerTime', '0s')) / num_queries,
            'wan_time': float(meta.get('TotalWANTime', 0)) / num_queries,
            'lan_time': float(meta.get('TotalLANTime', 0)) / num_queries,
            'faithfulness': float(meta.get('Faithfulness', 0)),
            'answer_relevancy': float(meta.get('AnswerRelevancy', 0)),
            'db_size_mb': float(meta.get('DBSizeInBytesMB', 0))
        }

        nested_data[method][dataset].append(run_info)

    return nested_data

if __name__ == "__main__":
    RESULTS_DIR = "../../../../datasets/results"
    OUTPUT_DIR = "./plots"

    setup_sleek_style()

    print("Loading extended nested data for multi-dimensional plots...")
    nested_data = load_extended_data(RESULTS_DIR, METHOD_ORDER)

    print("Generating network and quality plots per dataset...")
    datasets = ['msmarco', 'scifact', 'trec-covid']
    
    
    # for ds in datasets:
    #     # Example plotting calls utilizing the new parameter flags!
    #
    #     metric_by_configs.plot_metric_vs_lan_time(
    #         nested_data, OUTPUT_DIR, dataset=ds,
    #         method_order=METHOD_ORDER, colors=COLORS, target_configs=TARGET_CONFIGS,
    #         metric_key='mrr', metric_label='MRR',
    #         use_log_scale=False, enforce_monotonic=False
    #     )
    #
    #     metric_by_configs.plot_metric_vs_wan_time(
    #         nested_data, OUTPUT_DIR, dataset=ds,
    #         method_order=METHOD_ORDER, colors=COLORS, target_configs=TARGET_CONFIGS,
    #         metric_key='mrr', metric_label='MRR',
    #         use_log_scale=False, enforce_monotonic=False
    #     )
    #
    #     metric_by_configs.plot_metric_vs_total_time(
    #         nested_data, OUTPUT_DIR, dataset=ds,
    #         method_order=METHOD_ORDER, colors=COLORS, target_configs=TARGET_CONFIGS,
    #         metric_key='mrr', metric_label='MRR',
    #         use_log_scale=False, enforce_monotonic=False
    #     )
    #
    #     metric_by_configs.plot_quality_vs_time(
    #         nested_data, OUTPUT_DIR, dataset=ds,
    #         method_order=METHOD_ORDER, colors=COLORS, target_configs=TARGET_CONFIGS,
    #         time_key='total_time', time_label='Per-Query Total Time (s)',
    #         use_log_scale=False, enforce_monotonic=False
    #     )

    print("Generating Bins ablation plots...")

    # Generate the 3 plots varying Bin Size
    bins_ablation.plot_bins_ablations(
        nested_data, OUTPUT_DIR, param_to_vary='bs', target_k=100
    )

    # Generate the 3 plots varying Docs Per Bin
    bins_ablation.plot_bins_ablations(
        nested_data, OUTPUT_DIR, param_to_vary='dpb', target_k=100
    )


    print(f"Done! Plots saved to {OUTPUT_DIR}/")