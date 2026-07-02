import os
import re
import json
import matplotlib.pyplot as plt
import metric_by_configs

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
FONT_SIZE = 26

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
        'grid.color': '#E5E5E5',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.figsize': (10, 6),
        'lines.linewidth': 2.5,
        'lines.markersize': 9,
        'axes.xmargin': 0.015,
        'axes.ymargin': 0.015,
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
    Builds a structured nested map of metadata:
    nested_data[method][dataset] = [ list of config runs ]
    This allows for extremely fast debugging and logical filtering.
    """
    # Initialize dictionary hierarchy
    nested_data = {method: {} for method in method_order}

    if not os.path.exists(results_dir):
        return nested_data

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        parts = folder.split('_')
        if len(parts) < 4: continue

        method, dataset = parts[0], parts[1]
        if method not in method_order: continue

        # Ensure dataset key exists for this method
        if dataset not in nested_data[method]:
            nested_data[method][dataset] = []

        k_match = re.search(r'_k(\d+)', folder)
        k_val = int(k_match.group(1)) if k_match else 0

        # Isolate the configuration string
        config_str = folder.replace(f"{method}_{dataset}_", "").replace(f"k{k_val}", "").strip('_')
        if not config_str: config_str = "default"

        meta_path = os.path.join(folder_path, 'metadata.json')
        if not os.path.exists(meta_path): continue

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        dpb, bs = None, None
        if method == 'bins':
            dpb_match = re.search(r'dpb(\d+)', config_str)
            bs_match = re.search(r'bs([\d\.]+)', config_str)
            if dpb_match: dpb = int(dpb_match.group(1))
            if bs_match: bs = float(bs_match.group(1))

        mrr = float(meta.get('MRRPreReRank', 0)) if method == 'pacmann' else float(meta.get('MRR', 0))
        num_queries = float(meta.get('NumQueries', 1))

        if num_queries == 1:
            print(f"[DEBUG] Only 1 query detected for {method} on {dataset}?")
            if dataset == "msmarco":
                num_queries = 6980  # Hard fallback based on prior runs

        total_uint64 = float(meta.get('TotalUint64Sent', 0))
        # 1 uint64 = 8 bytes. Convert to MB, then divide by num_queries
        comm_cost_mb = (total_uint64 * 8) / (1024 * 1024) / num_queries

        run_info = {
            'method': method,
            'dataset': dataset,
            'k': k_val,
            'config': config_str,
            'dpb': dpb,
            'bs': bs,
            'mrr': mrr,
            'comm_cost': comm_cost_mb,
            # Normalize total time to strictly Per-Query
            'total_time': parse_duration_to_seconds(meta.get('TotalAnswerTime', '0s')) / num_queries,
            'wan_time': float(meta.get('TotalWANTime', 0)) / num_queries,
            'lan_time': float(meta.get('TotalLANTime', 0)) / num_queries,
            'faithfulness': float(meta.get('Faithfulness', 0)),
            'answer_relevancy': float(meta.get('AnswerRelevancy', 0)),
            'db_size_mb': float(meta.get('DBSizeInBytesMB', 0))
        }

        # Append to our structured nested map
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
    
    
    for ds in datasets:
        # Example plotting calls utilizing the new parameter flags!

        # metric_by_configs.plot_metric_vs_lan_time(
        #     nested_data, OUTPUT_DIR, dataset=ds,
        #     method_order=METHOD_ORDER, colors=COLORS, target_configs=TARGET_CONFIGS,
        #     metric_key='mrr', metric_label='MRR Score',
        #     use_log_scale=False, enforce_monotonic=False
        # )

        # metric_by_configs.plot_metric_vs_wan_time(
        #     nested_data, OUTPUT_DIR, dataset=ds,
        #     method_order=METHOD_ORDER, colors=COLORS, target_configs=TARGET_CONFIGS,
        #     metric_key='mrr', metric_label='MRR Score',
        #     use_log_scale=False, enforce_monotonic=False
        # )

        metric_by_configs.plot_metric_vs_total_time(
            nested_data, OUTPUT_DIR, dataset=ds,
            method_order=METHOD_ORDER, colors=COLORS, target_configs=TARGET_CONFIGS,
            metric_key='mrr', metric_label='MRR Score',
            use_log_scale=False, enforce_monotonic=False
        )

        # metric_by_configs.plot_quality_vs_time(
        #     nested_data, OUTPUT_DIR, dataset=ds,
        #     method_order=METHOD_ORDER, colors=COLORS, target_configs=TARGET_CONFIGS,
        #     time_key='total_time', time_label='Per-Query Total Time (s)',
        #     use_log_scale=False, enforce_monotonic=False
        # )

    print(f"Done! Plots saved to {OUTPUT_DIR}/")