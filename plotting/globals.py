"""
Global settings and style for every paper figure. Change something here and
every plot follows. Always import as `import globals as g`.

Sizes are real inches/points. The PDFs are meant to be included at their
natural size (\\includegraphics{figures/x.pdf}, or width=<FIG_SIZE[0]>in), so
FONT_SIZE is the point size you see in the paper. If you shrink a figure in
LaTeX the fonts shrink with it, so change FIG_SIZE here instead.
"""

import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================

RESULTS_DIR = os.environ.get('RESULTS_DIR', '/scratch/dblackle/results')
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

LOG_LEVEL = logging.INFO   # logging.DEBUG also prints every folder as it is parsed

# ==========================================
# EXPERIMENT SETTINGS
# ==========================================

K_MAIN = 10      # k for the method comparison plots
K_ABLATION = 10    # k for the bins ablations (the dpb=1500 sweep only exists at k10 right now)

BINS_FIXED_DPB = 1500                                  # docs per bin when sweeping bin size
BINS_FIXED_BS = {'msmarco': 1000000, 'scifact': 5000}  # bin size when sweeping docs per bin

# ==========================================
# ORDER, NAMES, COLOURS (Okabe-Ito)
# ==========================================

METHOD_ORDER = ['bins', 'tree', 'pacmann']   # order of lines, bars and legend entries

METHOD_LABELS = {'bins': 'BM25-Bin', 'tree': 'BM25-Tree', 'pacmann': 'PACMANN'}
METHOD_COLORS = {'bins': '#D55E00', 'tree': '#CC79A7', 'pacmann': '#0072B2'}
METHOD_MARKERS = {'bins': 's', 'tree': '^', 'pacmann': 'o'}
METHOD_HATCHES = {'bins': '///', 'tree': 'xxx', 'pacmann': ''}   # greyscale-safe second channel

DATASETS = ['msmarco', 'scifact']

DATASET_LABELS = {'msmarco': 'MS MARCO', 'scifact': 'SciFact'}
DATASET_COLORS = {'msmarco': '#009E73', 'scifact': '#E69F00'}   # kept distinct from the method colours
DATASET_MARKERS = {'msmarco': 'o', 'scifact': 's'}

BASELINE_LINESTYLE = ':'   # dotted reference lines (e.g. PACMANN DB size)

# Tree PIR stages, used by the stacked latency bars.
STAGE_ORDER = ['s1', 's2', 's3']
STAGE_LABELS = {'s1': 'Stage 1', 's2': 'Stage 2', 's3': 'Stage 3'}
STAGE_COLORS = {'s1': '#0072B2', 's2': '#56B4E9', 's3': '#E69F00'}
STAGE_HATCHES = {'s1': '', 's2': '///', 's3': 'xxx'}

# Bins DB layouts, used wherever vec0 and vec1 are compared directly.
VEC_LABELS = {1: '1-stage (single DB)', 0: '2-stage (split DBs)'}
VEC_COLORS = {1: '#009E73', 0: '#D55E00'}
VEC_HATCHES = {1: '', 0: '///'}

# ==========================================
# METRICS
# ==========================================

# Run key (see load_results.py) -> axis label and which direction is better.
# 'better' is what keep_improving() uses. '{k}' is filled in by label().
METRICS = {
    'mrr':                {'label': 'MRR@{k}',            'better': 'higher'},
    'mrr_pre_rerank':     {'label': 'MRR@{k} (no rerank)', 'better': 'higher'},
    'recall':             {'label': 'Recall@{k}',         'better': 'higher'},
    'recall_pre_rerank':  {'label': 'Recall@{k} (no rerank)', 'better': 'higher'},
    'faithfulness':       {'label': 'Faithfulness',       'better': 'higher'},
    'answer_relevancy':   {'label': 'Answer Relevancy',   'better': 'higher'},
    'lan_time':           {'label': 'LAN Latency (s)',    'better': 'lower'},
    'wan_time':           {'label': 'WAN Latency (s)',    'better': 'lower'},
    'total_time':         {'label': 'Computation (s)',    'better': 'lower'},
    'rerank_time':        {'label': 'Rerank Time (s)',    'better': 'lower'},
    'comm_kb':            {'label': 'Communication (KB)', 'better': 'lower'},
    'db_size_mb':         {'label': 'DB Size (MB)',       'better': 'lower'},
    'client_storage_mb':  {'label': 'Client Storage (MB)', 'better': 'lower'},
    'bs':                 {'label': 'Bin Size',           'better': 'lower'},
    'dpb':                {'label': 'Docs per Bin',       'better': 'lower'},
    'maintenance_time': {'label': 'Maintenance (s)', 'better': 'lower'},
    'preproc_rounds': {'label': 'Preproc. Rounds', 'better': 'lower'},
    'pir_rounds': {'label': 'PIR Rounds', 'better': 'lower'},
}


def label(key, k=K_MAIN):
    """Axis label for a run key. Split keys ('lan_time_vec') fall back to the base key."""
    base = key.replace('_bm25', '').replace('_vec', '')
    text = METRICS.get(key, METRICS.get(base, {'label': key}))['label'].format(k=k)
    if key.endswith('_bm25'):
        text = 'BM25 DB ' + text
    elif key.endswith('_vec'):
        text = 'Vec DB ' + text
    return text


# ==========================================
# SIZES AND FONTS
# ==========================================

FIG_SIZE = (2.7, 2.0)   # (width, height) in inches, two of these fit side by side in ICLR's 5.5in

FONT_SIZE = 11          # axis labels and titles
TICK_SIZE = 10          # tick labels
LEGEND_SIZE = 9         # legend text
LINE_WIDTH = 1.6
MARKER_SIZE = 4.5

# Shared legend look. Position (loc / bbox_to_anchor / ncol) is set in each plot file.
LEGEND_STYLE = dict(frameon=False, fontsize=LEGEND_SIZE, handlelength=1.8,
                    handletextpad=0.4, columnspacing=0.9, borderaxespad=0.1)


def setup_style():
    """Call once before plotting."""
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset':  'stix',
        'font.size':         FONT_SIZE,
        'axes.titlesize':    FONT_SIZE,
        'axes.labelsize':    FONT_SIZE,
        'xtick.labelsize':   TICK_SIZE,
        'ytick.labelsize':   TICK_SIZE,
        'legend.fontsize':   LEGEND_SIZE,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'grid.color':        '#DDDDDD',
        'grid.linewidth':    0.6,
        'axes.titlepad':     4,
        'axes.labelpad':     2,
        'lines.linewidth':   LINE_WIDTH,
        'lines.markersize':  MARKER_SIZE,
        'hatch.linewidth':   0.6,
        'savefig.format':    'pdf',
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype':      42,   # embed TrueType so text stays selectable
    })


def setup_logging():
    logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(name)s: %(message)s')
    for noisy in ('fontTools', 'matplotlib'):
        logging.getLogger(noisy).setLevel(logging.WARNING)   # PDF font subsetting spams INFO
