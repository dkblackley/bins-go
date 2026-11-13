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
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Polygon

# ==========================================
# PATHS
# ==========================================

RESULTS_DIR = os.environ.get('RESULTS_DIR', '/scratch/dblackle/results')
# Several directories can be given separated by ':' (os.pathsep), e.g. when the
# baselines write their runs somewhere else:
#   RESULTS_DIR=/data/results:/data/tiptoe:/data/coeus python main.py
_HERE = os.path.dirname(os.path.abspath(__file__))

# Searched as well as RESULTS_DIR, because the baselines (baselines/*/run_*.py)
# write their runs to their own folders. Relative paths are relative to this
# file, not to where you run python from. Folders that don't exist are skipped
# quietly (e.g. on the cluster). Override with EXTRA_RESULTS_DIRS=a:b, or set it
# to '' to only read RESULTS_DIR.
EXTRA_RESULTS_DIR = os.environ.get('EXTRA_RESULTS_DIRS', os.pathsep.join([
    '../../../../datasets/tiptoe',
    '../../../../datasets/coeus',
    '../../../../datasets/results',   # this machine: the tiptoe_* / coeus_* runs live here
]))
EXTRA_RESULTS_DIRS = [os.path.normpath(os.path.join(_HERE, d))
                      for d in EXTRA_RESULTS_DIR.split(os.pathsep) if d]

RESULTS_DIRS = [d for d in RESULTS_DIR.split(os.pathsep) if d]
# Only these methods' folders are read from EXTRA_RESULTS_DIRS, so a mixed folder like datasets/results (old real bins /
# tree runs next to the baselines) only contributes its baselines. EXTRA_METHODS="" reads every method there.
EXTRA_METHODS = os.environ.get('EXTRA_METHODS', 'tiptoe coeus').split()

# Simulated runs (SIMULATE=1, run_me.sh's default): same quality, bytes, LAN/WAN and client storage as a real run, but
# every server timing (TotalAnswerTime, PreprocessingTime, ...) is -1, loaded as NaN. With FALLBACK_TO_SIMULATED on,
# load_results() takes each config from RESULTS_DIRS when it has a metadata.json there, else from these folders, and
# marks every run with run['source'] = 'real' / 'simulated'. FALLBACK_TO_SIMULATED=0 reads RESULTS_DIRS only.
# default: hopper's folder there, else ../datasets/results_simulated next to this repo (this machine)
_SIM_DEFAULT = '/scratch/dblackle/results_simulated'
if not os.path.isdir(_SIM_DEFAULT):
    _SIM_DEFAULT = os.path.normpath(os.path.join(_HERE, '../../../../datasets/results_simulated'))
SIMULATED_RESULTS_DIR = os.environ.get('SIMULATED_RESULTS_DIR', _SIM_DEFAULT)
SIMULATED_RESULTS_DIRS = [d for d in SIMULATED_RESULTS_DIR.split(os.pathsep) if d]
FALLBACK_TO_SIMULATED = os.environ.get('FALLBACK_TO_SIMULATED', '1') != '0'
# ONLY_SIMULATED (on by default): read SIMULATED_RESULTS_DIRS only, no RESULTS_DIRS and no fallback to a real run
# when a simulated one is missing. EXTRA_RESULTS_DIRS (Tiptoe/Coeus, always simulated) are still read. Overrides
# FALLBACK_TO_SIMULATED. Every run gets source='simulated'. ONLY_SIMULATED=0 brings the real runs back.
ONLY_SIMULATED = os.environ.get('ONLY_SIMULATED', '1') != '0'
# the vec0 client-storage sweep (bins_vec0_..._vb<B>_vr<R>_ib<I>: embedding-DB batch / rounds, ID-DB batch) loads as more
# PILLAR-Bin configs, so the method comparisons can pick them; INCLUDE_VEC0_SWEEP=0 leaves them out everywhere
INCLUDE_VEC0_SWEEP = os.environ.get('INCLUDE_VEC0_SWEEP', '1') != '0'
# Methods left out of every client-storage comparison: their client_storage_mb is set to NaN at load time (the reported
# value stays in client_storage_mb_reported), so any plot or table with client storage on an axis skips them and every
# other metric keeps them. Tiptoe (centroids only: 0.001-17 MB; it downloads a 35-42 MB token per query instead) and
# Coeus (a 0.19 MB secret key; it sends 5-814 MB per query) dominate client storage outright, which hides the
# PILLAR vs PACMANN comparison. STORAGE_EXCLUDE_METHODS="" puts them back.
STORAGE_EXCLUDE_METHODS = os.environ.get('STORAGE_EXCLUDE_METHODS', 'tiptoe coeus').split()

# Runs of ANY method whose client storage is >= MAX_CLIENT_FRACTION x the dataset's embedding DB (EMBED_DB_MB) are
# dropped by load_results(): at that point the client might as well download the embeddings. The default (1.0) drops
# every run whose client keeps more than the whole embedding DB. FILTER_CLIENT_STORAGE=0 keeps every run (or pass
# filter_client=False); MAX_CLIENT_FRACTION=2/3 etc. moves the cut. Dropped runs are in load_results.DROPPED.
FILTER_CLIENT_STORAGE = os.environ.get('FILTER_CLIENT_STORAGE', '1') != '0'
MAX_CLIENT_FRACTION = float(os.environ.get('MAX_CLIENT_FRACTION', 1.0))
# Size (MB = 2^20 bytes, like DBSizeInBytesMB / ClientStorageMB) of ../datasets/<d>/<d>_corpus_embed_f32.npy on hopper,
# checked 2026-11-10: docs x 192 dims x float32 (the PCA'd e5 embeddings), + the .npy header.
EMBED_DB_MB = {
    'scifact': 3.8,
    'msmarco': 6475.9,
    'nq': 1964.0,
    'fever': 3967.2,
    'dbpedia-entity': 3395.5,
    'msmarco-v2-25m': 18310.5,
    'msmarco-v2-50m': 36621.1,
    'msmarco-v2': 101341.0,
    'trec-covid': 125.5,
}
FIGURE_DIR = os.path.join(_HERE, 'figures')

# Inputs of the per-query rebuttal numbers (rebuttal_numbers.py): the datasets folder (corpus.jsonl, qrels/,
# queries) and the repo's data/ folder (data/<d>_tree/queries.qrels.analyzed.json). Defaults match the local layout
# next to bins-go; DATASETS_ROOT / TREE_DATA_DIR override.
DATASETS_ROOT = os.environ.get('DATASETS_ROOT', os.path.normpath(os.path.join(_HERE, '../../../../datasets')))
TREE_DATA_DIR = os.environ.get('TREE_DATA_DIR', os.path.normpath(os.path.join(_HERE, '../data')))
# Non-private baselines (baselines.py writes <dataset>_k<k>.json): searched in these folders
BASELINES_DIRS = [d for d in os.environ.get('BASELINES_DIRS', os.pathsep.join([
    os.path.join(_HERE, '../baselines_out'), os.path.join(_HERE, '../baselines'),
    os.path.join(DATASETS_ROOT, 'baselines')])).split(os.pathsep) if d]

LOG_LEVEL = logging.INFO   # logging.DEBUG also prints every folder as it is parsed

# ==========================================
# EXPERIMENT SETTINGS
# ==========================================

K_MAIN = 10      # k for the method comparison plots
K_ABLATION = 10    # k for the bins ablations (the dpb=1500 sweep only exists at k10 right now)

BINS_FIXED_DPB = 1500                                  # docs per bin when sweeping bin size
BINS_FIXED_BS = {'msmarco': 1000000, 'scifact': 5000}  # bin size when sweeping docs per bin; a dataset not
                        # listed uses the bin size with the most dpb values (pu.fixed_bs)

# ==========================================
# ORDER, NAMES, COLOURS (Okabe-Ito)
# ==========================================

# Extra private baselines asked for by the reviewers (see baselines/README.md). Set to [] to drop them
# from every plot; methods without runs are skipped anyway, only the bar widths change.
EXTRA_BASELINES = ['coeus', 'tiptoe']

# The bins -text mode (bins_text_* folders: compressed doc text + client BM25). Set to [] to leave it out of the
# method comparisons (keeps the old bar widths); its runs are then not loaded.
TEXT_BINS = []   # bins_text_* runs load as plain 'bins' runs (vec=2), so no method / colour / legend entry of their own

METHOD_ORDER = ['bins'] + TEXT_BINS + ['tree', 'pacmann'] + EXTRA_BASELINES   # order of lines, bars and legend entries

METHOD_LABELS = {'bins': 'PILLAR-Bin', 'bins-text': 'PILLAR-Bin (text)', 'tree': 'PILLAR-Tree', 'pacmann': 'PACMANN',
                 'tiptoe': 'Tiptoe', 'coeus': 'Coeus'}
METHOD_COLORS = {'bins': '#009E73', 'bins-text': '#000000', 'tree': '#E69F00', 'pacmann': '#0072B2',
                 'coeus': '#D55E00', 'tiptoe': '#CC79A7'}
# Okabe-Ito has 8 colours, so the baselines reuse ones from elsewhere that never
# share a figure with a method comparison: Coeus = VEC_COLORS[0] (bins-only
# plots), Tiptoe = DATASET_COLORS['msmarco']. Sky blue (#56B4E9) is left out
# because it is too close to PACMANN's blue, yellow because it vanishes on white
METHOD_MARKERS = {'bins': 's', 'bins-text': 'P', 'tree': '^', 'pacmann': 'o', 'coeus': 'v', 'tiptoe': 'D'}
METHOD_HATCHES = {'bins': '///', 'bins-text': '++', 'tree': 'xxx', 'pacmann': '',
                  'coeus': '\\\\', 'tiptoe': '...'}   # greyscale-safe second channel

# Methods whose 'mrr' is read from MRRPreReRank instead of MRR: the local
# cosine rerank is part of our pipeline, not theirs, so they are compared on
# the list they actually return. 'mrr_pre_rerank' keeps the same value and the
# reranked number is still loaded as 'mrr_reranked' (see load_results.py)
MRR_FROM_PRE_RERANK = {'coeus', 'tiptoe'}

# every dataset run_me.sh / check_runs.sh cover (the rebuttal tables use these)
ALL_DATASETS = ['scifact', 'msmarco', 'nq', 'fever', 'dbpedia-entity', 'msmarco-v2-25m', 'msmarco-v2-50m', 'msmarco-v2']
NEW_DATASETS = ['nq', 'fever', 'msmarco-v2-25m', 'msmarco-v2-50m', 'msmarco-v2']
# Named groups for DATASETS below
DATASET_GROUPS = {'paper': ['msmarco', 'scifact'], 'new': NEW_DATASETS, 'all': ALL_DATASETS,
                  'v2': ['msmarco-v2-25m', 'msmarco-v2-50m', 'msmarco-v2']}


def _parse_datasets(text):
    """'nq,fever' / 'nq fever' / 'new' / 'paper,nq' -> a list of dataset names, groups expanded, duplicates dropped."""
    out = []
    for item in re.split(r'[\s,]+', text.strip()):
        out += DATASET_GROUPS.get(item, [item] if item else [])
    return list(dict.fromkeys(out))


# The datasets every plot loops over (one figure per dataset, or one column per dataset in the client storage
# figure), and the ones best_configs.py prints. Pick them without editing anything:
#   DATASETS=paper python main.py            MS MARCO + SciFact only (the old default)
#   DATASETS=new python main.py              nq, fever and the three MS MARCO v2 sizes
#   DATASETS=nq,msmarco-v2-25m python main.py
#   DATASETS=all python main.py              (groups: paper, new, v2, all; mix them with names)
DATASETS = _parse_datasets(os.environ.get('DATASETS', 'all'))

DATASET_LABELS = {'msmarco': 'MS MARCO', 'scifact': 'SciFact', 'nq': 'NQ', 'fever': 'FEVER',
                  'dbpedia-entity': 'DBPedia', 'msmarco-v2-25m': 'MS MARCO v2 (25M)',
                  'msmarco-v2-50m': 'MS MARCO v2 (50M)', 'msmarco-v2': 'MS MARCO v2'}
DATASET_COLORS = {'msmarco': '#CC79A7', 'scifact': '#F0E442', 'nq': '#56B4E9', 'fever': '#999999',
                  'msmarco-v2-25m': '#882255', 'msmarco-v2-50m': '#AA4499', 'msmarco-v2': '#332288',
                  'dbpedia-entity': '#44AA99'}   # kept distinct from the method colours
DATASET_MARKERS = {'msmarco': 'o', 'scifact': 's', 'nq': 'D', 'fever': 'v', 'msmarco-v2-25m': 'P',
                   'msmarco-v2-50m': 'X', 'msmarco-v2': '*', 'dbpedia-entity': 'h'}

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
    'answer_relevancy':   {'label': 'Relevancy',   'better': 'higher'},
    'lan_time':           {'label': 'LAN Latency (s)',    'better': 'lower'},
    'wan_time':           {'label': 'Latency (s)',    'better': 'lower'},
    'total_time':         {'label': 'Computation (s)',    'better': 'lower'},
    'rerank_time':        {'label': 'Rerank Time (s)',    'better': 'lower'},
    'comm_kb':            {'label': 'Comm (KB)', 'better': 'lower'},
    'db_size_mb':         {'label': 'DB Size (MB)',       'better': 'lower'},
    'client_storage_mb':  {'label': 'Client Storage (MB)', 'better': 'lower'},
    'bs':                 {'label': 'Hash Table Size (log)',           'better': 'lower'},
    'dpb':                {'label': 'Docs per Bin (log)',       'better': 'lower'},
    'maintenance_time': {'label': 'Preprocessing (s)', 'better': 'lower'},
    'preproc_rounds': {'label': 'Preproc. Rounds', 'better': 'lower'},
    'pir_rounds': {'label': 'PIR Rounds', 'better': 'lower'},
}


# Better-direction badge that add_arrow() puts after an axis label: a ring with
# an arrow pointing up the page (higher is better) or down (lower is better).
# Drawn as shapes, not a font glyph, so the arrow is always centred in the ring.
ARROW_SIZE = 0.75         # ring diameter, as a fraction of the label's font size
ARROW_RING_LW = 0.6       # ring line width, in points
ARROW_LENGTH = 0.65       # arrow length, as a fraction of the ring diameter
ARROW_SHAFT_LW = 0.9      # arrow shaft line width, in points
ARROW_HEAD = (0.3, 0.3)   # arrow head (length, width), as fractions of the ring diameter
ARROW_GAP = 2             # gap between the label and the ring, in points


def _metric_meta(key):
    base = key.replace('_bm25', '').replace('_vec', '')
    return METRICS.get(key, METRICS.get(base, {'label': key}))


def label(key, k=K_MAIN):
    """Axis label for a run key. Split keys ('lan_time_vec') fall back to the base key.
    Use add_arrow() on the resulting label to show the better direction."""
    text = _metric_meta(key)['label'].format(k=k)
    if key.endswith('_bm25'):
        text = 'BM25 DB ' + text
    elif key.endswith('_vec'):
        text = 'Vec DB ' + text
    return text


def add_arrow(text, key, axis):
    """Put a better-direction arrow just after an axis label.

    `text` is the Text that set_xlabel / set_ylabel / fig.supxlabel returns.
    The badge (see ARROW_*) is its own artist, positioned relative to the
    label's bbox at draw time so it follows the label wherever layout puts it.
    Does nothing for swept parameters with no 'better' direction."""
    better = _metric_meta(key).get('better')
    if not better:
        return None
    color = text.get_color()
    size = text.get_fontsize() * ARROW_SIZE   # DrawingArea units are points
    c = size / 2
    sign = 1 if better == 'higher' else -1
    tip = c + sign * size * ARROW_LENGTH / 2
    tail = c - sign * size * ARROW_LENGTH / 2
    head_len, head_w = size * ARROW_HEAD[0], size * ARROW_HEAD[1]
    base = tip - sign * head_len

    badge = DrawingArea(size, size, clip=False)
    badge.add_artist(Circle((c, c), c - ARROW_RING_LW / 2, fc='none', ec=color, lw=ARROW_RING_LW))
    badge.add_artist(Line2D([c, c], [tail, base + sign * 0.2], color=color, lw=ARROW_SHAFT_LW,
                            solid_capstyle='butt'))   # runs slightly into the head, so no gap
    badge.add_artist(Polygon([(c, tip), (c - head_w / 2, base), (c + head_w / 2, base)],
                             closed=True, fc=color, ec='none'))

    if axis == 'x':   # to the right of the label
        pos = dict(xy=(1, 0.5), xybox=(ARROW_GAP, 0), box_alignment=(0, 0.5))
    else:             # above the rotated label
        pos = dict(xy=(0.5, 1), xybox=(0, ARROW_GAP), box_alignment=(0.5, 0))
    arrow = AnnotationBbox(badge, xycoords=text, boxcoords='offset points',
                           frameon=False, pad=0, **pos)
    text.figure.add_artist(arrow)
    return arrow


# ==========================================
# SIZES AND FONTS
# ==========================================

FIG_SIZE = (2.0, 1.5)   # (width, height) in inches, two of these fit side by side in ICLR's 5.5in

FONT_SIZE = 12          # axis labels and titles
TICK_SIZE = 10          # tick labels
LEGEND_SIZE = 14         # legend text
LINE_WIDTH = 1.2
MARKER_SIZE = 4.5

# Shared legend look. Position (loc / bbox_to_anchor / ncol) is set in each plot file.
LEGEND_STYLE = dict(frameon=False, fontsize=LEGEND_SIZE, handlelength=1.0,
                    handletextpad=0.25, columnspacing=0.9, borderaxespad=0.1)

# The method legend (pu.method_legend), used by every plot with a line or bar
# per method: under the figure, one line per row, each line centred
METHOD_LEGEND_ROWS = [['bins'] + TEXT_BINS + ['tree'], ['pacmann'] + EXTRA_BASELINES]
METHOD_LEGEND_LABELS = {'bins': 'PILLAR-Bin (Ours)', 'bins-text': 'PILLAR-Bin text (Ours)', 'tree': 'PILLAR-Tree (Ours)',
                        'pacmann': 'PACMANN', 'tiptoe': 'Tiptoe', 'coeus': 'Coeus'}
METHOD_LEGEND_STYLE = dict(fontsize=15, columnspacing=0.25,   # columnspacing in font-size units
                           borderpad=0, borderaxespad=0)      # no padding, so the lines sit close
METHOD_LEGEND_LINE_GAP = 2   # gap above each line (and under the figure), in points


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
