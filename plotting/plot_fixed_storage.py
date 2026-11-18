"""
WAN latency and MRR@10 at fixed client storage, swept: x = a client storage S (GB), rows = WAN and MRR@10, a
column per dataset (PILLAR-Bin vs PACMANN on MS MARCO and MS MARCO v2 25M / 50M).

    figures/fixed_storage_sweep_wan_time.pdf, figures/fixed_storage_sweep_mrr.pdf
                                         (SPLIT_ROWS: one figure per row, each with its own suptitle and the
                                          dataset names over its panels; False = one figure, FIG_NAME.pdf)

At each S every config of a method whose client storage is within S +/- TOL_GB (half the step) is a candidate (MODE =
'budget': storage <= S instead), and the one with the lowest WAN is the method's point at S -- WAN rounded
down to 2 dp, so configs at the same WAN go to the highest MRR@10 (then Recall@10), as fixed_storage.py /
rebuttal_final.fixed_storage. One point per method per S, joined by a line per method; in each panel the
better method at each S (lower WAN / higher MRR@10; a tie goes to the lower WAN) is bigger, outlined and on top.
PICK_BY_ROW = True: the MRR@10 row picks the best-MRR@10 config in the window instead, the WAN row still the fastest. The terminal gets the numbers as tables, then every pick's folder.

The CONFIG block is the same as plot_fixed_mrr.py's; the drawing lives in fixed_sweep.py.

    python plot_fixed_storage.py
or from main.py:
    plot_fixed_storage.make_plots(nested_data)
"""

import frozen_values as fv
import globals as g
import load_results
import fixed_sweep

# ======================================================================
# CONFIG: WHAT IS PLOTTED
# ======================================================================

K = g.K_MAIN
DATASETS = ['msmarco', 'msmarco-v2-25m', 'msmarco-v2-50m']
METHODS = ['bins', 'pacmann']
BINS_FILTER = {}
ROWS = ['wan_time', 'mrr']          # y key of each panel row
NEED = ['mrr', 'recall', 'wan_time', 'client_storage_mb']
MIN_MRR = 0.01                      # drop grid-search leftovers whose rounded-down MRR@10 is below this

# Client storage targets (GB) per dataset: an even step, and at every target both methods have a config (each window
# is half the step, TOL_GB, so every config falls in its nearest target). Bin has the lower WAN at every target, and
# the higher MRR@10 (with PICK_BY_ROW = False) at all 4 on MS MARCO and at 3 of 4 on 25M (not 2 GB) and 50M (not
# 3 GB). With PICK_BY_ROW = True PACMANN has the better MRR@10 almost everywhere (checked on the local results,
# 2026-11-18). Other even grids where Bin wins WAN at every point: MS MARCO 1-1.75 by 0.25, 25M 1-2.5 by 0.5,
# 50M 2-5 by 1. A dataset not listed gets auto_targets(): AUTO_ANCHOR's sizes to 0.1 GB.
TARGETS = {
    'msmarco':        [0.5, 1.0, 1.5, 2.0],
    'msmarco-v2-25m': [0.5, 1.0, 1.5, 2.0],
    'msmarco-v2-50m': [1.0, 2.0, 3.0, 4.0],
}
# TARGETS = {
#     'msmarco':        [0.8, 1.25, 1.5, 1.75, 2.0],
#     'msmarco-v2-25m': [0.8, 1.25, 1.5, 1.75, 2.0],
#     'msmarco-v2-50m': [0.8, 1.25, 1.5, 1.75, 2.0],
# }
AUTO_ANCHOR = 'pacmann'
MODE = 'window'      # 'window': storage within S +/- TOL_GB; 'budget': storage <= S
TOL_GB = {'msmarco': 0.25, 'msmarco-v2-25m': 0.25, 'msmarco-v2-50m': 0.5}   # window per dataset (GB)
TOL_GB_DEFAULT = 0.1                                                         # a dataset not listed
# True: each panel row picks for itself -- the MRR@10 row takes the config in the window with the best MRR@10
# (rounded down to 2 dp; ties to the lower WAN), the WAN row the lowest WAN as below. False: one pick per method
# and target (lowest WAN) for every row, so the two rows show the same config.
PICK_BY_ROW = False
MB_PER_GB = 1024     # ClientStorageMB is MiB, so GB = GiB (as fixed_storage.py / rebuttal_final.py)

# ======================================================================
# CONFIG: THE AXES
# ======================================================================

X_LABEL = 'Client Storage (GB)'
X_PAD = 0.2          # x margin either side of the first / last target, in units of the gap between targets
X_CATEGORICAL = True    # True: targets evenly spaced in order (uneven targets like 3.3 / 3.7 GB stop colliding)
X_TICK_ROTATION = 0      # degrees, e.g. 45 when the tick labels run into each other
X_TABLE = '{t:g} GB'


def x_tick(v):
    return f'{v:g}'


AXES = {
    'wan_time':          dict(label='WAN Latency', fmt='time', log=False),
    'lan_time':          dict(label='LAN Latency', fmt='time', log=True),
    'comm_kb':           dict(label='Data Sent', fmt='bytes', base='KB', log=True),
    'mrr':               dict(fmt='%.2f'),   # tick labels at 3 dp, all the same width (no `lim`: Y_TICKS snaps them)
    'recall':            dict(fmt='%.2f'),
    # ('msmarco', 'mrr'): dict(lim=(0, 0.2)),
}
Y_TICKS = 4          # y gridlines per panel: the bottom one on the lowest point, the top one on the highest, evenly
                     # spaced between (log-even on a log axis); None = the old automatic limits and ticks
Y_PAD = 0.25         # y margin below the lowest / above the highest point, in units of the gap between y gridlines
SHARE_Y_ROW = False
Y_FIT = 'all'

# ======================================================================
# CONFIG: POINTS AND LINES
# ======================================================================

POINTS = 'picks'     # 'picks': one point per method per target; 'fade': also every other matching config, faint
CANDIDATE_X = 'target'   # (POINTS = 'fade' only) faint configs at x = the target or at their own client storage ('actual')
FADE_ALPHA = 0.25
FADE_SIZE = 0.5
DODGE = 0            # 0: every point exactly at its target; e.g. 0.15 spreads the methods round it by that
                     # fraction of the gap between targets

MARKER = 'o'
PER_METHOD_MARKERS = False
MARKER_SIZE = 5
# the winner at each target: best y (lowest WAN / highest MRR@10) at WIN_DP decimals, a tie there goes to the
# better WIN_TIE_BREAK (lower WAN). It gets a black outline, a bigger marker and is drawn on top; the rest are plain.
WIN_DP = 3           # 3, like the MRR@10 tick labels: at 2, v2 50M's 0.0447 (Bin) and 0.0464 (PACMANN) both read
                     # 0.04, tie, and the tie went to Bin's lower WAN although PACMANN's point is visibly higher
WIN_TIE_BREAK = 'wan_time'   # None: a tie goes to the better raw y
WIN_SIZE = 1.3       # x MARKER_SIZE
WIN_EDGE = 0.9       # black outline width (0 = none)
WIN_ALPHA = 1.0
LOSE_SIZE = 1.0      # x MARKER_SIZE
LOSE_ALPHA = 1.0

LINES = True         # join each method's picks
LINE_WIDTH = g.LINE_WIDTH
LINE_STYLE = '-'
LINE_ALPHA = 1.0
LINE_ACROSS_GAPS = False   # False: the line breaks where a method has no config at a target

Z_FADED = 2          # drawing order: faint configs, lines, the other points, the winner on top
Z_LINES = 3
Z_LOSE = 4
Z_WIN = 5
STACK_ORDER = None   # within a layer, first = on top; None = g.METHOD_ORDER

# ======================================================================
# CONFIG: LAYOUT, TITLES, LEGEND, OUTPUT
# ======================================================================

SPLIT_ROWS = True    # one figure per ROWS entry -> figures/<FIG_NAME>_<row>.pdf
FIGURE_GROUPS = None   # e.g. [['msmarco', 'msmarco-v2-25m'], ['msmarco-v2-50m']] for one figure per group
PER_DATASET = False
N_COLS = 3
PANEL_SIZE = (1.75, 1.35)
WSPACE = 0.01
HSPACE = 0.02

PANEL_TITLE = '{dataset}'
TITLE_SIZE = g.FONT_SIZE
SUPTITLE = 'PPRAG {ylabel} Comparison at Fixed Client Storage'   # {ylabel}: the row's y label (SPLIT_ROWS)
SUPTITLE_YLABEL = {'wan_time': 'Latency'}   # {ylabel} in the suptitle only; the y axis keeps 'WAN Latency'
SUPTITLE_SIZE = 15
SUPTITLE_OFFSET = 0 # points of white space between the suptitle and the panels (room for the tall y label)

X_LABEL_SIZE = g.FONT_SIZE
Y_LABEL_SIZE = g.FONT_SIZE
TICK_SIZE = 8
X_LABEL_EVERY_ROW = False
Y_LABEL_EVERY_COL = False
ALIGN_LABELS = True
SHOW_ARROWS = True
LOG_SUFFIX = ' (log)'
GRID = True

LEGEND = True
LEGEND_ROWS = [['bins', 'pacmann']]
LEGEND_MARKER_SIZE = 6
LEGEND_STYLE = dict(handlelength=1.5, fontsize=12)

FIG_NAME = 'fixed_storage_sweep'
TABLE_TITLE = lambda: (('best MRR@10 (MRR row) / lowest WAN (WAN row)' if PICK_BY_ROW else 'lowest WAN') +
                       ' at client storage ' + ('<= S' if MODE == 'budget' else 'S ± the window (TOL_GB)'))

PRINT_TABLE = True
WRITE_FILES = g.WRITE_TEXT_FILES
LOG_POINTS = False


# ======================================================================
# MATCHING (called by fixed_sweep)
# ======================================================================

def gb(r):
    return r['client_storage_mb'] / MB_PER_GB


def tol_gb(dataset):
    return TOL_GB.get(dataset, TOL_GB_DEFAULT) if isinstance(TOL_GB, dict) else TOL_GB


def candidates(runs, s, dataset=None):
    if MODE == 'budget':
        return [r for r in runs if gb(r) <= s + 1e-9]
    return [r for r in runs if abs(gb(r) - s) <= tol_gb(dataset) + 1e-9]


def pick(cands, y_key=None):
    if PICK_BY_ROW and y_key in ('mrr', 'recall'):   # best quality in the window, ties to the lower WAN
        return min(cands, key=lambda r: (-fv.rkey(r[y_key]), r['wan_time'], -r['mrr'], -r['recall']))
    return min(cands, key=lambda r: (fv.rkey(r['wan_time']), -r['mrr'], -r['recall'], r['wan_time']))


def run_x(r):
    return gb(r)


def auto_targets(per_method):
    runs = per_method.get(AUTO_ANCHOR) or [r for rs in per_method.values() for r in rs]
    return sorted({round(gb(r), 1) for r in runs})


def make_plots(nested_data, datasets=None, rows=None, name=None):
    import sys
    cfg, rows, name = sys.modules[__name__], list(rows or ROWS), name or FIG_NAME
    if not SPLIT_ROWS:
        return fixed_sweep.make_figures(nested_data, cfg, datasets, rows, name)
    return [p for y in rows for p in fixed_sweep.make_figures(nested_data, cfg, datasets, [y], f'{name}_{y}')]


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results(g.RESULTS_DIR))
