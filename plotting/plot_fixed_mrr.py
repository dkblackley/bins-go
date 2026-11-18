"""
WAN latency at fixed MRR@10, swept: x = a fixed MRR@10 value T, y = WAN, one panel per dataset.

    figures/fixed_mrr_sweep_msmarco_scifact.pdf, ..._fever_nq.pdf, ..._msmarco-v2-25m_msmarco-v2-50m.pdf
                                     (FIGURE_GROUPS: two datasets side by side per figure)

At each T every config of a method whose MRR@10 matches T is a candidate, and the one with the lowest WAN
(PICK_BY) is the method's point at T. "Matches" is frozen_values.py's rule: MRR@10 rounded DOWN to 2 decimals is
within T +/- TOL (0.01), so T = 0.15 takes every config from 0.14 to 0.16 (raw 0.140 ... 0.169). The picks are
one point per method per T, joined by a line per method (MATCH = 'at_least' takes every config at or above T
instead); at each T the fastest method's point is bigger, outlined and on top, the others plain.
The terminal gets the same numbers as tables (a row per method, a column per T), then every pick's folder.

Everything you'd want to change is in the CONFIG block below; the drawing lives in fixed_sweep.py and is shared
with plot_fixed_storage.py.

    python plot_fixed_mrr.py
or from main.py:
    plot_fixed_mrr.make_plots(nested_data)
    plot_fixed_mrr.make_plots(nested_data, rows=['wan_time', 'client_storage_mb'], name='fixed_mrr_wan_storage')
"""

import frozen_values as fv
import globals as g
import load_results
import fixed_sweep

# ======================================================================
# CONFIG: WHAT IS PLOTTED
# ======================================================================

K = g.K_MAIN
DATASETS = g.REBUTTAL_DATASETS
METHODS = ['bins', 'tree', 'pacmann']   # None = g.METHOD_ORDER (Tiptoe / Coeus too)
BINS_FILTER = {}                        # e.g. {'vec': 1}: only single-DB bins runs
ROWS = ['wan_time']                     # y key of each panel row, e.g. ['wan_time', 'client_storage_mb']
NEED = ['mrr']                          # keys a run must have to be used at all (ROWS are added)
MIN_MRR = 0.01                          # drop runs whose rounded-down MRR@10 is below this (0 = keep all)

# Fixed MRR@10 values per dataset: an even step, and at every value all three methods have a config and PILLAR
# (Tree or Bin) has the lowest WAN, with MATCH = 'window' (checked on the local results, 2026-11-17). NQ only works
# at a 0.01 step (PACMANN is fastest at 0.28-0.38 and has no config at 0.05 / 0.15 / 0.25); with MATCH = 'at_least'
# NQ 0.05 ... 0.25 works too. A dataset not listed gets auto_targets(): every multiple of AUTO_STEP that every
# method has a config at.
TARGETS = {
    'msmarco':        [0.10, 0.15, 0.20, 0.25],
    # 'scifact':      [0.34, 0.44, 0.54, 0.64],   # 0.1 step at +/- 0.01: Bin, Tree, Bin, Bin
    # 'scifact':      [0.4, 0.5, 0.6, 0.7],       # 0.1 step, TOL_SCALE 1.1 (+/- 10%): all Bin
    # 'fever':        [0.38, 0.48, 0.58, 0.68],   # 0.1 step: Bin, Bin, Tree, Tree (all Bin: 0.35 ... 0.65)
    'scifact':        [0.50, 0.55, 0.60, 0.65],   # 0.05 step, like Recall@10: all Bin (at +/- 0.01 and at TOL_SCALE's 10%)
    'fever':          [0.45, 0.50, 0.55, 0.60],   # 0.05 step, like Recall@10: all Bin (0.50 ... 0.65 too)
    'nq':             [0.05, 0.10, 0.15, 0.20],   # 0.05 step (all Bin at +/- 0.01 and at TOL_BY_DATASET's +/- 0.02)
    'msmarco-v2-25m': [0.05, 0.10, 0.15, 0.20],
    'msmarco-v2-50m': [0.04, 0.08, 0.12, 0.16],
}
AUTO_STEP = 0.05
MATCH = 'window'     # 'window': rounded-down MRR@10 within T +/- TOL; 'at_least': rounded-down MRR@10 >= T
                     # (what the posted rebuttal tables used, so a fast high-MRR config counts at every lower T)
TOL = 0.01           # match window round T, on the rounded-down MRR@10 (0 = exactly T at 2 dp)
# per dataset: widen the window to T +/- (scale - 1) x T, e.g. 1.1 -> T = 0.5 takes 0.45 ... 0.55 (never narrower
# than TOL; still on the rounded-down value). A dataset not listed keeps T +/- TOL.
TOL_BY_DATASET = {'nq': 0.02}   # per dataset, in place of TOL (nq: round 0.05-step Recall@10 needs +/- 0.02, 2026-11-18)
TOL_SCALE = {'scifact': 1.1}   # scifact: no 0.1-step grid on round values at +/- 0.01 (2026-11-18)
DP = 2               # MRR@10 is rounded down to this many decimals before matching
PICK_BY = 'wan_time' # the candidate with the lowest of this is the pick (ties: higher MRR@10)

# ======================================================================
# CONFIG: THE AXES
# ======================================================================

X_LABEL = 'Fixed MRR@10'
X_PAD = 0.2          # x margin either side of the first / last target, in units of the gap between targets
X_CATEGORICAL = False    # True: targets evenly spaced in order (uneven targets like 3.3 / 3.7 GB stop colliding)
X_TICK_ROTATION = 0      # degrees, e.g. 45 when the tick labels run into each other
X_TABLE = 'MRR@10 {t:.2f}'   # target name in the terminal table


def x_tick(v):
    return f'{v:.2f}'


# Same keys as plot_dataset_costs.AXES: label, lim, log, fmt, base, ticks; (dataset, key) wins for that dataset.
AXES = {
    'wan_time':          dict(label='WAN Latency', fmt='time', log=False),
    'lan_time':          dict(label='LAN Latency', fmt='time', log=True),
    'comm_kb':           dict(label='Data Sent', fmt='bytes', base='KB', log=True),
    'client_storage_mb': dict(label='Client Storage', fmt='bytes', base='MB', log=True),
    'mrr':               dict(fmt='plain', lim=(0, None)),
    'recall':            dict(fmt='plain', lim=(0, None)),
    # ('msmarco', 'wan_time'): dict(lim=(0.03, 3)),
}
Y_TICKS = 4          # y gridlines per panel: the bottom one on the lowest point, the top one on the highest, evenly
                     # spaced between (log-even on a log axis); None = the old automatic limits and ticks
Y_PAD = 0.25         # y margin below the lowest / above the highest point, in units of the gap between y gridlines
SHARE_Y_ROW = False  # same y limits across a row (the panels of a row are different datasets)
Y_FIT = 'all'        # 'all': y limits fit the faded candidates too; 'picks': only the picks

# ======================================================================
# CONFIG: POINTS AND LINES
# ======================================================================

POINTS = 'picks'     # 'picks': one point per method per target; 'fade': also every other matching config, faint
CANDIDATE_X = 'target'   # (POINTS = 'fade' only) faint configs at x = the target or at their own MRR@10 ('actual')
FADE_ALPHA = 0.25
FADE_SIZE = 0.5
DODGE = 0            # 0: every point exactly at its target; e.g. 0.15 spreads the methods round it by that
                     # fraction of the gap between targets

MARKER = 'o'
PER_METHOD_MARKERS = False
MARKER_SIZE = 5
# the winner at each target: best y (lowest WAN / highest MRR@10) at WIN_DP decimals, a tie there goes to the
# better WIN_TIE_BREAK (lower WAN). It gets a black outline, a bigger marker and is drawn on top; the rest are plain.
WIN_DP = 2
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

# one figure per group, the datasets side by side -> figures/<FIG_NAME>_<d1>_<d2>.pdf; None = one figure with every
# dataset (N_COLS per line)
FIGURE_GROUPS = [['msmarco', 'scifact'], ['fever', 'nq'], ['msmarco-v2-25m', 'msmarco-v2-50m']]
PER_DATASET = False  # True: one figure per dataset instead (wins over FIGURE_GROUPS)
N_COLS = 2           # datasets per line of panels (None = all on one line)
PANEL_SIZE = (1.75, 1.35)
WSPACE = 0.01
HSPACE = 0.01

PANEL_TITLE = '{dataset}'   # over each dataset's top panel; None for none
TITLE_SIZE = g.FONT_SIZE
SUPTITLE = 'PPRAG Latency at Fixed MRR@10'   # '{dataset}' works with PER_DATASET; None for none
SUPTITLE_ON = ['msmarco', 'scifact']   # only on the figure(s) holding these datasets (the top one); None = every figure
SUPTITLE_SIZE = 15
SUPTITLE_OFFSET = 0          # points of white space between the suptitle and the panels; the figure grows to fit it

X_LABEL_SIZE = g.FONT_SIZE
Y_LABEL_SIZE = g.FONT_SIZE
TICK_SIZE = 8
X_LABEL_EVERY_ROW = False   # False: only the bottom panel of each column
Y_LABEL_EVERY_COL = False   # False: only the left column
Y_LABEL_Y = 0.35             # y-label centre in axes fractions: 0.5 = middle, lower moves it down
ALIGN_LABELS = True
SHOW_ARROWS = True
LOG_SUFFIX = ' (log)'
GRID = True

LEGEND = True
LEGEND_ON = ['msmarco-v2-25m', 'msmarco-v2-50m']   # only under the figure(s) holding these datasets (the bottom one); None = every figure
LEGEND_ROWS = None          # None = g.METHOD_LEGEND_ROWS; or e.g. [['bins', 'tree', 'pacmann']]
LEGEND_MARKER_SIZE = 6
LEGEND_STYLE = dict(handlelength=1.5, fontsize=12)

FIG_NAME = 'fixed_mrr_sweep'
TABLE_TITLE = lambda: ('WAN at fixed MRR@10 (lowest WAN at MRR@10 >= T)' if MATCH == 'at_least' else
                       'WAN at fixed MRR@10 (lowest WAN within ± %.2f)' % TOL + ''.join(f'; {d} ± {x:.2f}' for d, x in TOL_BY_DATASET.items())
                       + ''.join(f'; {d} ± {round((f - 1) * 100)}%' for d, f in TOL_SCALE.items()))   # a string works too

PRINT_TABLE = True
WRITE_FILES = g.WRITE_TEXT_FILES   # also figures/<FIG_NAME>.md
LOG_POINTS = False


# ======================================================================
# MATCHING (called by fixed_sweep)
# ======================================================================

def window_steps(t, dataset=None):
    """half-width of the match window round t, in units of 10^-DP"""
    scale, tol = TOL_SCALE.get(dataset), TOL_BY_DATASET.get(dataset, TOL)
    return max(fv.rkey(tol, DP), fv.rkey((scale - 1) * t, DP)) if scale else fv.rkey(tol, DP)


def candidates(runs, t, dataset=None):
    steps, at = window_steps(t, dataset), fv.rkey(t, DP)
    if MATCH == 'at_least':
        return [r for r in runs if fv.rkey(r['mrr'], DP) >= at]
    return [r for r in runs if abs(fv.rkey(r['mrr'], DP) - at) <= steps]


def pick(cands, y_key=None):
    return min(cands, key=lambda r: (r[PICK_BY], -r['mrr']))


def run_x(r):
    return r['mrr']


def auto_targets(per_method):
    top = max(r['mrr'] for runs in per_method.values() for r in runs)
    grid = [round(AUTO_STEP * i, DP) for i in range(1, int(top / AUTO_STEP) + 2)]
    return [t for t in grid if all(candidates(runs, t) for runs in per_method.values())]


def make_plots(nested_data, datasets=None, rows=None, name=None):
    import sys
    return fixed_sweep.make_figures(nested_data, sys.modules[__name__], datasets, rows, name)


if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()
    make_plots(load_results.load_results(g.RESULTS_DIR, only_grid=False))   # every run, not only the current grid
