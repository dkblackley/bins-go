"""
Loads the results once and writes every figure to figures/. Comment out a line
to skip a figure; each plot file also runs on its own (python plot_db_size.py).

    python main.py                         # g.DATASETS, by default all of them (g.ALL_DATASETS)
    DATASETS=paper python main.py          # MS MARCO + SciFact only
    DATASETS=new python main.py            # nq, fever, msmarco-v2-25m / -50m / full
    DATASETS=nq,msmarco-v2-25m python main.py   (groups: paper, new, v2, all; see globals.py)
    RESULTS_DIR=/some/other/dir python main.py
"""

import best_configs
import globals as g
import load_results
import plot_best_histograms
import plot_bins_ablation
import plot_bins_ablation_grid
import plot_bins_ablation_split
import plot_bins_ablation_stacked
import plot_bins_comm_vs_bs
import plot_bins_stages
import plot_bins_vec_db
import fixed_recall
import fixed_score
import rebuttal_numbers
import rebuttal_tables
import plot_db_size
import plot_metric_vs_latency
import plot_bins_client_storage
import plot_bins_client_storage_split
import plot_pareto_grid
import plot_pareto_single
import plot_quality_vs_cost
import plot_tree_stages
import plot_pareto_frontier
import plot_pareto_vs_latency

if __name__ == '__main__':
    g.setup_logging()
    g.setup_style()

    nested_data = load_results.load_results(g.RESULTS_DIR)

    best_configs.best_configs(nested_data)            # terminal: the 5 picks per method/dataset and their stats
                                                      # (also figures/best_configs.csv)

    # plot_metric_vs_latency.make_plots(nested_data)    # quality vs latency, per dataset
    # plot_best_histograms.make_plots(nested_data)     # best of the selected configs, per metric
    # plot_pareto_frontier.plot_pareto_frontier(nested_data)
    # plot_pareto_vs_latency.make_plots(nested_data)    # same, Pareto frontier, per dataset
    # plot_pareto_single.make_plots(nested_data)        # one figure per dataset per plot_pareto_single.PAIRS
    # plot_bins_ablation_grid.make_plots(nested_data)  # bins: WAN, LAN, compute, comm vs bs and vs dpb
    # plot_bins_ablation_split.make_plots(nested_data)  # bins: storage, preproc, quality, a dataset each
    # plot_bins_vec_db.make_plots(nested_data)         # bins: the embedding DB, quality and cost
    # plot_bins_ablation_stacked.plot_bins_ablation_stacked(nested_data)
    # plot_pareto_grid.make_plots(nested_data)
    plot_pareto_grid.make_comparison_plot(nested_data)   # client storage vs MRR/Recall, a column per dataset

    # the one-PDF-per-metric versions the two grids above replace
    # plot_db_size.make_plots(nested_data)             # bins DB size vs PACMANN baseline
    # plot_bins_ablation.make_plots(nested_data)       # bins: every metric vs bs / dpb
    # plot_bins_comm_vs_bs.make_plots(nested_data)     # bins: communication vs bs, dpb=10
    # plot_bins_stages.make_plots(nested_data)         # bins: 1-stage vs 2-stage
    # plot_bins_client_storage.make_plots(nested_data)  # bins: client storage, vec0 vs vec1
    # plot_bins_client_storage_split.make_plots(nested_data)  # bins: client storage, vec1 vs vec0 and vec0's ID / embedding DB hints
    # plot_tree_stages.make_plots(nested_data)          # tree: latency split by PIR stage
    # plot_quality_vs_cost.make_plots(nested_data)      # all: quality vs preproc/storage cost

    # fixed_recall.fixed_recall(nested_data)              # tables: each method's cost at one shared recall

    # rebuttal: best 5 configs per method/dataset on each cost knee (client storage, WAN, LAN) -> figures/rebuttal_best5_k10.md
    #rebuttal_tables.make_tables(nested_data)
    # rebuttal: every [FILL] number existing results can give (items A-K, incl. fixed_score's equal-quality
    # comparison) -> figures/rebuttal_numbers.md; per-query files it still needs are listed at the end
    # rebuttal_numbers._add_stage3_batch(nested_data)
    # rebuttal_numbers.rebuttal_numbers(nested_data)

    print(f"Done, figures in {g.FIGURE_DIR}")
