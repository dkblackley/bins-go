from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import matplotlib.pyplot as plt


def set_minimal_plot_style(
    *,
    base_style: str = "seaborn-v0_8-whitegrid",
    font_size: int = 11,
    title_size: int = 12,
    label_size: int = 11,
    legend_size: int = 10,
    dpi: int = 160,
):
    """
    Call once near the top of your script/notebook.
    Uses Matplotlib's built-in seaborn style name (no seaborn dependency).
    """
    plt.style.use(base_style)
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "legend.fontsize": legend_size,

        # Whitespace/layout
        "figure.autolayout": True,

        # Lines/markers
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,

        # Grid tuning (subtle)
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })

def _coerce_points(
    method_points: Iterable[Any],
    k_key: str,
    mrr_key: str,
    time_key: str,
) -> List[Tuple[int, Optional[float], Optional[float]]]:
    """
    Accepts a list of:
      - dicts: {"k": ..., "mrr": ..., "time_s": ...}
      - tuples: (k, mrr, time_s) or (k, mrr) or (k, time_s)
    Returns sorted list of (k, mrr, time_s), with missing metrics as None.
    """
    out: List[Tuple[int, Optional[float], Optional[float]]] = []

    for p in method_points:
        if isinstance(p, dict):
            k = int(p[k_key])
            mrr = p.get(mrr_key, None)
            time_s = p.get(time_key, None)
        else:
            # tuple-ish
            seq = list(p)
            if len(seq) < 2:
                raise ValueError(f"Point {p!r} must have at least (k, metric).")
            k = int(seq[0])
            mrr = None
            time_s = None

            if len(seq) == 2:
                # ambiguous: treat as (k, mrr) by default
                mrr = seq[1]
            else:
                mrr = seq[1]
                time_s = seq[2]

        mrr = float(mrr) if mrr is not None else None
        time_s = float(time_s) if time_s is not None else None
        out.append((k, mrr, time_s))

    out.sort(key=lambda t: t[0])
    return out


def plot_mrr_vs_k(
    results: Mapping[str, Iterable[Any]],
    *,
    title: str = "MRR vs k",
    k_key: str = "k",
    mrr_key: str = "mrr",
    time_key: str = "time_s",
    marker: str = "o",
    grid: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots MRR as a function of k for each method on the same axes.
    """
    fig, ax = plt.subplots()

    for method, points in results.items():
        pts = _coerce_points(points, k_key=k_key, mrr_key=mrr_key, time_key=time_key)
        xs = [k for (k, mrr, _t) in pts if mrr is not None]
        ys = [mrr for (_k, mrr, _t) in pts if mrr is not None]
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, label=method)

    ax.set_title(title)
    ax.set_xlabel("k (items retrieved)")
    ax.set_ylabel("MRR")
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    return fig, ax


def plot_time_vs_k(
    results: Mapping[str, Iterable[Any]],
    *,
    title: str = "Time vs k",
    k_key: str = "k",
    mrr_key: str = "mrr",
    time_key: str = "time_s",
    marker: str = "o",
    grid: bool = True,
    log_y: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots time (seconds) as a function of k for each method on the same axes.
    """
    fig, ax = plt.subplots()

    for method, points in results.items():
        pts = _coerce_points(points, k_key=k_key, mrr_key=mrr_key, time_key=time_key)
        xs = [k for (k, _mrr, t) in pts if t is not None]
        ys = [t for (_k, _mrr, t) in pts if t is not None]
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, label=method)

    ax.set_title(title)
    ax.set_xlabel("k (items retrieved)")
    ax.set_ylabel("Time (s)")
    if log_y:
        ax.set_yscale("log")
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    return fig, ax


def plot_mrr_time_tradeoff(
    results: Mapping[str, Iterable[Any]],
    *,
    title: str = "MRR vs Time (tradeoff)",
    k_key: str = "k",
    mrr_key: str = "mrr",
    time_key: str = "time_s",
    marker: str = "o",
    annotate_k: bool = False,
    grid: bool = True,
    log_x: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots MRR against time to show quality/latency tradeoffs.
    Each point corresponds to a specific k.
    """
    fig, ax = plt.subplots()

    for method, points in results.items():
        pts = _coerce_points(points, k_key=k_key, mrr_key=mrr_key, time_key=time_key)
        xs = [t for (_k, mrr, t) in pts if (mrr is not None and t is not None)]
        ys = [mrr for (_k, mrr, t) in pts if (mrr is not None and t is not None)]
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, label=method)

        if annotate_k:
            for k, mrr, t in pts:
                if mrr is None or t is None:
                    continue
                ax.annotate(str(k), (t, mrr), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MRR")
    if log_x:
        ax.set_xscale("log")
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    return fig, ax


results = {
    "Bins": [
        {"k": 10, "mrr": 0.051, "time_s": 2.41 },
        {"k": 50, "mrr": 0.072, "time_s": 5.47 },
        {"k": 100, "mrr": 0.111, "time_s": 8.26},
        {"k": 500, "mrr": 0.171, "time_s": 22},
        {"k": 1000, "mrr": 0.180, "time_s": 45},
    ],
    "Pacmann": [
        {"k": 10, "mrr": 0.170, "time_s": 81.5},
        {"k": 50, "mrr": 0.175, "time_s": 82},
        {"k": 100, "mrr": 0.174, "time_s": 90},
        {"k": 500, "mrr": 0.181, "time_s": 97},
        {"k": 1000, "mrr": 0.18, "time_s": 107},
    ],

    # "tree": [
    #     {"k": 10, "mrr": 0.190, "time_s": 1.2},
    #     {"k": 50, "mrr": 0.235, "time_s": 3.8},
    #     {"k": 100, "mrr": 0.244, "time_s": 7.1},
    #     {"k": 500, "mrr": 0.221, "time_s": 0.11},
    #     {"k": 1000, "mrr": 0.221, "time_s": 0.11},
    # ],
    #
}

set_minimal_plot_style()

fig1, ax1 = plot_mrr_vs_k(results)
fig2, ax2 = plot_time_vs_k(results, log_y=True)  # log_y helps if time spans large range
fig3, ax3 = plot_mrr_time_tradeoff(results, annotate_k=True)

plt.show()
