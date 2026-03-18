from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from .result_parser import ResultRecord, SOLVER_ORDER, LEARNED_SOLVERS, discover_result_records

SWEEP_CASES = (1, 2)
FIXED_COUNT_CASES = (3, 4, 5)
SOLVER_LABEL = {"gnn": "GNN", "cnn": "CNN", "mlp": "MLP", "stlpy": "STLPY"}
CASE_LABEL = {
    1: "Case 1",
    2: "Case 2",
    3: "Case 3",
    4: "Case 4",
    5: "Case 5",
}
SOLVER_COLOR = {
    "gnn": "tab:blue",
    "cnn": "tab:orange",
    "mlp": "tab:green",
    "stlpy": "tab:red",
}
SOLVER_MARKER = {
    "gnn": "o",
    "cnn": "s",
    "mlp": "^",
    "stlpy": "x",
}


def _blend_with_white(color: str, amount: float) -> Tuple[float, float, float]:
    """Return a lighter variant of a matplotlib color."""
    r, g, b = to_rgb(color)
    amount = max(0.0, min(1.0, amount))
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def _case_style(solver: str, example_id: int) -> Dict[str, object]:
    """
    Case-aware styling for sweep plots.

    Keep solver identity visually consistent while making different cases
    clearly distinguishable.
    """
    base_color = SOLVER_COLOR.get(solver, "tab:gray")
    base_marker = SOLVER_MARKER.get(solver, "o")

    if int(example_id) == 1:
        return {
            "color": base_color,
            "marker": base_marker,
            "linestyle": "-",
            "markerfacecolor": base_color,
        }

    return {
        "color": _blend_with_white(base_color, 0.35),
        "marker": "D",
        "linestyle": "--",
        "markerfacecolor": "white",
    }


def _group(records: List[ResultRecord]) -> Dict[Tuple[int, int, str], ResultRecord]:
    out: Dict[Tuple[int, int, str], ResultRecord] = {}
    for r in records:
        out[(r.example_id, r.obstacles, r.solver)] = r
    return out


def _binary_count(example_id: int, obstacles: int) -> Optional[int]:
    """
    Binary-variable counts for the current benchmark settings:

      Case 1: B1(n) = 4n(T+1) + (T+1) = 84 n + 21
      Case 2: B2(n) = 4n(T+1) + 2(T-tau+1) + (T+1) = 104 n + 68
      Case 3: fixed n=4  -> (4n+2)(T+1) = 468
      Case 4: fixed n=3  -> (4n+2*types)(T+1) = 558
      Case 5: fixed n=5  -> 4n(T+1)+2*5(T+1)+(T+1) =1116
    """
    ex = int(example_id)
    n = int(obstacles)

    if ex == 1:
        return 84 * n + 21
    if ex == 2:
        return 104 * n + 68
    if ex == 3:
        return 468
    if ex == 4:
        return 558
    if ex == 5:
        return 1116
    return None


def _case_annotation(example_id: int, obstacles: int) -> str:
    ex = int(example_id)
    obs = int(obstacles)
    if ex in SWEEP_CASES:
        return f"C{ex}-o{obs}"
    return f"C{ex}"


def _compact_figsize(kind: str) -> Tuple[float, float]:
    """
    Compact figure sizes for paper-friendly plots.
    Width is kept close to the original to preserve readability,
    while height is reduced to save vertical space.
    """
    if kind == "sweep_line":
        return (7.4, 3.2)
    if kind == "sweep_speedup":
        return (7.6, 3.25)
    if kind == "binary_scatter":
        return (8.0, 3.5)
    if kind == "cost_bars":
        return (8.2, 3.2)
    if kind == "fixed_bars":
        return (7.8, 3.1)
    if kind == "fixed_speedup":
        return (7.8, 3.1)
    return (7.6, 3.2)


def _set_reasonable_y_limits(
    ax,
    values: List[float],
    *,
    metric: str,
    include_reference: Optional[float] = None,
) -> None:
    if not values:
        return

    ymin = min(values)
    ymax = max(values)

    if include_reference is not None:
        ymin = min(ymin, include_reference)
        ymax = max(ymax, include_reference)

    # Success-rate plots: compact, but not misleadingly tight.
    if metric == "pure_success_rate":
        lo = max(0.0, ymin - 0.035)
        hi = min(1.02, ymax + 0.02)

        # Avoid overly thin ranges in near-constant cases.
        if hi - lo < 0.10:
            center = 0.5 * (lo + hi)
            lo = max(0.0, center - 0.05)
            hi = min(1.02, center + 0.05)

        ax.set_ylim(lo, hi)
        return

    span = ymax - ymin
    if span <= 1e-12:
        if metric == "solved_k":
            pad = 0.5
        elif metric == "speedup":
            pad = max(0.15 * max(abs(ymax), 1.0), 0.25)
        elif metric == "cost":
            pad = max(0.08 * max(abs(ymax), 1.0), 0.05)
        else:
            pad = max(0.08 * max(abs(ymax), 1.0), 0.05)
    else:
        if metric == "speedup":
            pad = 0.08 * span
        elif metric == "solved_k":
            pad = 0.10 * span
        elif metric == "cost":
            pad = 0.08 * span
        else:
            pad = 0.09 * span

    lo = ymin - pad
    hi = ymax + pad

    if metric in {"speedup", "cost", "solved_k"}:
        lo = max(0.0, lo)

    ax.set_ylim(lo, hi)


def _speedup_vs_stlpy(
    grouped: Dict[Tuple[int, int, str], ResultRecord],
    example_id: int,
    obstacles: int,
    solver: str,
) -> Optional[float]:
    if solver == "stlpy":
        return None

    base = grouped.get((example_id, obstacles, "stlpy"))
    cur = grouped.get((example_id, obstacles, solver))
    if base is None or cur is None:
        return None
    if base.time_total_mean is None or cur.time_total_mean is None:
        return None
    if cur.time_total_mean <= 0:
        return None
    return float(base.time_total_mean) / float(cur.time_total_mean)


def _save_sweep_line_plot(
    records: List[ResultRecord],
    out_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
    solvers: List[str],
) -> Path:
    grouped = _group(records)

    if metric == "pure_success_rate":
        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=(8.6, 3.15),
            gridspec_kw={"width_ratios": [2.25, 1.0], "wspace": 0.05},
        )
        plotted_left: List[float] = []
        plotted_right: List[float] = []

        # Left: Cases 1 and 2 with obstacle-count sweep
        for ex in SWEEP_CASES:
            obs_values = sorted({r.obstacles for r in records if r.example_id == ex})
            for solver in solvers:
                xs: List[int] = []
                ys: List[float] = []
                for obs in obs_values:
                    r = grouped.get((ex, obs, solver))
                    if r is None:
                        continue
                    value = getattr(r, metric)
                    if value is None:
                        continue
                    xs.append(obs)
                    ys.append(float(value))
                if xs:
                    plotted_left.extend(ys)
                    style = _case_style(solver, ex)
                    ax_left.plot(
                        xs,
                        ys,
                        marker=style["marker"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        markerfacecolor=style["markerfacecolor"],
                        markeredgecolor=style["color"],
                        linewidth=1.8,
                        markersize=6.5,
                        label=f"{CASE_LABEL.get(ex, f'Case {ex}')}-{SOLVER_LABEL.get(solver, solver)}",
                    )

        ax_left.set_xlabel("Obstacle count (n_obs)")
        ax_left.set_ylabel(ylabel)
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(fontsize=7.6, ncol=2, columnspacing=0.9, handletextpad=0.5)
        _set_reasonable_y_limits(ax_left, plotted_left, metric="pure_success_rate")

        # Right: fixed-count Cases 3, 4, 5
        x_fixed = [0, 1, 2]
        fixed_case_markers = {3: "o", 4: "s", 5: "^"}

        for solver in solvers:
            ys: List[float] = []
            xs: List[int] = []
            exs: List[int] = []
            for xpos, ex in zip(x_fixed, FIXED_COUNT_CASES):
                candidates = [
                    getattr(r, metric)
                    for r in records
                    if r.example_id == ex and r.solver == solver
                ]
                candidates = [float(v) for v in candidates if v is not None]
                if not candidates:
                    continue
                xs.append(xpos)
                ys.append(candidates[0])
                exs.append(ex)

            if not xs:
                continue

            plotted_right.extend(ys)
            ax_right.plot(
                xs,
                ys,
                color=SOLVER_COLOR.get(solver, None),
                linewidth=1.6,
                alpha=0.85,
                zorder=2,
                label=SOLVER_LABEL.get(solver, solver),
            )

            for xpos, y, ex in zip(xs, ys, exs):
                ax_right.plot(
                    [xpos],
                    [y],
                    marker=fixed_case_markers.get(ex, "o"),
                    color=SOLVER_COLOR.get(solver, None),
                    markerfacecolor="white",
                    markeredgecolor=SOLVER_COLOR.get(solver, None),
                    markeredgewidth=1.2,
                    markersize=6.8,
                    linestyle="None",
                    zorder=3,
                )

        ax_right.set_xticks(x_fixed)
        ax_right.set_xticklabels(["Case 3", "Case 4", "Case 5"])
        ax_right.set_xlabel("Fixed-count cases")
        ax_right.grid(True, alpha=0.3)
        ax_right.yaxis.tick_right()
        ax_right.yaxis.set_label_position("right")
        ax_right.set_ylabel(ylabel)
        ax_right.legend(fontsize=7.6, ncol=1, columnspacing=0.9, handletextpad=0.5, loc="best")
        _set_reasonable_y_limits(ax_right, plotted_right, metric="pure_success_rate")

        fig.suptitle(ylabel + " vs obstacle count and Cases", y=0.98, fontsize=11)

        out_path = out_dir / filename
        fig.tight_layout(pad=0.35)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=_compact_figsize("sweep_line"))
    plotted_values: List[float] = []

    for ex in SWEEP_CASES:
        obs_values = sorted({r.obstacles for r in records if r.example_id == ex})
        for solver in solvers:
            xs: List[int] = []
            ys: List[float] = []
            for obs in obs_values:
                r = grouped.get((ex, obs, solver))
                if r is None:
                    continue
                value = getattr(r, metric)
                if value is None:
                    continue
                xs.append(obs)
                ys.append(float(value))
            if xs:
                plotted_values.extend(ys)
                style = _case_style(solver, ex)
                ax.plot(
                    xs,
                    ys,
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    markerfacecolor=style["markerfacecolor"],
                    markeredgecolor=style["color"],
                    linewidth=1.8,
                    markersize=6.5,
                    label=f"{CASE_LABEL.get(ex, f'Case {ex}')}-{SOLVER_LABEL.get(solver, solver)}",
                )

    ax.set_xlabel("Obstacle count (n_obs)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs obstacle count and Cases")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    if metric == "solved_k_mean":
        _set_reasonable_y_limits(ax, plotted_values, metric="solved_k")

    out_path = out_dir / filename
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_sweep_speedup_plot(records: List[ResultRecord], out_dir: Path) -> Path:
    grouped = _group(records)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.15),
        gridspec_kw={"width_ratios": [2.25, 1.0], "wspace": 0.05},
    )
    plotted_left: List[float] = []
    plotted_right: List[float] = []

    # Left: original sweep cases (Cases 1 and 2).
    for ex in SWEEP_CASES:
        obs_values = sorted({r.obstacles for r in records if r.example_id == ex})
        for solver in LEARNED_SOLVERS:
            xs: List[int] = []
            ys: List[float] = []
            for obs in obs_values:
                sp = _speedup_vs_stlpy(grouped, ex, obs, solver)
                if sp is None:
                    continue
                xs.append(obs)
                ys.append(float(sp))
            if xs:
                plotted_left.extend(ys)
                style = _case_style(solver, ex)
                ax_left.plot(
                    xs,
                    ys,
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    markerfacecolor=style["markerfacecolor"],
                    markeredgecolor=style["color"],
                    linewidth=1.8,
                    markersize=6.5,
                    label=f"{CASE_LABEL.get(ex, f'Case {ex}')}-{SOLVER_LABEL.get(solver, solver)}",
                )

    ax_left.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
        color="gray",
        label="STLPY boundary (×1)",
    )
    ax_left.set_xlabel("Obstacle count (n_obs)")
    ax_left.set_ylabel("Speedup over STLPY (×)")
    # ax_left.set_title("Cases 1–2")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=7.6, ncol=2, columnspacing=0.9, handletextpad=0.5)
    _set_reasonable_y_limits(ax_left, plotted_left, metric="speedup", include_reference=1.0)
    ymax = ax_left.get_ylim()[1]
    ax_left.set_ylim(-0.05 * ymax, ymax)

    # Right: fixed-count cases (Cases 3, 4, and 5) with an independent y-axis.
    x_fixed = [0, 1, 2]
    fixed_case_markers = {3: "o", 4: "s", 5: "^"}
    fixed_case_linestyles = {3: "-", 4: "--", 5: ":"}

    for solver in LEARNED_SOLVERS:
        ys: List[float] = []
        xs: List[int] = []
        for xpos, ex in zip(x_fixed, FIXED_COUNT_CASES):
            obs_values = sorted({r.obstacles for r in records if r.example_id == ex})
            if not obs_values:
                continue
            sp = _speedup_vs_stlpy(grouped, ex, obs_values[0], solver)
            if sp is None:
                continue
            xs.append(xpos)
            ys.append(float(sp))

        if not xs:
            continue

        plotted_right.extend(ys)
        ax_right.plot(
            xs,
            ys,
            color=SOLVER_COLOR.get(solver, None),
            linewidth=1.6,
            alpha=0.85,
            zorder=2,
            label=SOLVER_LABEL.get(solver, solver),
        )

        for xpos, y, ex in zip(xs, ys, [FIXED_COUNT_CASES[i] for i in xs]):
            ax_right.plot(
                [xpos],
                [y],
                marker=fixed_case_markers.get(ex, "o"),
                color=SOLVER_COLOR.get(solver, None),
                markerfacecolor="white",
                markeredgecolor=SOLVER_COLOR.get(solver, None),
                markeredgewidth=1.2,
                markersize=6.8,
                linestyle="None",
                zorder=3,
            )

    ax_right.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
        color="gray",
        label="STLPY boundary (×1)",
    )
    ax_right.set_xticks(x_fixed)
    ax_right.set_xticklabels(["Case 3", "Case 4", "Case 5"])
    ax_right.set_xlabel("Fixed-count cases")
    # ax_right.set_title("Cases 3–5")
    ax_right.grid(True, alpha=0.3)
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_right.set_ylabel("Speedup over STLPY (×)")
    ax_right.legend(fontsize=7.6, ncol=1, columnspacing=0.9, handletextpad=0.5, loc="best")
    _set_reasonable_y_limits(ax_right, plotted_right, metric="speedup", include_reference=1.0)
    ymax = ax_right.get_ylim()[1]
    ax_right.set_ylim(-0.05 * ymax, ymax)

    fig.suptitle("End-to-end speedup over STLPY vs obstacle count and Cases", y=0.98, fontsize=11)

    out_path = out_dir / "speedup_vs_obstacle_count.pdf"
    fig.tight_layout(pad=0.35)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_speedup_vs_binary_count_plot(records: List[ResultRecord], out_dir: Path) -> Path:
    grouped = _group(records)
    fig, ax = plt.subplots(figsize=_compact_figsize("binary_scatter"))
    plotted_values: List[float] = []

    for solver in LEARNED_SOLVERS:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        solver_records = sorted(
            [r for r in records if r.solver == solver],
            key=lambda r: (r.example_id, r.obstacles),
        )

        for r in solver_records:
            b = _binary_count(r.example_id, r.obstacles)
            if b is None:
                continue

            sp = _speedup_vs_stlpy(grouped, r.example_id, r.obstacles, solver)
            if sp is None:
                continue

            xs.append(float(b))
            ys.append(float(sp))
            labels.append(_case_annotation(r.example_id, r.obstacles))

        if not xs:
            continue

        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        labels = [labels[i] for i in order]

        plotted_values.extend(ys)
        ax.plot(
            xs,
            ys,
            color=SOLVER_COLOR.get(solver, None),
            linewidth=1.4,
            alpha=0.28,
            zorder=1,
        )
        ax.scatter(
            xs,
            ys,
            s=56,
            marker=SOLVER_MARKER.get(solver, "o"),
            color=SOLVER_COLOR.get(solver, None),
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            label=SOLVER_LABEL.get(solver, solver),
            zorder=2,
        )

        for x, y, txt in zip(xs, ys, labels):
            ax.annotate(
                txt,
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=SOLVER_COLOR.get(solver, "black"),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Number of binary variables")
    ax.set_ylabel("Speedup over STLPY (×)")
    ax.set_title("End-to-end speedup over STLPY vs binary-variable count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if plotted_values:
        ymax = max(plotted_values)
        ymin = min(plotted_values)
        span = max(ymax - ymin, 1.0)
        ax.set_ylim(min(-0.04 * ymax, -0.03 * span), ymax + 0.08 * span)

    out_path = out_dir / "speedup_vs_binary_count.pdf"
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_cost_bars(records: List[ResultRecord], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=_compact_figsize("cost_bars"))
    cases = sorted({r.example_id for r in records})
    width = 0.18
    x_positions = list(range(len(cases)))
    plotted_values: List[float] = []

    for s_idx, solver in enumerate(SOLVER_ORDER):
        means: List[float] = []
        for ex in cases:
            vals = [r.cost_mean for r in records if r.example_id == ex and r.solver == solver and r.cost_mean is not None]
            means.append(sum(vals) / len(vals) if vals else 0.0)
        plotted_values.extend(means)
        offsets = [x + (s_idx - 1.5) * width for x in x_positions]
        ax.bar(
            offsets,
            means,
            width=width,
            label=SOLVER_LABEL.get(solver, solver),
            color=SOLVER_COLOR.get(solver, None),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([CASE_LABEL.get(ex, f"Case {ex}") for ex in cases])
    ax.set_ylabel("Average cost")
    ax.set_title("Cost comparison across solvers")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    _set_reasonable_y_limits(ax, plotted_values, metric="cost")

    out_path = out_dir / "cost_comparison_across_solvers.pdf"
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_fixed_case_bars(
    records: List[ResultRecord],
    out_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
    solvers: List[str],
) -> Path:
    fig, ax = plt.subplots(figsize=_compact_figsize("fixed_bars"))
    width = 0.18
    x_positions = list(range(len(FIXED_COUNT_CASES)))
    plotted_values: List[float] = []

    for s_idx, solver in enumerate(solvers):
        vals: List[float] = []
        for ex in FIXED_COUNT_CASES:
            candidates = [getattr(r, metric) for r in records if r.example_id == ex and r.solver == solver]
            candidates = [float(v) for v in candidates if v is not None]
            vals.append(candidates[0] if candidates else 0.0)
        plotted_values.extend(vals)
        offsets = [x + (s_idx - (len(solvers) - 1) / 2.0) * width for x in x_positions]
        ax.bar(
            offsets,
            vals,
            width=width,
            label=SOLVER_LABEL.get(solver, solver),
            color=SOLVER_COLOR.get(solver, None),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Case 3 (n_obs=4)", "Case 4 (n_obs=3)", "Case 5 (n_obs=5)"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} for fixed-count cases")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    if metric == "pure_success_rate":
        _set_reasonable_y_limits(ax, plotted_values, metric="pure_success_rate")
    elif metric == "solved_k_mean":
        _set_reasonable_y_limits(ax, plotted_values, metric="solved_k")

    out_path = out_dir / filename
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_fixed_case_speedup_bars(records: List[ResultRecord], out_dir: Path) -> Path:
    grouped = _group(records)
    fig, ax = plt.subplots(figsize=_compact_figsize("fixed_speedup"))
    width = 0.22
    x_positions = list(range(len(FIXED_COUNT_CASES)))
    plotted_values: List[float] = []

    for s_idx, solver in enumerate(LEARNED_SOLVERS):
        vals: List[float] = []
        for ex in FIXED_COUNT_CASES:
            obs_values = sorted({r.obstacles for r in records if r.example_id == ex})
            if not obs_values:
                vals.append(0.0)
                continue
            obs = obs_values[0]
            sp = _speedup_vs_stlpy(grouped, ex, obs, solver)
            vals.append(float(sp) if sp is not None else 0.0)

        plotted_values.extend(vals)
        offsets = [x + (s_idx - 1.0) * width for x in x_positions]
        ax.bar(
            offsets,
            vals,
            width=width,
            label=SOLVER_LABEL.get(solver, solver),
            color=SOLVER_COLOR.get(solver, None),
        )

    ax.axhline(1.0, linewidth=1.0, linestyle="--", color="gray")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Case 3 (n_obs=4)", "Case 4 (n_obs=3)", "Case 5 (n_obs=5)"])
    ax.set_ylabel("Speedup over STLPY (×)")
    ax.set_title("End-to-end speedup over STLPY for fixed-count cases")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    _set_reasonable_y_limits(ax, plotted_values, metric="speedup", include_reference=1.0)

    out_path = out_dir / "fixed_case_speedup_over_stlpy.pdf"
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_plots(records: List[ResultRecord], out_dir: str | Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    paths.append(
        _save_sweep_line_plot(
            records,
            out_dir,
            metric="pure_success_rate",
            ylabel="Success rate",
            filename="pure_success_rate_vs_obstacle_count.pdf",
            solvers=LEARNED_SOLVERS,
        )
    )

    paths.append(_save_sweep_speedup_plot(records, out_dir))

    paths.append(_save_speedup_vs_binary_count_plot(records, out_dir))

    paths.append(
        _save_sweep_line_plot(
            records,
            out_dir,
            metric="solved_k_mean",
            ylabel="Solved top-k index",
            filename="solved_k_vs_obstacle_count.pdf",
            solvers=LEARNED_SOLVERS,
        )
    )

    paths.append(_save_cost_bars(records, out_dir))

    paths.append(
        _save_fixed_case_bars(
            records,
            out_dir,
            metric="pure_success_rate",
            ylabel="Pure success rate",
            filename="fixed_case_pure_success_rates.pdf",
            solvers=LEARNED_SOLVERS,
        )
    )

    paths.append(_save_fixed_case_speedup_bars(records, out_dir))

    paths.append(
        _save_fixed_case_bars(
            records,
            out_dir,
            metric="solved_k_mean",
            ylabel="Solved top-k index",
            filename="fixed_case_solved_k.pdf",
            solvers=LEARNED_SOLVERS,
        )
    )

    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Generate benchmark plots from summary.json files.")
    p.add_argument("--root_dir", type=str, default="results")
    args = p.parse_args()

    records = discover_result_records(args.root_dir)
    out_dir = Path(args.root_dir) / "benchmark" / "plots"
    paths = generate_plots(records, out_dir)
    for path in paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()