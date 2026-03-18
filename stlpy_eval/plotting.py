from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


Rect = Tuple[float, float, float, float]


def _add_rect(ax, rect: Rect, *, edgecolor="k", facecolor="none", alpha=1.0, linestyle="-", linewidth=1.5):
    x1, x2, y1, y2 = [float(v) for v in rect]
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    )


def save_case_png(
    *,
    sample: Dict[str, Any],
    X: np.ndarray | None,
    example_id: int,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = sample["env"]
    obstacles = env.get("obstacles", []) or []
    regions = env.get("regions", {}) or {}
    x0 = np.asarray(env["x0"], dtype=float).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6))

    for obs in obstacles:
        _add_rect(ax, tuple(obs), edgecolor="k", facecolor="k", alpha=0.35)

    if int(example_id) == 1:
        if "G" in regions:
            _add_rect(ax, tuple(regions["G"]), edgecolor="green", facecolor="green", alpha=0.35)

    elif int(example_id) == 2:
        if "G" in regions:
            _add_rect(ax, tuple(regions["G"]), edgecolor="green", facecolor="green", alpha=0.35)
        if "T1" in regions:
            _add_rect(ax, tuple(regions["T1"]), edgecolor="blue", facecolor="blue", alpha=0.25)
        if "T2" in regions:
            _add_rect(ax, tuple(regions["T2"]), edgecolor="blue", facecolor="blue", alpha=0.25)

    elif int(example_id) == 3:
        for k in sorted(regions.keys()):
            if k.startswith("G"):
                _add_rect(ax, tuple(regions[k]), edgecolor="green", facecolor="green", alpha=0.25)

    elif int(example_id) == 4:
        colors = plt.cm.tab10.colors
        type_to_color: Dict[str, Any] = {}
        cidx = 0
        for k in sorted(regions.keys()):
            if not k.startswith("T_"):
                continue
            parts = k.split("_", 2)
            if len(parts) != 3:
                continue
            typ = str(parts[1])
            if typ not in type_to_color:
                type_to_color[typ] = colors[cidx % len(colors)]
                cidx += 1
            _add_rect(
                ax,
                tuple(regions[k]),
                edgecolor=type_to_color[typ],
                facecolor=type_to_color[typ],
                alpha=0.25,
            )

    elif int(example_id) == 5:
        if "G" in regions:
            _add_rect(ax, tuple(regions["G"]), edgecolor="green", facecolor="green", alpha=0.35)
        for k in sorted(regions.keys()):
            if k.startswith("D"):
                _add_rect(ax, tuple(regions[k]), edgecolor="red", facecolor="red", alpha=0.25)
            elif k.startswith("K"):
                _add_rect(ax, tuple(regions[k]), edgecolor="blue", facecolor="blue", alpha=0.25)

    ax.plot(float(x0[0]), float(x0[1]), "go", markersize=7)

    if X is not None:
        ax.plot(X[0, :], X[1, :], "-o", markersize=2)

    ax.set_aspect("equal", "box")
    ax.set_title(f"example_id={int(example_id)}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)