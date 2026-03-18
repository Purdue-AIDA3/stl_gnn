from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gurobipy import GRB

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EX3_GOAL_PALETTE = ["tab:orange", "tab:purple", "tab:green", "tab:brown"]

EX4_TYPE_PALETTE = {
    "type0": "tab:orange",
    "type1": "tab:green",
    "type2": "tab:purple",
    "type3": "tab:brown",
}

EX5_PAIR_PALETTE = ["tab:orange", "tab:green", "tab:purple", "tab:brown"]


def _pick_from_palette(idx: int, palette):
    return palette[idx % len(palette)]


def _color_for_type_name(typ: str):
    return EX4_TYPE_PALETTE.get(str(typ), "tab:gray")


def visualize_qp_result(
    *,
    out_path: str,
    X: Optional[np.ndarray],
    x0_env: Optional[np.ndarray] = None,
    obstacles: List[Tuple[float, float, float, float]],
    G_rect: Optional[Tuple[float, float, float, float]] = None,
    extra_rects: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None,
    faces_by_pred: Optional[Dict[str, np.ndarray]] = None,
    sat: bool,
    avoid_ok: bool,
    reach_ok: bool,
    extra_ok: Optional[bool] = None,
    filled_rects: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None,
    edgecolor_by_name: Optional[Dict[str, Any]] = None,
    highlight_names: Optional[set] = None,
    fillcolor_by_name: Optional[Dict[str, Any]] = None,
    linestyle_by_name: Optional[Dict[str, str]] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    for j, O_rect in enumerate(obstacles):
        ox1, ox2, oy1, oy2 = O_rect
        ax.add_patch(
            plt.Rectangle(
                (ox1, oy1),
                ox2 - ox1,
                oy2 - oy1,
                fill=True,
                alpha=0.25,
                color="red",
                label="Obstacle" if j == 0 else None,
            )
        )

    if G_rect is not None:
        gx1, gx2, gy1, gy2 = G_rect
        ax.add_patch(
            plt.Rectangle(
                (gx1, gy1),
                gx2 - gx1,
                gy2 - gy1,
                fill=True,
                alpha=0.25,
                color="green",
                label="Goal",
            )
        )

    if filled_rects is not None:
        for rect, name in filled_rects:
            x1, x2, y1, y2 = rect
            fc = None
            if fillcolor_by_name is not None:
                fc = fillcolor_by_name.get(str(name), None)
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=True,
                    alpha=0.25,
                    color=("green" if fc is None else fc),
                    label=f"Selected {name}",
                )
            )

    if extra_rects is not None:
        for rect, name in extra_rects:
            x1, x2, y1, y2 = rect
            nm = str(name)

            is_selected = False
            if highlight_names is not None:
                is_selected = nm in highlight_names

            ec = None
            if edgecolor_by_name is not None:
                ec = edgecolor_by_name.get(nm, None)

            lw = 3 if is_selected else 2
            ls = "-" if is_selected else "--"

            if linestyle_by_name is not None and nm in linestyle_by_name:
                ls = str(linestyle_by_name[nm])

            legend_label = None if nm == "G" else nm

            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=lw,
                    linestyle=ls,
                    edgecolor=ec,
                    label=legend_label,
                )
            )

    if X is not None and X.ndim == 2 and X.shape[0] >= 1:
        ax.plot(X[:, 0], X[:, 1], "-o", label="Trajectory")

    sx = sy = None
    if x0_env is not None:
        sx, sy = float(x0_env[0]), float(x0_env[1])
    elif X is not None and X.ndim == 2 and X.shape[0] >= 1:
        sx, sy = float(X[0, 0]), float(X[0, 1])
    if sx is not None and sy is not None:
        ax.scatter(sx, sy, s=100, marker="*", label="Start")

    if X is not None and faces_by_pred is not None and "O0" in faces_by_pred:
        faces = faces_by_pred["O0"]
        for t in range(min(len(faces), X.shape[0])):
            ax.scatter(X[t, 0], X[t, 1], s=18)

    ax.set_aspect("equal")
    ax.legend()

    extra = "" if extra_ok is None else f" extra_ok={bool(extra_ok)}"
    ax.set_title(f"SAT={bool(sat)} avoid={bool(avoid_ok)} reach={bool(reach_ok)}{extra}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_case_png(
    *,
    save_png: int,
    png_dir: str,
    i: int,
    X: Optional[np.ndarray],
    x0_env: Optional[np.ndarray],
    obstacles: List[Tuple[float, float, float, float]],
    G_rect: Optional[Tuple[float, float, float, float]],
    sat: bool,
    avoid_ok: bool,
    reach_ok: bool,
    status_qp: int,
    extra_ok: Optional[bool] = None,
    extra_rects: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None,
    faces_by_pred: Optional[Dict[str, np.ndarray]] = None,
    filled_rects: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None,
    edgecolor_by_name: Optional[Dict[str, Any]] = None,
    highlight_names: Optional[set] = None,
    fillcolor_by_name: Optional[Dict[str, Any]] = None,
    linestyle_by_name: Optional[Dict[str, str]] = None,
) -> None:
    if int(save_png) != 1:
        return

    os.makedirs(str(png_dir) or ".", exist_ok=True)
    out_png = os.path.join(str(png_dir), f"case_{int(i):04d}.png")

    is_opt = int(status_qp) == int(GRB.OPTIMAL) and X is not None
    if is_opt:
        X_vis = X
        sat_vis = bool(sat)
        avoid_vis = bool(avoid_ok)
        reach_vis = bool(reach_ok)
        extra_vis = (None if extra_ok is None else bool(extra_ok))
    else:
        X_vis = None
        sat_vis = False
        avoid_vis = False
        reach_vis = False
        extra_vis = None
        faces_by_pred = None

    visualize_qp_result(
        out_path=out_png,
        X=X_vis,
        x0_env=x0_env,
        obstacles=obstacles,
        G_rect=G_rect,
        extra_rects=extra_rects,
        faces_by_pred=faces_by_pred,
        sat=sat_vis,
        avoid_ok=avoid_vis,
        reach_ok=reach_vis,
        extra_ok=extra_vis,
        filled_rects=filled_rects,
        edgecolor_by_name=edgecolor_by_name,
        highlight_names=highlight_names,
        fillcolor_by_name=fillcolor_by_name,
        linestyle_by_name=linestyle_by_name,
    )