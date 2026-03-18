"""
Geometry primitives and Gurobi constraint templates.

This module currently supports axis-aligned rectangles and provides:
- Membership constraints: (px, py) inside a rectangle
- Non-membership constraints: outside a rectangle via a 4-face disjunction using Big-M
"""

from __future__ import annotations

from typing import List, Tuple, Union

import gurobipy as gp


Rect = Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)


def rect_in_constraints(model: gp.Model, px, py, rect: Rect, prefix: str) -> None:
    """
    Enforce (px, py) to lie inside the rectangle.
    Adds four linear constraints:
      xmin <= px <= xmax
      ymin <= py <= ymax
    """
    xmin, xmax, ymin, ymax = rect
    model.addConstr(px >= float(xmin), name=f"{prefix}_xmin")
    model.addConstr(px <= float(xmax), name=f"{prefix}_xmax")
    model.addConstr(py >= float(ymin), name=f"{prefix}_ymin")
    model.addConstr(py <= float(ymax), name=f"{prefix}_ymax")


def rect_outside_bigM(
    model: gp.Model,
    px,
    py,
    rect: Rect,
    face_bin: List[gp.Var],
    prefix: str,
    M: float = 1e3,
    rhs: Union[float, gp.LinExpr] = 1.0,
) -> None:
    """
    Enforce (px, py) to lie outside the rectangle via a 4-face disjunction.

    Faces (binary indicators):
      face_bin[0] -> left  : px <= xmin
      face_bin[1] -> right : px >= xmax
      face_bin[2] -> below : py <= ymin
      face_bin[3] -> above : py >= ymax

    Big-M linearization:
      px <= xmin + M*(1 - b0)
      px >= xmax - M*(1 - b1)
      py <= ymin + M*(1 - b2)
      py >= ymax - M*(1 - b3)

    Disjunction activation:
      sum(face_bin) >= rhs

    Notes:
    - Use rhs=1.0 for unconditional outside constraints (e.g., Always(OutsideRect)).
    - Use rhs=pre_t (0/1 expression) to conditionally activate outside constraints
      (e.g., Until pre-trigger safety).
    """
    xmin, xmax, ymin, ymax = rect
    if len(face_bin) != 4:
        raise ValueError(f"face_bin must have length 4, got {len(face_bin)}")

    # Face constraints (deactivated when corresponding binary is 0)
    model.addConstr(px <= float(xmin) + float(M) * (1 - face_bin[0]), name=f"{prefix}_left")
    model.addConstr(px >= float(xmax) - float(M) * (1 - face_bin[1]), name=f"{prefix}_right")
    model.addConstr(py <= float(ymin) + float(M) * (1 - face_bin[2]), name=f"{prefix}_below")
    model.addConstr(py >= float(ymax) - float(M) * (1 - face_bin[3]), name=f"{prefix}_above")

    # Disjunction activation: at least one face must be active to enforce outside
    model.addConstr(gp.quicksum(face_bin) >= rhs, name=f"{prefix}_outside_or")


__all__ = [
    "Rect",
    "rect_in_constraints",
    "rect_outside_bigM",
]