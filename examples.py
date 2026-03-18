"""
Example factories and parameter presets.

Normalized env design:
- obstacles: always Sequence[Rect]
- regions: stored separately in env by scripts/descriptor layer

This file provides:
- STL AST template constructors for Examples 1–5
- Geometry constraints + validation
- Perturbation sampler that perturbs obstacles AND regions coherently
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from geometry import Rect
from stl import STLNode, And, Or, Always, Eventually, InRect, OutsideRect, Until


# ==============================================================================
# 1) Spec factories (programmatic AST) — obstacles always Sequence[Rect]
# ==============================================================================

def make_example_1(T: int, obstacles: Sequence[Rect], G: Rect) -> STLNode:
    """
    Example 1 — Reach-Avoid:
        Box_[0,T] (∧_j ¬O_j)  ∧  Diamond_[0,T] G
    """
    avoid = Always(0, T, And(tuple(OutsideRect(f"O{j}", r) for j, r in enumerate(obstacles))))
    return And((
        avoid,
        Eventually(0, T, InRect("G", G)),
    ))


def make_example_2(T: int, tau: int, obstacles: Sequence[Rect], G: Rect, T1: Rect, T2: Rect) -> STLNode:
    """
    Example 2 — Either-Or (Dwell Choice), pinned to exactly TWO targets:
        Diamond_[0, T-tau] ( Box_[0,tau] T1  OR  Box_[0,tau] T2 )
        AND Diamond_[0,T] G
        AND Box_[0,T] (∧_j ¬O_j)
    """
    tau = int(tau)
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if T - tau < 0:
        raise ValueError(f"Need T - tau >= 0, got T={T}, tau={tau}")

    avoid = Always(0, T, And(tuple(OutsideRect(f"O{j}", r) for j, r in enumerate(obstacles))))
    either_or = Or((
        Always(0, tau, InRect("T1", T1)),
        Always(0, tau, InRect("T2", T2)),
    ))
    return And((
        Eventually(0, T - tau, either_or),
        Eventually(0, T, InRect("G", G)),
        avoid,
    ))


def make_example_3(T: int, obstacles: Sequence[Rect], goals: Sequence[Rect]) -> STLNode:
    """
    Example 3 — Narrow Passage:
        Box_[0,T] (∧_j ¬O_j)  ∧  Diamond_[0,T] (∨_m G_m)
    """
    if len(goals) < 1:
        raise ValueError("Example 3 requires at least one goal.")
    avoid = Always(0, T, And(tuple(OutsideRect(f"O{j}", r) for j, r in enumerate(obstacles))))
    reach_one = Eventually(0, T, Or(tuple(InRect(f"G{m}", r) for m, r in enumerate(goals))))
    return And((avoid, reach_one))


def make_example_4(T: int, obstacles: Sequence[Rect], targets_by_type: Dict[str, Sequence[Rect]]) -> STLNode:
    """
    Example 4 — Multi-Target:
        Box_[0,T] (∧_j ¬O_j)  ∧  ∧_{c in types} Diamond_[0,T] (∨_{k in targets_c} T_{c,k})
    """
    if len(targets_by_type) < 1:
        raise ValueError("Example 4 requires at least one type.")
    avoid = Always(0, T, And(tuple(OutsideRect(f"O{j}", r) for j, r in enumerate(obstacles))))

    per_type: List[STLNode] = []
    for c, rects in targets_by_type.items():
        rs = list(rects)
        if len(rs) < 1:
            raise ValueError(f"Example 4 requires at least one target for type '{c}'.")
        per_type.append(Eventually(0, T, Or(tuple(InRect(f"T_{c}_{k}", r) for k, r in enumerate(rs)))))

    return And((avoid, And(tuple(per_type))))


def make_example_5(T: int, obstacles: Sequence[Rect], doors: Sequence[Rect], keys: Sequence[Rect], G: Rect) -> STLNode:
    r"""
    Example 5 — Door Puzzle (Until-Based):
        ∧_{i=1..N} ( ¬D_i U_[0,T] K_i )  ∧  F_[0,T] G  ∧  G_[0,T] (∧_{j=1..M} ¬O_j)

    Project constraint:
      Until left: OutsideRect(D_i)
      Until right: InRect(K_i)
    """
    ds = list(doors)
    ks = list(keys)
    if len(ds) < 1 or len(ks) < 1:
        raise ValueError("Example 5 requires at least one door and one key.")
    if len(ds) != len(ks):
        raise ValueError(f"Example 5 requires len(doors)==len(keys), got {len(ds)} vs {len(ks)}")

    untils: List[STLNode] = []
    for i, (Di, Ki) in enumerate(zip(ds, ks)):
        untils.append(Until(0, T, OutsideRect(f"D{i}", Di), InRect(f"K{i}", Ki)))

    avoid = Always(0, T, And(tuple(OutsideRect(f"O{j}", r) for j, r in enumerate(obstacles))))
    reach_goal = Eventually(0, T, InRect("G", G))
    return And((And(tuple(untils)), reach_goal, avoid))


# ==============================================================================
# 2) Geometry / validation / perturbation
# ==============================================================================

def shift_rect(r: Rect, dx: float, dy: float) -> Rect:
    x1, x2, y1, y2 = r
    return (float(x1) + float(dx), float(x2) + float(dx), float(y1) + float(dy), float(y2) + float(dy))


def shift_x0(x0: np.ndarray, dx: float, dy: float) -> np.ndarray:
    x0p = np.array(x0, dtype=float).copy()
    x0p[0] += float(dx)
    x0p[1] += float(dy)
    return x0p


@dataclass(frozen=True)
class GeometryConstraints:
    min_goal_obstacle_clearance: float = 0.0
    min_start_obstacle_clearance: float = 0.0
    min_start_goal_clearance: float = 0.0
    min_obstacle_obstacle_clearance: float = 0.0
    require_rects_in_bounds: bool = True
    require_start_in_bounds: bool = True


def rect_expand(r: Rect, margin: float) -> Rect:
    xmin, xmax, ymin, ymax = r
    m = float(margin)
    return (float(xmin) - m, float(xmax) + m, float(ymin) - m, float(ymax) + m)


def rect_intersects(a: Rect, b: Rect) -> bool:
    axmin, axmax, aymin, aymax = a
    bxmin, bxmax, bymin, bymax = b
    return not (axmax < bxmin or axmin > bxmax or aymax < bymin or aymin > bymax)


def rect_contains_point(r: Rect, x: float, y: float, margin: float = 0.0) -> bool:
    xmin, xmax, ymin, ymax = rect_expand(r, margin)
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def rect_within_bounds(r: Rect, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> bool:
    xmin, xmax, ymin, ymax = r
    return (xmin >= xlim[0]) and (xmax <= xlim[1]) and (ymin >= ylim[0]) and (ymax <= ylim[1])


def start_within_bounds(x0: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> bool:
    px, py = float(x0[0]), float(x0[1])
    return (xlim[0] <= px <= xlim[1]) and (ylim[0] <= py <= ylim[1])


def validate_geometry(
    *,
    x0: np.ndarray,
    obstacles: Sequence[Rect],
    regions: Mapping[str, Rect],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    rules: GeometryConstraints,
) -> Tuple[bool, str]:
    """
    Validation rules:
      - start in bounds (optional)
      - obstacles + regions in bounds (optional)
      - obstacles must not overlap each other
      - each region must not overlap any obstacle
      - start must not be inside/too close to any obstacle
      - optional start-region clearance for all regions
    """
    obs = list(obstacles)
    regs = dict(regions)

    if rules.require_start_in_bounds and not start_within_bounds(x0, xlim, ylim):
        return False, "start_out_of_bounds"

    if rules.require_rects_in_bounds:
        for j, Oj in enumerate(obs):
            if not rect_within_bounds(Oj, xlim, ylim):
                return False, f"obstacle_{j}_out_of_bounds"
        for name, R in regs.items():
            if not rect_within_bounds(R, xlim, ylim):
                return False, f"region_{name}_out_of_bounds"

    m_obs = float(rules.min_obstacle_obstacle_clearance)
    for i in range(len(obs)):
        Oi = rect_expand(obs[i], m_obs)
        for j in range(i + 1, len(obs)):
            Oj = rect_expand(obs[j], m_obs)
            if rect_intersects(Oi, Oj):
                return False, f"obstacle_{i}_overlaps_obstacle_{j}"

    m_goal = float(rules.min_goal_obstacle_clearance)
    for j, Oj in enumerate(obs):
        Oe = rect_expand(Oj, m_goal)
        for name, R in regs.items():
            if rect_intersects(Oe, R):
                return False, f"region_{name}_overlaps_obstacle_{j}"

    m_start = float(rules.min_start_obstacle_clearance)
    for j, Oj in enumerate(obs):
        if rect_contains_point(Oj, float(x0[0]), float(x0[1]), margin=m_start):
            return False, f"start_inside_or_too_close_to_obstacle_{j}"

    if rules.min_start_goal_clearance > 0.0:
        m = float(rules.min_start_goal_clearance)
        for name, R in regs.items():
            if rect_contains_point(R, float(x0[0]), float(x0[1]), margin=m):
                return False, f"start_inside_or_too_close_to_region_{name}"

    return True, "ok"


def sample_valid_perturbation(
    *,
    x0_base: np.ndarray,
    obstacles_base: Sequence[Rect],
    regions_base: Mapping[str, Rect],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    dx0_range: Tuple[float, float],
    dy0_range: Tuple[float, float],
    dO_range: Tuple[float, float, float, float],
    dR_range: Tuple[float, float, float, float],
    rules: GeometryConstraints,
    rng: np.random.Generator,
    max_tries: int = 200,
    fixed_obstacle_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """
    Rejection-sample perturbations until geometry is valid.

    - Perturbs all obstacles unless fixed_obstacle_mask[i] is True
    - Perturbs all regions coherently in the same environment sample
    - Obstacles must satisfy obstacle-obstacle non-overlap
    - Regions must satisfy region-obstacle clearance, bounds, and start-clearance rules

    Returns dict with keys:
      x0: np.ndarray
      obstacles: List[Rect]
      regions: Dict[str, Rect]
      deltas: Dict[str, Any]
      tries: int
    """
    obs0 = list(obstacles_base)
    if len(obs0) < 1:
        raise ValueError("obstacles_base must contain at least one obstacle.")

    fixed: List[bool]
    if fixed_obstacle_mask is None:
        fixed = [False] * len(obs0)
    else:
        fixed = list(fixed_obstacle_mask)
        if len(fixed) != len(obs0):
            raise ValueError(f"fixed_obstacle_mask length mismatch: got {len(fixed)}, expected {len(obs0)}")

    dx_min, dx_max = float(dO_range[0]), float(dO_range[1])
    dy_min, dy_max = float(dO_range[2]), float(dO_range[3])
    rx_min, rx_max = float(dR_range[0]), float(dR_range[1])
    ry_min, ry_max = float(dR_range[2]), float(dR_range[3])

    region_names = list(regions_base.keys())

    for t in range(1, int(max_tries) + 1):
        dx0 = float(rng.uniform(dx0_range[0], dx0_range[1]))
        dy0 = float(rng.uniform(dy0_range[0], dy0_range[1]))
        x0 = shift_x0(x0_base, dx0, dy0)

        obstacles: List[Rect] = []
        dOx: List[float] = []
        dOy: List[float] = []
        for i, Oi in enumerate(obs0):
            if fixed[i]:
                obstacles.append(Oi)
                dOx.append(0.0)
                dOy.append(0.0)
            else:
                ox = float(rng.uniform(dx_min, dx_max))
                oy = float(rng.uniform(dy_min, dy_max))
                obstacles.append(shift_rect(Oi, ox, oy))
                dOx.append(ox)
                dOy.append(oy)

        regions: Dict[str, Rect] = {}
        dRx: Dict[str, float] = {}
        dRy: Dict[str, float] = {}
        for name in region_names:
            r0 = regions_base[name]
            rx = float(rng.uniform(rx_min, rx_max))
            ry = float(rng.uniform(ry_min, ry_max))
            regions[name] = shift_rect(r0, rx, ry)
            dRx[name] = rx
            dRy[name] = ry

        ok, _ = validate_geometry(
            x0=x0,
            obstacles=obstacles,
            regions=regions,
            xlim=xlim,
            ylim=ylim,
            rules=rules,
        )
        if ok:
            return dict(
                x0=x0,
                obstacles=obstacles,
                regions=regions,
                deltas=dict(dx0=dx0, dy0=dy0, dOx=dOx, dOy=dOy, fixed=fixed, dRx=dRx, dRy=dRy),
                tries=t,
            )

    raise RuntimeError(f"Failed to sample valid perturbation after max_tries={max_tries}.")