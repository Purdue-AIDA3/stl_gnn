from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from stlpy.benchmarks.common import inside_rectangle_formula, outside_rectangle_formula
from stlpy.systems import DoubleIntegrator


Rect = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SolverConfig:
    robustness_cost: bool
    Q: np.ndarray
    R: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray


def _rect_tuple(r: Any) -> Rect:
    rr = tuple(float(v) for v in r)
    if len(rr) != 4:
        raise ValueError(f"Rectangle must have 4 values, got {rr}")
    return rr  # type: ignore[return-value]


def _eventually_inside(rect: Rect, T: int):
    return inside_rectangle_formula(rect, 0, 1, 6).eventually(0, int(T))


def _always_outside(rect: Rect, T: int):
    return outside_rectangle_formula(rect, 0, 1, 6).always(0, int(T))


def _always_inside(rect: Rect, tau: int):
    return inside_rectangle_formula(rect, 0, 1, 6).always(0, int(tau))


def _until_outside_then_key(door_rect: Rect, key_rect: Rect, T: int):
    no_door = outside_rectangle_formula(door_rect, 0, 1, 6)
    at_key = inside_rectangle_formula(key_rect, 0, 1, 6)
    return no_door.until(at_key, 0, int(T))


def _obstacle_avoidance(obstacles: List[Rect], T: int):
    if len(obstacles) < 1:
        raise ValueError("At least one obstacle is required.")
    phi = _always_outside(obstacles[0], T)
    for obs in obstacles[1:]:
        phi = phi & _always_outside(obs, T)
    return phi


def _regions_from_sample(sample: Dict[str, Any]) -> Dict[str, Rect]:
    env = sample["env"]
    regions = env.get("regions", {}) or {}
    out: Dict[str, Rect] = {}
    for k, v in regions.items():
        out[str(k)] = _rect_tuple(v)
    return out


def _obstacles_from_sample(sample: Dict[str, Any]) -> List[Rect]:
    env = sample["env"]
    obstacles = env.get("obstacles", []) or []
    return [_rect_tuple(o) for o in obstacles]


def _x0_from_sample(sample: Dict[str, Any]) -> np.ndarray:
    env = sample["env"]
    x0 = np.asarray(env["x0"], dtype=float).reshape(-1)
    return x0


def build_spec_and_system(
    *,
    sample: Dict[str, Any],
    example_id: int,
):
    T = int(sample["T"])
    env = sample["env"]
    meta = env.get("meta", {}) or {}

    obstacles = _obstacles_from_sample(sample)
    regions = _regions_from_sample(sample)

    if int(example_id) == 1:
        if "G" not in regions:
            raise ValueError("Example 1 requires region 'G'.")
        phi = _obstacle_avoidance(obstacles, T) & _eventually_inside(regions["G"], T)

    elif int(example_id) == 2:
        if "G" not in regions or "T1" not in regions or "T2" not in regions:
            raise ValueError("Example 2 requires regions 'G', 'T1', and 'T2'.")
        tau = int(meta.get("tau", env.get("tau", None)))
        if tau is None:
            raise ValueError("Example 2 requires dwell time 'tau' in env['meta'] or env.")

        at_t1 = _always_inside(regions["T1"], tau)
        at_t2 = _always_inside(regions["T2"], tau)
        at_mid = (at_t1 | at_t2).eventually(0, T - tau)
        phi = _obstacle_avoidance(obstacles, T) & at_mid & _eventually_inside(regions["G"], T)

    elif int(example_id) == 3:
        goals = [regions[k] for k in sorted(regions.keys()) if k.startswith("G")]
        if len(goals) < 1:
            raise ValueError("Example 3 requires at least one goal region 'G*'.")

        any_goal = _eventually_inside(goals[0], T)
        for g in goals[1:]:
            any_goal = any_goal | _eventually_inside(g, T)

        phi = _obstacle_avoidance(obstacles, T) & any_goal

    elif int(example_id) == 4:
        target_groups: Dict[str, List[Rect]] = {}
        for rk, rect in regions.items():
            if not rk.startswith("T_"):
                continue
            parts = rk.split("_", 2)
            if len(parts) != 3:
                continue
            typ = str(parts[1])
            target_groups.setdefault(typ, []).append(rect)

        if len(target_groups) < 1:
            raise ValueError("Example 4 requires target regions named like 'T_type_idx'.")

        phi = _obstacle_avoidance(obstacles, T)
        for typ in sorted(target_groups.keys()):
            rects = target_groups[typ]
            group_phi = _eventually_inside(rects[0], T)
            for r in rects[1:]:
                group_phi = group_phi | _eventually_inside(r, T)
            phi = phi & group_phi

    elif int(example_id) == 5:
        if "G" not in regions:
            raise ValueError("Example 5 requires region 'G'.")

        door_keys = sorted([k for k in regions.keys() if k.startswith("D")])
        key_keys = sorted([k for k in regions.keys() if k.startswith("K")])
        n_pairs = min(len(door_keys), len(key_keys))
        if n_pairs < 1:
            raise ValueError("Example 5 requires at least one door/key pair.")

        phi = _obstacle_avoidance(obstacles, T)
        for i in range(n_pairs):
            di = door_keys[i]
            ki = key_keys[i]
            phi = phi & _until_outside_then_key(regions[di], regions[ki], T)

        phi = phi & _eventually_inside(regions["G"], T)

    else:
        raise ValueError(f"Unsupported example_id={example_id}")

    sys = DoubleIntegrator(2)
    x0 = _x0_from_sample(sample)
    return phi, sys, x0


def get_solver_config(example_id: int) -> SolverConfig:
    """
    Keep stlpy_eval consistent with scripts.build_problem_fn(sample).

    scripts.py uses the same cost matrices for all examples:
        Q = 1e-1 * diag([0, 0, 1, 1])
        R = 1e-1 * I

    We keep bounds example-dependent, but make Q/R identical across examples.
    """
    common_Q = 1e-1 * np.diag([0.0, 0.0, 1.0, 1.0])
    common_R = 1e-1 * np.eye(2)

    if int(example_id) == 1:
        return SolverConfig(
            robustness_cost=False,
            Q=common_Q,
            R=common_R,
            u_min=np.array([-0.5, -0.5], dtype=float),
            u_max=np.array([0.5, 0.5], dtype=float),
            x_min=np.array([0.0, 0.0, -1.0, -1.0], dtype=float),
            x_max=np.array([10.0, 10.0, 1.0, 1.0], dtype=float),
        )

    if int(example_id) == 2:
        return SolverConfig(
            robustness_cost=False,
            Q=common_Q,
            R=common_R,
            u_min=np.array([-0.5, -0.5], dtype=float),
            u_max=np.array([0.5, 0.5], dtype=float),
            x_min=np.array([0.0, 0.0, -1.0, -1.0], dtype=float),
            x_max=np.array([10.0, 10.0, 1.0, 1.0], dtype=float),
        )

    if int(example_id) == 3:
        return SolverConfig(
            robustness_cost=False,
            Q=common_Q,
            R=common_R,
            u_min=np.array([-0.5, -0.5], dtype=float),
            u_max=np.array([0.5, 0.5], dtype=float),
            x_min=np.array([0.0, 0.0, -1.0, -1.0], dtype=float),
            x_max=np.array([12.0, 12.0, 1.0, 1.0], dtype=float),
        )

    if int(example_id) == 4:
        return SolverConfig(
            robustness_cost=False,
            Q=common_Q,
            R=common_R,
            u_min=np.array([-0.5, -0.5], dtype=float),
            u_max=np.array([0.5, 0.5], dtype=float),
            x_min=np.array([0.0, 0.0, -1.0, -1.0], dtype=float),
            x_max=np.array([10.0, 10.0, 1.0, 1.0], dtype=float),
        )

    if int(example_id) == 5:
        return SolverConfig(
            robustness_cost=False,
            Q=common_Q,
            R=common_R,
            u_min=np.array([-0.5, -0.5], dtype=float),
            u_max=np.array([0.5, 0.5], dtype=float),
            x_min=np.array([0.0, 0.0, -2.0, -2.0], dtype=float),
            x_max=np.array([15.0, 10.0, 2.0, 2.0], dtype=float),
        )

    raise ValueError(f"Unsupported example_id={example_id}")