from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import json
import os
import time
import numpy as np
import torch

from stl import STLBinaryExtractor, STLNode


# ==============================================================================
# Env parsing
# ==============================================================================

def _get_obstacles(env: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
    obs = env.get("obstacles", [])
    if not isinstance(obs, list) or len(obs) < 1:
        raise ValueError("env['obstacles'] must be a non-empty list.")
    out: List[Tuple[float, float, float, float]] = []
    for r in obs:
        if not isinstance(r, (list, tuple)) or len(r) != 4:
            raise ValueError(f"Bad obstacle rect: {r}")
        out.append((float(r[0]), float(r[1]), float(r[2]), float(r[3])))
    return out


def _get_regions(env: Dict[str, Any]) -> Dict[str, Tuple[float, float, float, float]]:
    regions = env.get("regions", {})
    if not isinstance(regions, dict):
        raise ValueError("env['regions'] missing.")
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for k, v in regions.items():
        if not isinstance(v, (list, tuple)) or len(v) != 4:
            raise ValueError(f"Bad region rect {k}: {v}")
        out[str(k)] = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return out


# ==============================================================================
# Face helpers
# ==============================================================================

def _start_consistent_face(
    *,
    x0: float,
    y0: float,
    rect: Tuple[float, float, float, float],
) -> int:
    x1, x2, y1, y2 = rect
    if x0 <= x1:
        return 0
    if x0 >= x2:
        return 1
    if y0 <= y1:
        return 2
    return 3


def _force_start_consistent_faces_inplace(
    *,
    prob,
    rect: Tuple[float, float, float, float],
    faces: np.ndarray,
) -> None:
    if not isinstance(faces, np.ndarray) or faces.ndim != 1 or faces.shape[0] < 1:
        return
    x0 = float(prob.x0[0])
    y0 = float(prob.x0[1])
    faces[0] = int(_start_consistent_face(x0=x0, y0=y0, rect=rect))


# ==============================================================================
# Strategy decoding
# ==============================================================================

def decode_strategy_from_logits(
    *,
    ast: STLNode,
    T: int,
    bin_logits_by_id: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, int], List[Any]]:
    descs = STLBinaryExtractor(int(T)).extract(ast)

    logit: Dict[str, float] = {}
    missing: List[str] = []
    for d in descs:
        if d.id not in bin_logits_by_id:
            missing.append(d.id)
        else:
            v = bin_logits_by_id[d.id]
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"bin_logits_by_id[{d.id}] is not a Tensor.")
            logit[d.id] = float(v.detach().cpu().item())
    if missing:
        head = missing[:10]
        raise RuntimeError(
            f"Missing {len(missing)} descriptor ids in bin_logits_by_id. First few: {head}."
        )

    event_groups: DefaultDict[str, List[str]] = defaultdict(list)
    outside_groups: DefaultDict[Tuple[str, int], List[str]] = defaultdict(list)
    always_or_groups: DefaultDict[Tuple[str, int], List[str]] = defaultdict(list)

    for d in descs:
        if d.role == "event_time":
            event_groups[str(d.node_tag)].append(d.id)
        elif d.role == "outside_face":
            t = int(d.meta.get("t", d.meta.get("k", 0)))
            pred = str(d.meta.get("pred", ""))
            if pred == "":
                raise RuntimeError(f"outside_face descriptor {d.id} missing meta['pred']. meta={d.meta}")
            outside_groups[(pred, int(t))].append(d.id)
        elif d.role == "always_or_choice":
            k = int(d.meta.get("k", 0))
            always_or_groups[(str(d.node_tag), int(k))].append(d.id)

    strat: Dict[str, int] = {d.id: 0 for d in descs}

    def _pick_one(ids: List[str]) -> str:
        best = ids[0]
        bestv = logit[best]
        for sid in ids[1:]:
            v = logit[sid]
            if v > bestv:
                best = sid
                bestv = v
        return best

    for _, ids in event_groups.items():
        strat[_pick_one(ids)] = 1
    for _, ids in outside_groups.items():
        strat[_pick_one(ids)] = 1
    for _, ids in always_or_groups.items():
        strat[_pick_one(ids)] = 1

    return strat, descs


def _extract_selected_event_times(
    *,
    descs: List[Any],
    strategy: Dict[str, int],
) -> List[Tuple[str, int, Dict[str, Any]]]:
    out: List[Tuple[str, int, Dict[str, Any]]] = []
    for d in descs:
        if d.role != "event_time":
            continue
        if int(strategy.get(d.id, 0)) != 1:
            continue
        pred = str(d.meta.get("pred", ""))
        k = int(d.meta.get("k", 0))
        out.append((pred, k, dict(d.meta)))
    return out


def _extract_selected_faces_by_pred(
    *,
    descs: List[Any],
    strategy: Dict[str, int],
    T: int,
) -> Dict[str, np.ndarray]:
    t_to_face: Dict[str, Dict[int, int]] = defaultdict(dict)

    for d in descs:
        if d.role != "outside_face":
            continue
        if int(strategy.get(d.id, 0)) != 1:
            continue
        pred = str(d.meta.get("pred", ""))
        t = int(d.meta.get("t", d.meta.get("k", 0)))
        h = int(d.meta.get("face", d.meta.get("h", 0)))
        t_to_face[pred][t] = h

    faces_by_pred: Dict[str, np.ndarray] = {}
    for pred, mp in t_to_face.items():
        faces = np.zeros((T + 1,), dtype=int)
        for t in range(T + 1):
            if t not in mp:
                raise RuntimeError(f"Missing outside_face selection for pred={pred} at t={t}")
            faces[t] = int(mp[t])
        faces_by_pred[pred] = faces

    return faces_by_pred


# ==============================================================================
# Checks
# ==============================================================================

def _in_rect(X: np.ndarray, rect: Tuple[float, float, float, float]) -> np.ndarray:
    x1, x2, y1, y2 = rect
    return (X[:, 0] >= x1) & (X[:, 0] <= x2) & (X[:, 1] >= y1) & (X[:, 1] <= y2)


def check_example1(
    *,
    X: np.ndarray,
    obstacles: List[Tuple[float, float, float, float]],
    G_rect: Tuple[float, float, float, float],
) -> Tuple[bool, bool, bool]:
    avoid_ok = True
    for O in obstacles:
        if bool(np.any(_in_rect(X, O))):
            avoid_ok = False
            break
    reach_ok = bool(np.any(_in_rect(X, G_rect)))
    sat = bool(avoid_ok and reach_ok)
    return avoid_ok, reach_ok, sat


def check_example2(
    *,
    X: np.ndarray,
    obstacles: List[Tuple[float, float, float, float]],
    G_rect: Tuple[float, float, float, float],
    T1_rect: Tuple[float, float, float, float],
    T2_rect: Tuple[float, float, float, float],
    tau: int,
) -> Tuple[bool, bool, bool, bool]:
    avoid_ok = True
    for O in obstacles:
        if bool(np.any(_in_rect(X, O))):
            avoid_ok = False
            break
    reach_ok = bool(np.any(_in_rect(X, G_rect)))

    in_T1 = _in_rect(X, T1_rect)
    in_T2 = _in_rect(X, T2_rect)

    TT = X.shape[0] - 1
    mid_ok = False
    for k in range(0, TT - int(tau) + 1):
        w = slice(k, k + int(tau) + 1)
        if bool(np.all(in_T1[w])) or bool(np.all(in_T2[w])):
            mid_ok = True
            break

    sat = bool(avoid_ok and reach_ok and mid_ok)
    return avoid_ok, reach_ok, mid_ok, sat


def check_example3(
    *,
    X: np.ndarray,
    obstacles: List[Tuple[float, float, float, float]],
    goals: List[Tuple[float, float, float, float]],
) -> Tuple[bool, bool, bool]:
    avoid_ok = True
    for O in obstacles:
        if bool(np.any(_in_rect(X, O))):
            avoid_ok = False
            break
    reach_ok = any(bool(np.any(_in_rect(X, G))) for G in goals)
    sat = bool(avoid_ok and reach_ok)
    return avoid_ok, reach_ok, sat


def check_example4(
    *,
    X: np.ndarray,
    obstacles: List[Tuple[float, float, float, float]],
    targets_by_type: Dict[str, List[Tuple[float, float, float, float]]],
) -> Tuple[bool, bool, bool]:
    avoid_ok = True
    for O in obstacles:
        if bool(np.any(_in_rect(X, O))):
            avoid_ok = False
            break

    reach_all = True
    for typ, rects in targets_by_type.items():
        ok_t = any(bool(np.any(_in_rect(X, r))) for r in rects)
        if not ok_t:
            reach_all = False
            break

    sat = bool(avoid_ok and reach_all)
    return avoid_ok, reach_all, sat


def check_example5(
    *,
    X: np.ndarray,
    obstacles: List[Tuple[float, float, float, float]],
    doors: List[Tuple[float, float, float, float]],
    keys: List[Tuple[float, float, float, float]],
    G_rect: Tuple[float, float, float, float],
) -> Tuple[bool, bool, bool, bool]:
    avoid_ok = True
    for O in obstacles:
        if bool(np.any(_in_rect(X, O))):
            avoid_ok = False
            break

    reach_ok = bool(np.any(_in_rect(X, G_rect)))

    doors_ok = True
    for D, K in zip(doors, keys):
        inK = _in_rect(X, K)
        if not bool(np.any(inK)):
            doors_ok = False
            break
        t_key = int(np.argmax(inK))
        if bool(np.any(_in_rect(X[:t_key], D))):
            doors_ok = False
            break

    sat = bool(avoid_ok and reach_ok and doors_ok)
    return avoid_ok, reach_ok, doors_ok, sat


# ==============================================================================
# Benchmark utilities
# ==============================================================================

class BenchmarkStats:
    """
    Container for benchmark statistics collected during evaluation.
    """

    def __init__(self):
        self.times_total: List[float] = []
        self.times_qp: List[float] = []
        self.solved_k: List[int] = []
        self.fallback_count: int = 0
        self.costs: List[float] = []

    def add_case(
        self,
        *,
        time_total: float,
        time_qp: float,
        solved_k: int,
        used_fallback: bool,
        cost: float,
    ) -> None:
        self.times_total.append(float(time_total))
        self.times_qp.append(float(time_qp))
        self.solved_k.append(int(solved_k))
        if used_fallback:
            self.fallback_count += 1
        self.costs.append(float(cost))

    def compute(self) -> Dict[str, float]:
        times_total = np.asarray(self.times_total, dtype=float) if self.times_total else np.zeros((1,), dtype=float)
        times_qp = np.asarray(self.times_qp, dtype=float) if self.times_qp else np.zeros((1,), dtype=float)
        solved_k = np.asarray(self.solved_k, dtype=float) if self.solved_k else np.zeros((1,), dtype=float)
        costs = np.asarray(self.costs, dtype=float) if self.costs else np.zeros((1,), dtype=float)

        return {
            "time_total_mean": float(np.mean(times_total)),
            "time_total_std": float(np.std(times_total)),
            "time_qp_mean": float(np.mean(times_qp)),
            "time_qp_std": float(np.std(times_qp)),
            "solved_k_mean": float(np.mean(solved_k)),
            "solved_k_std": float(np.std(solved_k)),
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
        }


def start_timer() -> float:
    """
    Start a high-resolution timer.
    """
    return float(time.perf_counter())


def stop_timer(t0: float) -> float:
    """
    Stop timer and return elapsed seconds.
    """
    return float(time.perf_counter() - float(t0))


def solve_with_stlpy_fallback(
    *,
    sample: Dict[str, Any],
    example_id: int,
):
    """
    Solve STL problem using stlpy as fallback solver.
    """
    from stlpy.solvers import GurobiMICPSolver
    from stlpy_eval.scenario_builder import build_spec_and_system
    from stlpy_eval.scenario_builder import get_solver_config

    phi, sys, x0 = build_spec_and_system(sample=sample, example_id=int(example_id))
    cfg = get_solver_config(int(example_id))
    T = int(sample["T"])

    solver = GurobiMICPSolver(
        phi,
        sys,
        x0,
        T,
        robustness_cost=bool(cfg.robustness_cost),
    )
    solver.model.setParam("Threads", 16)                 # small integers for preventing segmentation faults (observed empirically)
    solver.model.setParam("OutputFlag", 0)
    solver.model.setParam("LogToConsole", 0)

    solver.AddControlBounds(cfg.u_min, cfg.u_max)
    solver.AddStateBounds(cfg.x_min, cfg.x_max)
    solver.AddQuadraticCost(cfg.Q, cfg.R)

    t0 = time.perf_counter()
    x, u, rho, _ = solver.Solve()
    t1 = time.perf_counter()

    return x, u, float(t1 - t0)


def compute_solution_cost(
    *,
    prob: Any,
    X: np.ndarray,
    U: np.ndarray,
) -> float:
    """
    Compute the quadratic running cost using prob.Q and prob.R.

    The planning pipeline defines Q and R on PlanningProblem in build_problem_fn,
    so evaluation must use those exact matrices.
    """
    if X is None or U is None:
        return float("nan")

    if not hasattr(prob, "Q") or not hasattr(prob, "R"):
        raise AttributeError("PlanningProblem must have attributes Q and R.")

    Q = np.asarray(prob.Q, dtype=float)
    R = np.asarray(prob.R, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    U_arr = np.asarray(U, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
    if U_arr.ndim != 2:
        raise ValueError(f"U must be 2D, got shape {U_arr.shape}")
    if X_arr.shape[0] == Q.shape[0] and X_arr.shape[1] != Q.shape[0]:
        X_arr = X_arr.T
    if U_arr.shape[0] == R.shape[0] and U_arr.shape[1] != R.shape[0]:
        U_arr = U_arr.T

    if X_arr.shape[1] != Q.shape[0]:
        raise ValueError(
            f"State dimension mismatch: X has shape {X_arr.shape}, Q has shape {Q.shape}"
        )
    if U_arr.shape[1] != R.shape[0]:
        raise ValueError(
            f"Control dimension mismatch: U has shape {U_arr.shape}, R has shape {R.shape}"
        )

    T_u = int(U_arr.shape[0])
    if X_arr.shape[0] < T_u:
        raise ValueError(
            f"State trajectory too short for control horizon: X shape {X_arr.shape}, U shape {U_arr.shape}"
        )

    cost = 0.0
    for t in range(T_u):
        xt = X_arr[t]
        ut = U_arr[t]
        cost += float(xt @ Q @ xt + ut @ R @ ut)
    return float(cost)


def write_summary_json(
    *,
    out_dir: str,
    solver_name: str,
    example_id: int,
    n_cases: int,
    success_rate: float,
    stats: BenchmarkStats,
) -> None:
    """
    Write standardized summary.json file.
    """
    agg = stats.compute()

    summary = {
        "solver": str(solver_name),
        "example_id": int(example_id),
        "n_cases": int(n_cases),
        "success_rate": float(success_rate),
        "time_total_mean": float(agg["time_total_mean"]),
        "time_total_std": float(agg["time_total_std"]),
        "time_qp_mean": float(agg["time_qp_mean"]),
        "time_qp_std": float(agg["time_qp_std"]),
        "solved_k_mean": float(agg["solved_k_mean"]),
        "solved_k_std": float(agg["solved_k_std"]),
        "cost_mean": float(agg["cost_mean"]),
        "cost_std": float(agg["cost_std"]),
        "fallback_rate": float(stats.fallback_count / max(1, int(n_cases))),
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)