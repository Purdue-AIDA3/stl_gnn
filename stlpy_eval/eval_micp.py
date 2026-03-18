from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from examples import GeometryConstraints
from scripts import build_problem_fn, build_samples
from stlpy.solvers import GurobiMICPSolver
from gnn_train.eval_shared import BenchmarkStats, compute_solution_cost, write_summary_json

from .plotting import save_case_png
from .scenario_builder import build_spec_and_system, get_solver_config


def _status_from_solution(x: np.ndarray | None) -> str:
    return "OPTIMAL" if x is not None else "FAILED"


def _solve_one_case(
    *,
    sample: Dict[str, Any],
    example_id: int,
    gurobi_out: int,
):
    prob = build_problem_fn(sample)
    phi, sys, x0 = build_spec_and_system(sample=sample, example_id=int(example_id))
    cfg = get_solver_config(int(example_id))
    T = int(sample["T"])

    t0 = time.perf_counter()
    solver = GurobiMICPSolver(
        phi,
        sys,
        x0,
        T,
        robustness_cost=bool(cfg.robustness_cost),
    )
    solver.model.setParam("Threads", 16)                 # small integers for preventing segmentation faults (observed empirically)
    solver.model.setParam("OutputFlag", gurobi_out)
    solver.model.setParam("LogToConsole", gurobi_out)

    solver.AddControlBounds(cfg.u_min, cfg.u_max)
    solver.AddStateBounds(cfg.x_min, cfg.x_max)
    solver.AddQuadraticCost(cfg.Q, cfg.R)

    x, u, rho, _ = solver.Solve()
    t1 = time.perf_counter()

    wall_time = float(t1 - t0)
    status = _status_from_solution(x)
    cost = compute_solution_cost(prob=prob, X=x, U=u) if (x is not None and u is not None) else float("nan")

    return {
        "x": x,
        "u": u,
        "rho": rho,
        "status": status,
        "solve_time": wall_time,
        "cost": cost,
    }


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--example_id", type=int, required=True)

    p.add_argument("--dx0", type=float, nargs=2, default=[-0.5, 0.5])
    p.add_argument("--dy0", type=float, nargs=2, default=[-0.5, 0.5])
    p.add_argument("--dO", type=float, nargs=4, default=[-0.5, 0.5, -0.5, 0.5])
    p.add_argument("--dG", type=float, nargs=4, default=[-0.5, 0.5, -0.5, 0.5])
    p.add_argument("--goal_obs_clear", type=float, default=0.2)
    p.add_argument("--start_obs_clear", type=float, default=0.2)
    p.add_argument("--max_tries", type=int, default=300)
    p.add_argument("--n_obstacles", type=int, default=1)

    p.add_argument("--n_goals", type=int, default=2)
    p.add_argument("--targets_per_type", type=int, default=2)
    p.add_argument("--n_doors", type=int, default=2)

    p.add_argument("--save_png", type=int, default=0)
    p.add_argument("--png_dir", type=str, default="results/stlpy_eval_results")
    p.add_argument("--summary_json", type=str, default="")
    p.add_argument("--gurobi_out", type=int, default=0)

    args = p.parse_args()

    ex_id = int(args.example_id)
    if ex_id not in (1, 2, 3, 4, 5):
        raise ValueError(f"eval_micp supports example_id 1..5. Got {ex_id}")

    rules = GeometryConstraints(
        min_goal_obstacle_clearance=float(args.goal_obs_clear),
        min_start_obstacle_clearance=float(args.start_obs_clear),
        min_start_goal_clearance=0.0,
        require_rects_in_bounds=True,
        require_start_in_bounds=True,
    )

    samples = build_samples(
        n=int(args.n),
        seed=int(args.seed),
        example_id=ex_id,
        dx0_range=(float(args.dx0[0]), float(args.dx0[1])),
        dy0_range=(float(args.dy0[0]), float(args.dy0[1])),
        dO_range=(float(args.dO[0]), float(args.dO[1]), float(args.dO[2]), float(args.dO[3])),
        dR_range=(float(args.dG[0]), float(args.dG[1]), float(args.dG[2]), float(args.dG[3])),
        rules=rules,
        max_tries=int(args.max_tries),
        n_obstacles=int(args.n_obstacles),
        n_goals=int(args.n_goals),
        types=None,
        targets_per_type=int(args.targets_per_type),
        n_doors=int(args.n_doors),
    )

    os.makedirs(str(args.png_dir), exist_ok=True)

    n_ok = 0
    times: List[float] = []
    rhos: List[float] = []
    per_case: List[Dict[str, Any]] = []
    bench_stats = BenchmarkStats()

    for i, sample in enumerate(samples):
        try:
            res = _solve_one_case(
                sample=sample,
                example_id=ex_id,
                gurobi_out=int(args.gurobi_out),
            )
        except Exception as e:
            res = {
                "x": None,
                "u": None,
                "rho": None,
                "status": f"ERROR: {type(e).__name__}: {e}",
                "solve_time": float("nan"),
                "cost": float("nan"),
            }

        x = res["x"]
        u = res["u"]
        rho = res["rho"]
        status = str(res["status"])
        solve_time = float(res["solve_time"])
        cost = float(res["cost"])

        ok = x is not None
        n_ok += int(ok)
        if np.isfinite(solve_time):
            times.append(solve_time)
        if rho is not None:
            try:
                rhos.append(float(rho))
            except Exception:
                pass
        if ok:
            bench_stats.add_case(
                time_total=float(solve_time),
                time_qp=float(solve_time),
                solved_k=1,
                used_fallback=False,
                cost=float(cost),
            )

        if int(args.save_png) == 1 and x is not None:
            save_case_png(
                sample=sample,
                X=x,
                example_id=ex_id,
                save_path=os.path.join(str(args.png_dir), f"case_{i:04d}.png"),
            )

        per_case.append(
            {
                "case_idx": int(i),
                "status": status,
                "solve_time": solve_time,
                "cost": cost,
                "rho": (None if rho is None else float(rho)),
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == int(args.n):
            avg_t = sum(times) / max(1, len(times))
            print(
                f"[{i + 1:4d}/{int(args.n):4d}] "
                f"ok={n_ok} "
                f"avg_time={avg_t:.4f}s"
            )

    avg_time = sum(times) / max(1, len(times)) if len(times) > 0 else float("nan")
    med_time = float(np.median(np.asarray(times))) if len(times) > 0 else float("nan")
    avg_rho = sum(rhos) / max(1, len(rhos)) if len(rhos) > 0 else float("nan")
    agg = bench_stats.compute()

    print("\n==== Summary ====")
    print(f"example_id={ex_id}  n={int(args.n)}  seed={int(args.seed)}")
    print(f"success={int(n_ok)}/{int(args.n)}")
    print(f"avg_time={float(avg_time):.6f}s")
    print(f"median_time={float(med_time):.6f}s")
    print(f"cost_mean={float(agg['cost_mean']):.6f}  cost_std={float(agg['cost_std']):.6f}")
    print(f"avg_rho={float(avg_rho):.6f}")

    summary_json = str(args.summary_json).strip()
    if summary_json == "":
        summary_json = os.path.join(str(args.png_dir), "summary.json")

    os.makedirs(os.path.dirname(summary_json), exist_ok=True)
    write_summary_json(
        out_dir=str(os.path.dirname(summary_json)),
        solver_name="stlpy",
        example_id=int(ex_id),
        n_cases=int(args.n),
        success_rate=float(n_ok / max(1, int(args.n))),
        stats=bench_stats,
    )

    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    summary["seed"] = int(args.seed)
    summary["avg_time"] = float(avg_time)
    summary["median_time"] = float(med_time)
    summary["avg_rho"] = float(avg_rho)
    summary["n_success"] = int(n_ok)
    summary["per_case"] = per_case

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved summary to: {summary_json}")


if __name__ == "__main__":
    main()