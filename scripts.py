# scripts.py
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from dynamics import DoubleIntegrator2D
from examples import GeometryConstraints
from descriptor import get_descriptor
from micp import PlanningProblem, generate_dataset


def build_samples(
    *,
    n: int,
    seed: int,
    example_id: int,
    dx0_range: Tuple[float, float],
    dy0_range: Tuple[float, float],
    dO_range: Tuple[float, float, float, float],
    dR_range: Tuple[float, float, float, float],
    rules: GeometryConstraints,
    max_tries: int,
    # descriptor controls
    n_obstacles: int = 1,
    n_goals: int = 2,
    types: Optional[List[str]] = None,
    targets_per_type: int = 2,
    n_doors: int = 2,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    desc = get_descriptor(
        int(example_id),
        n_obstacles=int(n_obstacles),
        n_goals=int(n_goals),
        types=types,
        targets_per_type=int(targets_per_type),
        n_doors=int(n_doors),
    )

    bounds = {
        "x_min": desc.bounds.x_min,
        "x_max": desc.bounds.x_max,
        "u_min": desc.bounds.u_min,
        "u_max": desc.bounds.u_max,
    }

    xlim = (float(bounds["x_min"][0]), float(bounds["x_max"][0]))
    ylim = (float(bounds["x_min"][1]), float(bounds["x_max"][1]))

    samples: List[Dict[str, Any]] = []
    for _ in range(n):
        env = desc.sample_env(
            rng=rng,
            xlim=xlim,
            ylim=ylim,
            dx0_range=dx0_range,
            dy0_range=dy0_range,
            dO_range=dO_range,
            dR_range=dR_range,
            rules=rules,
            max_tries=max_tries,
            fixed_obstacle_mask=None,
        )

        samples.append({
            "example_id": int(example_id),
            "T": int(desc.T),
            "bounds": {
                "x_min": bounds["x_min"].tolist(),
                "x_max": bounds["x_max"].tolist(),
                "u_min": bounds["u_min"].tolist(),
                "u_max": bounds["u_max"].tolist(),
            },
            "env": env,
        })

    return samples


def build_ast_fn(sample: Dict[str, Any]):
    # Avoid special handlers: reconstruct descriptor only from sample-level config is unnecessary.
    # Use descriptor registry with defaults encoded in env["meta"] when present.
    ex = int(sample.get("example_id", 1))

    # Recover descriptor configuration from env["meta"] when it exists;
    # if a key is absent, get_descriptor default applies.
    meta = sample.get("env", {}).get("meta", {}) or {}
    types = meta.get("types", None)

    desc = get_descriptor(
        ex,
        n_obstacles=len(sample["env"]["obstacles"]),
        n_goals=int(meta.get("n_goals", 2)),
        types=types,
        targets_per_type=int(meta.get("targets_per_type", 2)),
        n_doors=int(meta.get("n_doors", 2)),
    )
    return desc.build_ast_from_sample(sample)


def build_problem_fn(sample: Dict[str, Any]) -> PlanningProblem:
    T = int(sample["T"])
    b = sample["bounds"]
    env = sample["env"]

    dyn = DoubleIntegrator2D(dt=1.0)

    x0 = np.array(env["x0"], dtype=float)
    x_min = np.array(b["x_min"], dtype=float)
    x_max = np.array(b["x_max"], dtype=float)
    u_min = np.array(b["u_min"], dtype=float)
    u_max = np.array(b["u_max"], dtype=float)

    Q = 1e-1 * np.diag([0.0, 0.0, 1.0, 1.0])
    R = 1e-1 * np.eye(2)

    return PlanningProblem(
        T=T,
        dynamics=dyn,
        x0=x0,
        x_min=x_min,
        x_max=x_max,
        u_min=u_min,
        u_max=u_max,
        Q=Q,
        R=R,
        bigM=1e3,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/dataset.jsonl.gz")
    parser.add_argument("--example_id", type=int, default=1)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gurobi_out", type=int, default=0)

    parser.add_argument("--dx0", type=float, nargs=2, default=[-0.5, 0.5])
    parser.add_argument("--dy0", type=float, nargs=2, default=[-0.5, 0.5])
    parser.add_argument("--dO", type=float, nargs=4, default=[-0.5, 0.5, -0.5, 0.5])
    parser.add_argument("--dR", type=float, nargs=4, default=[-0.5, 0.5, -0.5, 0.5])

    parser.add_argument("--goal_obs_clear", type=float, default=0.2)
    parser.add_argument("--start_obs_clear", type=float, default=0.2)
    parser.add_argument("--max_tries", type=int, default=1000)

    # Descriptor controls (defaults preserve prior CLI behavior for ex1/ex2)
    parser.add_argument("--n_obstacles", type=int, default=1)
    parser.add_argument("--n_goals", type=int, default=1)
    parser.add_argument("--types", type=str, nargs="*", default=None)
    parser.add_argument("--targets_per_type", type=int, default=2)
    parser.add_argument("--n_doors", type=int, default=2)

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

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
        example_id=int(args.example_id),
        dx0_range=(float(args.dx0[0]), float(args.dx0[1])),
        dy0_range=(float(args.dy0[0]), float(args.dy0[1])),
        dO_range=(float(args.dO[0]), float(args.dO[1]), float(args.dO[2]), float(args.dO[3])),
        dR_range=(float(args.dR[0]), float(args.dR[1]), float(args.dR[2]), float(args.dR[3])),
        rules=rules,
        max_tries=int(args.max_tries),
        n_obstacles=int(args.n_obstacles),
        n_goals=int(args.n_goals),
        types=list(args.types) if args.types is not None else None,
        targets_per_type=int(args.targets_per_type),
        n_doors=int(args.n_doors),
    )

    generate_dataset(
        out_path=str(args.out),
        samples=samples,
        build_ast_fn=build_ast_fn,
        build_problem_fn=build_problem_fn,
        output_flag=int(args.gurobi_out),
    )


if __name__ == "__main__":
    main()