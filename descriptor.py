from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from geometry import Rect
from examples import (
    GeometryConstraints,
    sample_valid_perturbation,
    make_example_1,
    make_example_2,
    make_example_3,
    make_example_4,
    make_example_5,
    validate_geometry,
)


def _rect_inflate(r: Rect, eps: float) -> Rect:
    x1, x2, y1, y2 = r
    return (x1 - eps, x2 + eps, y1 - eps, y2 + eps)


def _rects_overlap(a: Rect, b: Rect) -> bool:
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    return (min(ax2, bx2) > max(ax1, bx1)) and (min(ay2, by2) > max(ay1, by1))


def _point_in_rect(x: float, y: float, r: Rect) -> bool:
    x1, x2, y1, y2 = r
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def _sample_rect_uniform(
    *,
    rng: np.random.Generator,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    w: float,
    h: float,
) -> Rect:
    x_min, x_max = float(xlim[0]), float(xlim[1])
    y_min, y_max = float(ylim[0]), float(ylim[1])
    if (x_max - x_min) <= w or (y_max - y_min) <= h:
        raise ValueError("Workspace too small for requested rectangle size.")
    x1 = float(rng.uniform(x_min, x_max - w))
    y1 = float(rng.uniform(y_min, y_max - h))
    return (x1, x1 + w, y1, y1 + h)


def _sample_point_uniform(
    *,
    rng: np.random.Generator,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Tuple[float, float]:
    x = float(rng.uniform(float(xlim[0]), float(xlim[1])))
    y = float(rng.uniform(float(ylim[0]), float(ylim[1])))
    return x, y


def _shift_rect(r: Rect, dx: float, dy: float) -> Rect:
    x1, x2, y1, y2 = r
    return (float(x1) + float(dx), float(x2) + float(dx), float(y1) + float(dy), float(y2) + float(dy))


def _shift_x0(x0: np.ndarray, dx: float, dy: float) -> np.ndarray:
    x0p = np.array(x0, dtype=float).copy()
    x0p[0] += float(dx)
    x0p[1] += float(dy)
    return x0p


def _sample_x0_and_shifted_regions(
    *,
    rng: np.random.Generator,
    x0_base: np.ndarray,
    regions_base: Mapping[str, Rect],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    dx0_range: Tuple[float, float],
    dy0_range: Tuple[float, float],
    dR_range: Tuple[float, float, float, float],
    rules: GeometryConstraints,
    max_tries: int,
) -> Dict[str, Any]:
    rx_min, rx_max = float(dR_range[0]), float(dR_range[1])
    ry_min, ry_max = float(dR_range[2]), float(dR_range[3])

    for t in range(1, int(max_tries) + 1):
        dx0 = float(rng.uniform(dx0_range[0], dx0_range[1]))
        dy0 = float(rng.uniform(dy0_range[0], dy0_range[1]))
        x0 = _shift_x0(x0_base, dx0, dy0)

        regions: Dict[str, Rect] = {}
        dRx: Dict[str, float] = {}
        dRy: Dict[str, float] = {}

        for name, r0 in regions_base.items():
            rx = float(rng.uniform(rx_min, rx_max))
            ry = float(rng.uniform(ry_min, ry_max))
            regions[name] = _shift_rect(r0, rx, ry)
            dRx[name] = rx
            dRy[name] = ry

        ok, _ = validate_geometry(
            x0=x0,
            obstacles=[],
            regions=regions,
            xlim=xlim,
            ylim=ylim,
            rules=rules,
        )
        if ok:
            return {
                "x0": x0,
                "regions": regions,
                "deltas": {
                    "dx0": dx0,
                    "dy0": dy0,
                    "dOx": [],
                    "dOy": [],
                    "fixed": [],
                    "dRx": dRx,
                    "dRy": dRy,
                },
                "tries": t,
            }

    raise RuntimeError(f"Failed to sample valid x0/regions after max_tries={max_tries}.")


def _place_random_obstacles_last(
    *,
    rng: np.random.Generator,
    x0: np.ndarray,
    regions: Mapping[str, Rect],
    n_obstacles: int,
    obstacle_w: float,
    obstacle_h: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    rules: GeometryConstraints,
    max_tries: int,
) -> List[Rect]:
    obstacles: List[Rect] = []
    for i in range(int(n_obstacles)):
        placed = False
        for _ in range(int(max_tries)):
            cand = _sample_rect_uniform(
                rng=rng,
                xlim=xlim,
                ylim=ylim,
                w=obstacle_w,
                h=obstacle_h,
            )
            ok, _ = validate_geometry(
                x0=x0,
                obstacles=obstacles + [cand],
                regions=regions,
                xlim=xlim,
                ylim=ylim,
                rules=rules,
            )
            if ok:
                obstacles.append(cand)
                placed = True
                break
        if not placed:
            raise RuntimeError(f"Failed to place obstacle {i} within max_tries={max_tries}.")
    return obstacles


@dataclass(frozen=True)
class BoundsSpec:
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray


@dataclass(frozen=True)
class ExampleDescriptor:
    example_id: int
    T: int
    x0_base: np.ndarray
    bounds: BoundsSpec
    obstacles_base: List[Rect]
    regions_base: Dict[str, Rect]
    meta_base: Dict[str, Any]

    def sample_env(
        self,
        *,
        rng: np.random.Generator,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        dx0_range: Tuple[float, float],
        dy0_range: Tuple[float, float],
        dO_range: Tuple[float, float, float, float],
        dR_range: Tuple[float, float, float, float],
        rules: GeometryConstraints,
        max_tries: int,
        fixed_obstacle_mask: Optional[Sequence[bool]] = None,
    ) -> Dict[str, Any]:
        if int(self.example_id) == 5:
            meta = self.meta_base or {}
            n_doors = int(meta.get("n_doors", 2))
            if n_doors != 2:
                raise ValueError(f"Example 5 expects n_doors=2, got {n_doors}")

            if len(self.obstacles_base) < 5:
                raise ValueError("Example 5 requires at least 5 fixed obstacles in obstacles_base.")

            fixed_obs: List[Rect] = [tuple(r) for r in self.obstacles_base[:5]]
            G: Rect = tuple(self.regions_base["G"])
            D0: Rect = tuple(self.regions_base["D0"])
            D1: Rect = tuple(self.regions_base["D1"])

            key_w = float(meta.get("key_w", 1.0))
            key_h = float(meta.get("key_h", 1.0))
            extra_w = float(meta.get("extra_obstacle_w", 2.0))
            extra_h = float(meta.get("extra_obstacle_h", 2.0))

            eps = float(getattr(rules, "min_goal_obstacle_clearance", 0.0))
            eps = max(eps, float(getattr(rules, "min_start_obstacle_clearance", 0.0)), 0.0)

            blocked: List[Rect] = (
                [_rect_inflate(r, eps) for r in fixed_obs]
                + [_rect_inflate(D0, eps), _rect_inflate(D1, eps), _rect_inflate(G, eps)]
            )

            tries = 0

            x0x = x0y = 0.0
            ok = False
            for _ in range(int(max_tries)):
                tries += 1
                x, y = _sample_point_uniform(rng=rng, xlim=xlim, ylim=ylim)
                if any(_point_in_rect(x, y, r) for r in blocked):
                    continue
                x0x, x0y = float(x), float(y)
                ok = True
                break
            if not ok:
                raise RuntimeError("Example 5: failed to sample valid x0 within max_tries.")

            keys: List[Rect] = []
            for ki in range(2):
                ok = False
                for _ in range(int(max_tries)):
                    tries += 1
                    r = _sample_rect_uniform(rng=rng, xlim=xlim, ylim=ylim, w=key_w, h=key_h)
                    if _point_in_rect(x0x, x0y, r):
                        continue
                    if any(_rects_overlap(r, br) for br in blocked):
                        continue
                    if any(_rects_overlap(_rect_inflate(r, eps), _rect_inflate(kprev, eps)) for kprev in keys):
                        continue
                    keys.append(r)
                    ok = True
                    break
                if not ok:
                    raise RuntimeError(f"Example 5: failed to sample key K{ki} within max_tries.")

            extra_count = max(0, len(self.obstacles_base) - 5)
            extras: List[Rect] = []
            for oi in range(extra_count):
                ok = False
                for _ in range(int(max_tries)):
                    tries += 1
                    r = _sample_rect_uniform(rng=rng, xlim=xlim, ylim=ylim, w=extra_w, h=extra_h)
                    if _point_in_rect(x0x, x0y, r):
                        continue
                    if any(_rects_overlap(r, br) for br in blocked):
                        continue
                    if any(_rects_overlap(_rect_inflate(r, eps), _rect_inflate(k, eps)) for k in keys):
                        continue
                    if any(_rects_overlap(_rect_inflate(r, eps), _rect_inflate(e, eps)) for e in extras):
                        continue
                    extras.append(r)
                    ok = True
                    break
                if not ok:
                    raise RuntimeError(f"Example 5: failed to sample extra obstacle {oi} within max_tries.")

            return {
                "x0": [x0x, x0y, 0.0, 0.0],
                "obstacles": [list(r) for r in (fixed_obs + extras)],
                "regions": {
                    "G": list(G),
                    "D0": list(D0),
                    "D1": list(D1),
                    "K0": list(keys[0]),
                    "K1": list(keys[1]),
                },
                "meta": dict(self.meta_base),
                "deltas": {},
                "tries": int(tries),
            }

        if int(self.example_id) in (1, 2):
            meta = self.meta_base or {}

            base = _sample_x0_and_shifted_regions(
                rng=rng,
                x0_base=self.x0_base,
                regions_base=self.regions_base,
                xlim=xlim,
                ylim=ylim,
                dx0_range=dx0_range,
                dy0_range=dy0_range,
                dR_range=dR_range,
                rules=rules,
                max_tries=max_tries,
            )

            obstacles_now = _place_random_obstacles_last(
                rng=rng,
                x0=base["x0"],
                regions=base["regions"],
                n_obstacles=len(self.obstacles_base),
                obstacle_w=float(meta.get("obstacle_w", 2.0)),
                obstacle_h=float(meta.get("obstacle_h", 2.0)),
                xlim=xlim,
                ylim=ylim,
                rules=rules,
                max_tries=max_tries,
            )

            return {
                "x0": base["x0"].tolist(),
                "obstacles": [list(r) for r in obstacles_now],
                "regions": {k: list(v) for k, v in base["regions"].items()},
                "meta": dict(self.meta_base),
                "deltas": base["deltas"],
                "tries": int(base["tries"]),
            }

        if int(self.example_id) == 3:
            meta = self.meta_base or {}

            goal_w = float(meta.get("goal_w", 1.0))
            goal_h = float(meta.get("goal_h", 1.0))

            obstacles_now: List[Rect] = [tuple(r) for r in self.obstacles_base]

            tries = 0
            x0_now: Optional[np.ndarray] = None
            dx0 = 0.0
            dy0 = 0.0

            for _ in range(int(max_tries)):
                tries += 1
                dx0 = float(rng.uniform(dx0_range[0], dx0_range[1]))
                dy0 = float(rng.uniform(dy0_range[0], dy0_range[1]))
                cand_x0 = _shift_x0(self.x0_base, dx0, dy0)

                ok, _ = validate_geometry(
                    x0=cand_x0,
                    obstacles=obstacles_now,
                    regions={},
                    xlim=xlim,
                    ylim=ylim,
                    rules=rules,
                )
                if ok:
                    x0_now = cand_x0
                    break

            if x0_now is None:
                raise RuntimeError("Failed to place valid x0 for Example 3.")

            goal_specs = [
                ("G0", (5.0, float(xlim[1])), (6.0, float(ylim[1]))),
                ("G1", (9.0, float(xlim[1])), (float(ylim[0]), float(ylim[1]))),
            ]

            regions_now: Dict[str, Rect] = {}
            placed_goals: List[Rect] = []

            for name, gxlim, gylim in goal_specs:
                placed = False
                for _ in range(int(max_tries)):
                    tries += 1

                    r = _sample_rect_uniform(
                        rng=rng,
                        xlim=gxlim,
                        ylim=gylim,
                        w=goal_w,
                        h=goal_h,
                    )

                    if any(_rects_overlap(r, prev) for prev in placed_goals):
                        continue

                    test_regions = dict(regions_now)
                    test_regions[name] = r

                    ok, _ = validate_geometry(
                        x0=x0_now,
                        obstacles=obstacles_now,
                        regions=test_regions,
                        xlim=xlim,
                        ylim=ylim,
                        rules=rules,
                    )
                    if not ok:
                        continue

                    regions_now[name] = r
                    placed_goals.append(r)
                    placed = True
                    break

                if not placed:
                    raise RuntimeError(
                        f"Failed to place goal {name} for Example 3 within max_tries={max_tries}."
                    )

            return {
                "x0": x0_now.tolist(),
                "obstacles": [list(r) for r in obstacles_now],
                "regions": {k: list(v) for k, v in regions_now.items()},
                "meta": dict(self.meta_base),
                "deltas": {
                    "dx0": dx0,
                    "dy0": dy0,
                    "dOx": [],
                    "dOy": [],
                    "fixed": [True] * len(obstacles_now),
                    "dRx": {},
                    "dRy": {},
                },
                "tries": int(tries),
            }

        if int(self.example_id) == 4:
            meta = self.meta_base or {}
            types_list = meta.get("types", None)
            targets_per_type = int(meta.get("targets_per_type", 2))
            if types_list is None:
                raise ValueError("Example 4 requires meta['types'] to be defined.")
            types_list = list(types_list)
            if len(types_list) < 1:
                raise ValueError("Example 4 requires at least one type in meta['types'].")
            if targets_per_type < 1:
                raise ValueError("Example 4 requires targets_per_type >= 1.")

            dx0 = float(rng.uniform(dx0_range[0], dx0_range[1]))
            dy0 = float(rng.uniform(dy0_range[0], dy0_range[1]))
            x0_now = _shift_x0(self.x0_base, dx0, dy0)

            ok, _ = validate_geometry(
                x0=x0_now,
                obstacles=[],
                regions={},
                xlim=xlim,
                ylim=ylim,
                rules=rules,
            )
            if not ok:
                raise RuntimeError("Failed to place valid x0 for Example 4.")

            tw = float(meta.get("target_w", 1.0))
            th = float(meta.get("target_h", 1.0))
            eps_tgt = float(meta.get("target_target_clearance", 0.0))
            target_tries_budget = int(meta.get("target_max_tries", max_tries))

            placed: List[Rect] = []
            regions_now: Dict[str, Rect] = {}

            tries = 0
            for c in types_list:
                for k in range(targets_per_type):
                    ok = False
                    while tries < target_tries_budget:
                        tries += 1
                        r = _sample_rect_uniform(rng=rng, xlim=xlim, ylim=ylim, w=tw, h=th)

                        if _point_in_rect(float(x0_now[0]), float(x0_now[1]), r):
                            continue

                        bad = False
                        r_chk = _rect_inflate(r, eps_tgt) if eps_tgt > 0.0 else r
                        for tprev in placed:
                            tchk = _rect_inflate(tprev, eps_tgt) if eps_tgt > 0.0 else tprev
                            if _rects_overlap(r_chk, tchk):
                                bad = True
                                break
                        if bad:
                            continue

                        placed.append(r)
                        regions_now[f"T_{c}_{k}"] = r
                        ok = True
                        break

                    if not ok:
                        raise RuntimeError(f"Failed to place target T_{c}_{k} within {target_tries_budget} tries.")

            obstacles_now = _place_random_obstacles_last(
                rng=rng,
                x0=x0_now,
                regions=regions_now,
                n_obstacles=len(self.obstacles_base),
                obstacle_w=float(meta.get("obstacle_w", 2.0)),
                obstacle_h=float(meta.get("obstacle_h", 2.0)),
                xlim=xlim,
                ylim=ylim,
                rules=rules,
                max_tries=max_tries,
            )

            return {
                "x0": x0_now.tolist(),
                "obstacles": [list(r) for r in obstacles_now],
                "regions": {k: list(v) for k, v in regions_now.items()},
                "meta": dict(self.meta_base),
                "deltas": {
                    "dx0": dx0,
                    "dy0": dy0,
                    "dOx": [],
                    "dOy": [],
                    "fixed": [False] * len(self.obstacles_base),
                    "dRx": {},
                    "dRy": {},
                },
                "tries": int(tries),
            }

        if fixed_obstacle_mask is None:
            m = self.meta_base.get("fixed_obstacle_mask", None)
            if m is not None:
                fixed_obstacle_mask = list(m)

        pert = sample_valid_perturbation(
            x0_base=self.x0_base,
            obstacles_base=self.obstacles_base,
            regions_base=self.regions_base,
            xlim=xlim,
            ylim=ylim,
            dx0_range=dx0_range,
            dy0_range=dy0_range,
            dO_range=dO_range,
            dR_range=dR_range,
            rules=rules,
            rng=rng,
            max_tries=max_tries,
            fixed_obstacle_mask=fixed_obstacle_mask,
        )

        return {
            "x0": pert["x0"].tolist(),
            "obstacles": [list(r) for r in pert["obstacles"]],
            "regions": {k: list(v) for k, v in pert["regions"].items()},
            "meta": dict(self.meta_base),
            "deltas": pert["deltas"],
            "tries": int(pert["tries"]),
        }

    def build_ast_from_sample(self, sample: Dict[str, Any]):
        T = int(sample["T"])
        env = sample["env"]
        obstacles = [tuple(r) for r in env["obstacles"]]
        regions = {k: tuple(v) for k, v in env["regions"].items()}
        meta = env.get("meta", {})

        ex = int(sample.get("example_id", self.example_id))

        if ex == 1:
            return make_example_1(T, obstacles, regions["G"])

        if ex == 2:
            tau = int(meta["tau"])
            return make_example_2(T, tau, obstacles, regions["G"], regions["T1"], regions["T2"])

        if ex == 3:
            goals = [regions[k] for k in sorted(regions.keys()) if k.startswith("G")]
            return make_example_3(T, obstacles, goals)

        if ex == 4:
            targets_by_type: Dict[str, List[Rect]] = {}
            for key in regions.keys():
                if not key.startswith("T_"):
                    continue
                parts = key.split("_", 2)
                if len(parts) != 3:
                    continue
                c = parts[1]
                targets_by_type.setdefault(c, []).append(regions[key])
            return make_example_4(T, obstacles, targets_by_type)

        if ex == 5:
            doors: List[Rect] = []
            keys: List[Rect] = []
            for k in sorted(regions.keys()):
                if k.startswith("D"):
                    doors.append(regions[k])
                if k.startswith("K"):
                    keys.append(regions[k])
            return make_example_5(T, obstacles, doors, keys, regions["G"])

        raise ValueError(f"Unsupported example_id: {ex}")


def _bounds_default() -> BoundsSpec:
    return BoundsSpec(
        x_min=np.array([0.0, 0.0, -1.0, -1.0], dtype=float),
        x_max=np.array([10.0, 10.0, 1.0, 1.0], dtype=float),
        u_min=np.array([-0.5, -0.5], dtype=float),
        u_max=np.array([0.5, 0.5], dtype=float),
    )


def get_descriptor(
    example_id: int,
    *,
    n_obstacles: int = 1,
    n_goals: int = 2,
    types: Optional[Sequence[str]] = None,
    targets_per_type: int = 2,
    n_doors: int = 2,
) -> ExampleDescriptor:
    ex = int(example_id)
    n_obstacles = int(n_obstacles)
    if n_obstacles < 1:
        raise ValueError(f"n_obstacles must be >= 1, got {n_obstacles}")

    b = _bounds_default()

    if ex == 1:
        T = 20
        x0 = np.array([2.0, 2.0, 0.0, 0.0], dtype=float)

        O0: Rect = (3.0, 5.0, 4.0, 6.0)
        obstacles = [(O0[0] + 0.5 * i, O0[1] + 0.5 * i, O0[2] - 0.3 * i, O0[3] - 0.3 * i) for i in range(n_obstacles)]

        G: Rect = (7.0, 8.0, 8.0, 9.0)
        regions = {"G": G}

        meta = {
            "obstacle_w": float(O0[1] - O0[0]),
            "obstacle_h": float(O0[3] - O0[2]),
        }
        return ExampleDescriptor(ex, T, x0, b, list(obstacles), dict(regions), dict(meta))

    if ex == 2:
        T = 25
        tau = 5
        x0 = np.array([2.0, 2.0, 0.0, 0.0], dtype=float)

        O0: Rect = (3.0, 4.5, 4.0, 5.5)
        obstacles = [(O0[0] + 0.4 * i, O0[1] + 0.4 * i, O0[2] + 0.2 * i, O0[3] + 0.2 * i) for i in range(n_obstacles)]

        regions = {
            "G": (8.0, 9.0, 8.0, 9.0),
            "T1": (7.0, 8.0, 2.0, 3.0),
            "T2": (2.0, 3.0, 7.0, 8.0),
        }
        meta = {
            "tau": int(tau),
            "obstacle_w": float(O0[1] - O0[0]),
            "obstacle_h": float(O0[3] - O0[2]),
        }
        return ExampleDescriptor(ex, T, x0, b, list(obstacles), dict(regions), dict(meta))

    if ex == 3:
        T = 25
        x0 = np.array([3.0, 2.6, 0.0, 0.0], dtype=float)

        rects: List[Rect] = [
            (2.0, 5.0, 4.0, 6.0),
            (5.5, 9.0, 3.8, 5.7),
            (4.6, 8.0, 0.5, 3.5),
            (2.2, 4.4, 6.4, 11.0),
        ]

        if n_obstacles > 4:
            raise ValueError(f"n_obstacles must be <= 4 for Example 3 (fixed layout), got {n_obstacles}")

        obstacles = rects[:n_obstacles]

        regions: Dict[str, Rect] = {
            "G0": (7.0, 8.0, 8.0, 9.0),
            "G1": (9.5, 10.5, 1.5, 2.5),
        }

        b3 = BoundsSpec(
            x_min=np.array([0.0, 0.0, -1.0, -1.0], dtype=float),
            x_max=np.array([12.0, 12.0, 1.0, 1.0], dtype=float),
            u_min=np.array([-0.5, -0.5], dtype=float),
            u_max=np.array([0.5, 0.5], dtype=float),
        )

        meta = {
            "n_goals": 2,
            "goal_w": 1.0,
            "goal_h": 1.0,
            "fixed_obstacle_mask": [True] * n_obstacles,
        }

        return ExampleDescriptor(ex, T, x0, b3, list(obstacles), dict(regions), dict(meta))

    if ex == 4:
        T = 30
        x0 = np.array([2.0, 2.0, 0.0, 0.0], dtype=float)

        O0: Rect = (3.0, 5.0, 4.0, 6.0)
        obstacles = [(O0[0] + 0.7 * i, O0[1] + 0.7 * i, O0[2] + 0.1 * i, O0[3] + 0.1 * i) for i in range(n_obstacles)]

        if types is None:
            types_list = ["type0", "type1", "type2"]
        else:
            types_list = list(types)
            if len(types_list) < 1:
                raise ValueError("types must contain at least one type string.")

        targets_per_type = int(targets_per_type)
        if targets_per_type < 1:
            raise ValueError(f"targets_per_type must be >= 1, got {targets_per_type}")

        base_target: Rect = (7.0, 8.0, 2.0, 3.0)
        regions: Dict[str, Rect] = {}
        for ti, c in enumerate(types_list):
            for k in range(targets_per_type):
                rect = (
                    base_target[0] - 0.8 * ti,
                    base_target[1] - 0.8 * ti,
                    base_target[2] + 0.9 * k,
                    base_target[3] + 0.9 * k,
                )
                regions[f"T_{c}_{k}"] = rect

        meta = {
            "types": list(types_list),
            "targets_per_type": int(targets_per_type),
            "target_w": 1.0,
            "target_h": 1.0,
            "target_target_clearance": 0.0,
            "target_max_tries": int(20 * max(1, len(types_list) * targets_per_type)),
            "obstacle_w": float(O0[1] - O0[0]),
            "obstacle_h": float(O0[3] - O0[2]),
        }
        return ExampleDescriptor(ex, T, x0, b, list(obstacles), dict(regions), dict(meta))

    if ex == 5:
        T = 35

        b5 = BoundsSpec(
            x_min=np.array([0.0, 0.0, -1.0, -1.0], dtype=float),
            x_max=np.array([15.0, 10.0, 1.0, 1.0], dtype=float),
            u_min=np.array([-0.5, -0.5], dtype=float),
            u_max=np.array([0.5, 0.5], dtype=float),
        )

        x0 = np.array([6.0, 1.0, 0.0, 0.0], dtype=float)

        if n_obstacles < 5:
            raise ValueError(f"Example 5 requires n_obstacles >= 5 (first five are fixed walls), got {n_obstacles}")

        n_doors = int(n_doors)
        if n_doors != 2:
            raise ValueError(f"Example 5 requires n_doors == 2 (fixed doors), got {n_doors}")

        fixed_obstacles: List[Rect] = [
            (8.0, 15.1, -0.1, 4.0),
            (8.0, 15.1, 6.0, 10.1),
            (3.5, 5.0, -0.1, 2.5),
            (-0.1, 2.5, 4.0, 6.0),
            (3.5, 5.0, 7.5, 10.1),
        ]

        extra_count = n_obstacles - 5
        extra_placeholders: List[Rect] = [(0.0, 0.0, 0.0, 0.0) for _ in range(extra_count)]
        obstacles = fixed_obstacles + extra_placeholders

        regions: Dict[str, Rect] = {
            "G": (14.1, 14.9, 4.1, 5.9),
            "D0": (12.8, 14.0, 3.99, 6.01),
            "D1": (11.5, 12.7, 3.99, 6.01),
            "K0": (0.0, 0.0, 0.0, 0.0),
            "K1": (0.0, 0.0, 0.0, 0.0),
        }

        fixed_mask = [True] * 5 + [False] * max(0, extra_count)

        meta = {
            "n_doors": 2,
            "fixed_obstacle_mask": fixed_mask,
            "key_w": 1.0,
            "key_h": 1.0,
            "extra_obstacle_w": 2.0,
            "extra_obstacle_h": 2.0,
        }

        return ExampleDescriptor(ex, T, x0, b5, list(obstacles), dict(regions), dict(meta))

    raise ValueError(f"Unsupported example_id={ex}")