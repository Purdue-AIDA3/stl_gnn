from __future__ import annotations

import argparse
import gzip
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Rect = Tuple[float, float, float, float]


def _import_eval_vis():
    """
    Prefer repo import:
      from gnn_train.eval_vis import visualize_qp_result
    Fallback to local eval_vis.py if this script is run inside gnn_train/.
    """
    try:
        from gnn_train.eval_vis import visualize_qp_result
        return visualize_qp_result
    except Exception:
        import eval_vis
        return eval_vis.visualize_qp_result


def _as_np_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        try:
            return np.asarray(x, dtype=float)
        except Exception:
            return None
    return None


def _read_jsonl_gz(path: str):
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _infer_example_id(rec: Dict[str, Any]) -> Optional[int]:
    sample = rec.get("sample", {}) or {}
    if "example_id" in sample:
        try:
            return int(sample["example_id"])
        except Exception:
            pass
    if "example_id" in rec:
        try:
            return int(rec["example_id"])
        except Exception:
            pass
    return None


def _get_obstacles(env: Dict[str, Any]) -> List[Rect]:
    obs = env.get("obstacles", []) or []
    out: List[Rect] = []
    for r in obs:
        if isinstance(r, (list, tuple)) and len(r) == 4:
            out.append((float(r[0]), float(r[1]), float(r[2]), float(r[3])))
    return out


def _get_regions(env: Dict[str, Any]) -> Dict[str, Rect]:
    regs = env.get("regions", {}) or {}
    out: Dict[str, Rect] = {}
    for k, v in regs.items():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            out[str(k)] = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return out


def _point_in_rect(x: float, y: float, r: Rect) -> bool:
    x1, x2, y1, y2 = r
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def _path_hits_rect(X: np.ndarray, r: Rect) -> bool:
    if X is None or X.ndim != 2 or X.shape[0] == 0:
        return False
    for t in range(X.shape[0]):
        if _point_in_rect(float(X[t, 0]), float(X[t, 1]), r):
            return True
    return False


def _path_hits_any_rect(X: np.ndarray, rects: List[Rect]) -> bool:
    if X is None or X.ndim != 2 or X.shape[0] == 0:
        return False
    for r in rects:
        if _path_hits_rect(X, r):
            return True
    return False


def check_example5(
    *,
    X: np.ndarray,
    obstacles: List[Rect],
    doors: List[Rect],
    keys: List[Rect],
    G_rect: Rect,
) -> Tuple[bool, bool, bool, bool]:
    """
    Lightweight checker for visualization only.
    """
    avoid_ok = not _path_hits_any_rect(X, obstacles)
    reach_ok = _path_hits_rect(X, G_rect)
    keys_ok = all(_path_hits_rect(X, k) for k in keys)
    sat = bool(avoid_ok and reach_ok and keys_ok)
    return avoid_ok, reach_ok, keys_ok, sat


def _build_ex4_items(
    regions: Dict[str, Rect],
) -> Tuple[
    Optional[Rect],
    List[Tuple[Rect, str]],
    Dict[str, Any],
    Dict[str, str],
    bool,
    bool,
    bool,
    Optional[bool],
]:
    target_keys = sorted([k for k in regions.keys() if str(k).startswith("T_")])

    by_type: Dict[str, List[str]] = {}
    for k in target_keys:
        parts = str(k).split("_", 2)  # T, type, idx
        if len(parts) != 3:
            continue
        typ = str(parts[1])
        by_type.setdefault(typ, []).append(str(k))

    cmap = plt.get_cmap("tab20")
    edgecolor_by_name: Dict[str, Any] = {}
    linestyle_by_name: Dict[str, str] = {}

    type_names = sorted(by_type.keys())
    for i, typ in enumerate(type_names):
        c = cmap(i % cmap.N)
        for k in sorted(by_type[typ]):
            edgecolor_by_name[k] = c
            linestyle_by_name[k] = "--"

    extra_rects: List[Tuple[Rect, str]] = [(regions[k], str(k)) for k in target_keys]

    # Example 4 has no single "G" region for eval_vis, so keep it None.
    G_rect: Optional[Rect] = None

    # Visualization-only flags; ex4 checker not implemented here.
    sat = False
    avoid_ok = False
    reach_ok = False
    extra_ok = None

    return G_rect, extra_rects, edgecolor_by_name, linestyle_by_name, sat, avoid_ok, reach_ok, extra_ok


def _build_ex5_items(
    *,
    regions: Dict[str, Rect],
    obstacles: List[Rect],
    X: np.ndarray,
) -> Tuple[
    Rect,
    List[Tuple[Rect, str]],
    Dict[str, Any],
    Dict[str, str],
    bool,
    bool,
    bool,
    bool,
]:
    door_keys = sorted([k for k in regions.keys() if str(k).startswith("D")])
    key_keys = sorted([k for k in regions.keys() if str(k).startswith("K")])

    if "G" not in regions:
        raise RuntimeError("Example 5 requires regions['G'].")

    G_rect = regions["G"]
    n_pairs = min(len(door_keys), len(key_keys))

    doors = [regions[f"D{i}"] for i in range(n_pairs)]
    keys = [regions[f"K{i}"] for i in range(n_pairs)]

    avoid_ok, reach_ok, keys_ok, sat = check_example5(
        X=X,
        obstacles=obstacles,
        doors=doors,
        keys=keys,
        G_rect=G_rect,
    )

    cmap = plt.get_cmap("tab20")
    edgecolor_by_name: Dict[str, Any] = {}
    linestyle_by_name: Dict[str, str] = {}

    for i in range(n_pairs):
        Di = f"D{i}"
        Ki = f"K{i}"
        c = cmap(i % cmap.N)
        edgecolor_by_name[Di] = c
        edgecolor_by_name[Ki] = c
        linestyle_by_name[Di] = "-"
        linestyle_by_name[Ki] = "--"

    edgecolor_by_name["G"] = "tab:green"
    linestyle_by_name["G"] = "-."

    extra_rects: List[Tuple[Rect, str]] = (
        [(regions[k], str(k)) for k in sorted(regions.keys()) if str(k).startswith("D") or str(k).startswith("K")]
        + [(regions["G"], "G")]
    )

    return G_rect, extra_rects, edgecolor_by_name, linestyle_by_name, sat, avoid_ok, reach_ok, keys_ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to *.jsonl.gz")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save PNGs")
    p.add_argument("--example_id", type=int, default=None, help="Override example id (4 or 5). If omitted, infer from record.")
    p.add_argument("--n", type=int, default=50, help="Number of records to visualize")
    p.add_argument("--skip", type=int, default=0, help="Skip first K records")
    args = p.parse_args()

    visualize_qp_result = _import_eval_vis()
    os.makedirs(args.out_dir, exist_ok=True)

    k_saved = 0
    for idx, rec in enumerate(_read_jsonl_gz(args.data)):
        if idx < int(args.skip):
            continue
        if k_saved >= int(args.n):
            break

        sample = rec.get("sample", {}) or {}
        env = sample.get("env", {}) or {}
        micp = rec.get("micp", {}) or {}

        ex = int(args.example_id) if args.example_id is not None else _infer_example_id(rec)
        if ex is None:
            raise RuntimeError("Could not infer example_id from record. Pass --example_id explicitly.")
        if ex not in (4, 5):
            raise RuntimeError(f"viz_jsonl.py supports example_id 4 or 5 only. Got {ex}.")

        obstacles = _get_obstacles(env)
        regions = _get_regions(env)
        X = _as_np_array(micp.get("X", None))
        x0_env = _as_np_array(env.get("x0", None))

        if X is None or X.ndim != 2 or X.shape[0] < 1:
            continue

        if ex == 4:
            (
                G_rect,
                extra_rects,
                edgecolor_by_name,
                linestyle_by_name,
                sat,
                avoid_ok,
                reach_ok,
                extra_ok,
            ) = _build_ex4_items(regions)
        else:
            (
                G_rect,
                extra_rects,
                edgecolor_by_name,
                linestyle_by_name,
                sat,
                avoid_ok,
                reach_ok,
                extra_ok,
            ) = _build_ex5_items(
                regions=regions,
                obstacles=obstacles,
                X=X,
            )

        out_path = os.path.join(args.out_dir, f"viz_{idx:04d}.png")
        visualize_qp_result(
            out_path=out_path,
            X=X,
            x0_env=x0_env,
            obstacles=obstacles,
            G_rect=G_rect,
            extra_rects=extra_rects,
            sat=bool(sat),
            avoid_ok=bool(avoid_ok),
            reach_ok=bool(reach_ok),
            extra_ok=(None if extra_ok is None else bool(extra_ok)),
            edgecolor_by_name=edgecolor_by_name,
            linestyle_by_name=linestyle_by_name,
        )

        k_saved += 1

    print(f"[OK] Wrote {k_saved} PNGs to: {args.out_dir}")


if __name__ == "__main__":
    main()