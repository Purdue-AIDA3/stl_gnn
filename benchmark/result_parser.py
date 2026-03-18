from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SOLVER_ORDER = ["gnn", "cnn", "mlp", "stlpy"]
LEARNED_SOLVERS = ["gnn", "cnn", "mlp"]
FOLDER_RE = re.compile(r"^(?:(stlpy|cnn|mlp)_)?eval_ex(\d+)_obs(\d+)$")


@dataclass
class ResultRecord:
    solver: str
    example_id: int
    obstacles: int
    n_cases: Optional[int]

    success_rate: Optional[float]
    pure_success_rate: Optional[float]
    fallback_rate: Optional[float]

    n_success: Optional[int]
    n_pure_success: Optional[int]
    fallback_count: Optional[int]

    time_total_mean: Optional[float]
    time_total_std: Optional[float]
    time_qp_mean: Optional[float]
    time_qp_std: Optional[float]

    solved_k_mean: Optional[float]
    solved_k_std: Optional[float]

    cost_mean: Optional[float]
    cost_std: Optional[float]

    avg_time: Optional[float]
    median_time: Optional[float]
    avg_rho: Optional[float]

    seed: Optional[int]
    source_directory: str
    summary_json: str
    raw_summary: Dict[str, Any]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def _normalize_solver(solver_from_dir: Optional[str], solver_from_json: Any) -> str:
    if isinstance(solver_from_json, str) and solver_from_json.strip():
        return solver_from_json.strip().lower()
    if solver_from_dir is None:
        raise ValueError("Could not infer solver from folder name or summary.json")
    return solver_from_dir


def parse_result_dir_name(dirname: str) -> Tuple[str, int, int]:
    m = FOLDER_RE.match(dirname)
    if not m:
        raise ValueError(f"Directory name does not match benchmark convention: {dirname}")
    prefix, example_id, obstacles = m.groups()
    solver = prefix if prefix is not None else "gnn"
    return solver, int(example_id), int(obstacles)


def _fallback_count(summary: Dict[str, Any]) -> Optional[int]:
    return _safe_int(summary.get("fallback_count"))


def _fallback_rate(summary: Dict[str, Any], n_cases: Optional[int]) -> Optional[float]:
    direct = _safe_float(summary.get("fallback_rate"))
    if direct is not None:
        return _clip01(direct)

    fallback_count = _fallback_count(summary)
    if fallback_count is not None and n_cases and n_cases > 0:
        return _clip01(float(fallback_count) / float(n_cases))
    return None


def _pure_success_rate(
    solver: str,
    success_rate: Optional[float],
    fallback_rate: Optional[float],
) -> Optional[float]:
    if solver not in LEARNED_SOLVERS:
        return None
    if success_rate is None:
        return None
    fb = 0.0 if fallback_rate is None else float(fallback_rate)
    return _clip01(float(success_rate) - fb)


def load_result_record(summary_json_path: Path) -> ResultRecord:
    summary_json_path = Path(summary_json_path)
    result_dir = summary_json_path.parent
    solver_dir, ex_dir, obs_dir = parse_result_dir_name(result_dir.name)

    with summary_json_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    solver = _normalize_solver(solver_dir, summary.get("solver"))
    example_id = _safe_int(summary.get("example_id"))
    if example_id is None:
        example_id = ex_dir

    obstacles = _safe_int(summary.get("obstacles"))
    if obstacles is None:
        obstacles = obs_dir

    n_cases = _safe_int(summary.get("n_cases"))
    success_rate = _clip01(_safe_float(summary.get("success_rate")))
    fallback_rate = _fallback_rate(summary, n_cases)
    pure_success_rate = _pure_success_rate(solver, success_rate, fallback_rate)

    n_success = _safe_int(summary.get("n_success"))
    if n_success is None and n_cases is not None and success_rate is not None:
        n_success = int(round(success_rate * n_cases))

    fallback_count = _fallback_count(summary)
    if fallback_count is None and n_cases is not None and fallback_rate is not None:
        fallback_count = int(round(fallback_rate * n_cases))

    n_pure_success: Optional[int] = None
    if solver in LEARNED_SOLVERS and n_cases is not None and pure_success_rate is not None:
        n_pure_success = int(round(pure_success_rate * n_cases))

    return ResultRecord(
        solver=solver,
        example_id=int(example_id),
        obstacles=int(obstacles),
        n_cases=n_cases,
        success_rate=success_rate,
        pure_success_rate=pure_success_rate,
        fallback_rate=fallback_rate,
        n_success=n_success,
        n_pure_success=n_pure_success,
        fallback_count=fallback_count,
        time_total_mean=_safe_float(summary.get("time_total_mean", summary.get("avg_time"))),
        time_total_std=_safe_float(summary.get("time_total_std")),
        time_qp_mean=_safe_float(summary.get("time_qp_mean", summary.get("avg_time"))),
        time_qp_std=_safe_float(summary.get("time_qp_std")),
        solved_k_mean=_safe_float(summary.get("solved_k_mean")),
        solved_k_std=_safe_float(summary.get("solved_k_std")),
        cost_mean=_safe_float(summary.get("cost_mean")),
        cost_std=_safe_float(summary.get("cost_std")),
        avg_time=_safe_float(summary.get("avg_time")),
        median_time=_safe_float(summary.get("median_time")),
        avg_rho=_safe_float(summary.get("avg_rho")),
        seed=_safe_int(summary.get("seed")),
        source_directory=str(result_dir),
        summary_json=str(summary_json_path),
        raw_summary=summary,
    )


def discover_result_records(root_dir: str | Path) -> List[ResultRecord]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root}")

    records: List[ResultRecord] = []
    for summary_json in sorted(root.rglob("summary.json")):
        try:
            parse_result_dir_name(summary_json.parent.name)
        except ValueError:
            continue
        try:
            records.append(load_result_record(summary_json))
        except Exception as e:
            print(f"[WARN] Failed to parse {summary_json}: {type(e).__name__}: {e}")
    return records


def records_to_dicts(records: Iterable[ResultRecord]) -> List[Dict[str, Any]]:
    return [asdict(rec) for rec in records]


def _fmt_float(v: Optional[float], digits: int = 4) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _print_summary(records: List[ResultRecord]) -> None:
    print(f"Parsed {len(records)} result records.")
    if not records:
        return

    by_solver: Dict[str, int] = {k: 0 for k in SOLVER_ORDER}
    by_example: Dict[int, int] = {}
    obstacles_seen: Dict[int, List[int]] = {}
    for r in records:
        by_solver[r.solver] = by_solver.get(r.solver, 0) + 1
        by_example[r.example_id] = by_example.get(r.example_id, 0) + 1
        obstacles_seen.setdefault(r.example_id, []).append(r.obstacles)

    print("Records by solver:")
    for solver in SOLVER_ORDER:
        print(f"  {solver:5s}: {by_solver.get(solver, 0)}")

    print("Records by example:")
    for ex in sorted(by_example):
        uniq_obs = sorted(set(obstacles_seen.get(ex, [])))
        print(f"  ex{ex}: {by_example[ex]} records  obstacles={uniq_obs}")

    print("Sample rows:")
    for rec in records[: min(8, len(records))]:
        print(
            f"  {rec.solver:5s} ex{rec.example_id} obs{rec.obstacles}  "
            f"success={_fmt_float(rec.success_rate, 3)}  "
            f"pure={_fmt_float(rec.pure_success_rate, 3)}  "
            f"fallback={_fmt_float(rec.fallback_rate, 3)}  "
            f"time={_fmt_float(rec.time_total_mean, 4)}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Discover and normalize benchmark summary.json files.")
    p.add_argument("--root_dir", type=str, default="results")
    p.add_argument("--save_json", type=str, default="", help="Optional path to save normalized records as JSON.")
    args = p.parse_args()

    records = discover_result_records(args.root_dir)
    _print_summary(records)

    if str(args.save_json).strip():
        out_path = Path(args.save_json)
    else:
        out_path = Path(args.root_dir) / "benchmark" / "parsed_records.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records_to_dicts(records), f, indent=2)
    print(f"Saved normalized records to: {out_path}")


if __name__ == "__main__":
    main()