from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .result_parser import ResultRecord, SOLVER_ORDER, LEARNED_SOLVERS, discover_result_records

SWEEP_CASES = (1, 2)
FIXED_COUNT_CASES = (3, 4, 5)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(v: Optional[float], digits: int = 4) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _pad_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "  ".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def _group_single(records: List[ResultRecord]) -> Dict[Tuple[int, int, str], ResultRecord]:
    out: Dict[Tuple[int, int, str], ResultRecord] = {}
    for r in records:
        out[(r.example_id, r.obstacles, r.solver)] = r
    return out


def _speedup_values_for_solver(records: List[ResultRecord], ex: int, solver: str) -> List[float]:
    grouped = _group_single(records)
    obs_values = sorted({r.obstacles for r in records if r.example_id == ex})
    vals: List[float] = []
    for obs in obs_values:
        base = grouped.get((ex, obs, "stlpy"))
        cur = grouped.get((ex, obs, solver))
        if base is None or cur is None:
            continue
        if base.time_total_mean is None or cur.time_total_mean is None:
            continue
        if cur.time_total_mean <= 0:
            continue
        vals.append(float(base.time_total_mean) / float(cur.time_total_mean))
    return vals


def _aggregate_case_solver(records: List[ResultRecord]) -> List[List[str]]:
    grouped: Dict[Tuple[int, str], List[ResultRecord]] = defaultdict(list)
    for r in records:
        grouped[(r.example_id, r.solver)].append(r)

    rows: List[List[str]] = []
    for ex in sorted({r.example_id for r in records}):
        for solver in SOLVER_ORDER:
            rs = sorted(grouped.get((ex, solver), []), key=lambda x: x.obstacles)
            if not rs:
                continue

            n_obs_list = ",".join(str(r.obstacles) for r in rs)
            speedups = _speedup_values_for_solver(records, ex, solver) if solver in LEARNED_SOLVERS else []

            rows.append(
                [
                    f"Case {ex}",
                    solver,
                    n_obs_list,
                    str(len(rs)),
                    _fmt(_mean(r.pure_success_rate for r in rs), 3) if solver in LEARNED_SOLVERS else "-",
                    _fmt(_mean(r.success_rate for r in rs), 3),
                    _fmt(_mean(r.fallback_rate for r in rs), 3) if solver in LEARNED_SOLVERS else "-",
                    _fmt(_mean(r.solved_k_mean for r in rs), 2) if solver in LEARNED_SOLVERS else "-",
                    _fmt(_mean(speedups), 2) if solver in LEARNED_SOLVERS else "-",
                    _fmt(_mean(r.time_total_mean for r in rs), 4),
                    _fmt(_mean(r.time_qp_mean for r in rs), 4),
                    _fmt(_mean(r.cost_mean for r in rs), 4),
                    _fmt(_mean(r.avg_rho for r in rs), 4),
                    _fmt(_mean(r.n_cases for r in rs), 1),
                ]
            )
    return rows


def _obstacle_detail_rows(records: List[ResultRecord]) -> List[List[str]]:
    grouped: Dict[Tuple[int, int, str], ResultRecord] = {}
    for r in records:
        grouped[(r.example_id, r.obstacles, r.solver)] = r

    rows: List[List[str]] = []
    for ex in sorted({r.example_id for r in records}):
        obstacle_values = sorted({r.obstacles for r in records if r.example_id == ex})
        for obs in obstacle_values:
            for solver in SOLVER_ORDER:
                r = grouped.get((ex, obs, solver))
                if r is None:
                    rows.append(
                        [f"Case {ex}", str(obs), solver, "MISSING", "-", "-", "-", "-", "-", "-", "-", "-"]
                    )
                else:
                    speedup_x = "-"
                    if solver in LEARNED_SOLVERS:
                        base = grouped.get((ex, obs, "stlpy"))
                        if (
                            base is not None
                            and base.time_total_mean is not None
                            and r.time_total_mean is not None
                            and r.time_total_mean > 0
                        ):
                            speedup_x = _fmt(float(base.time_total_mean) / float(r.time_total_mean), 2)

                    rows.append(
                        [
                            f"Case {ex}",
                            str(obs),
                            solver,
                            "sweep" if ex in SWEEP_CASES else "fixed-count",
                            _fmt(r.pure_success_rate, 3) if solver in LEARNED_SOLVERS else "-",
                            _fmt(r.success_rate, 3),
                            _fmt(r.fallback_rate, 3) if solver in LEARNED_SOLVERS else "-",
                            _fmt(r.solved_k_mean, 2) if solver in LEARNED_SOLVERS else "-",
                            speedup_x,
                            _fmt(r.time_total_mean, 4),
                            _fmt(r.time_qp_mean, 4),
                            _fmt(r.cost_mean, 4),
                        ]
                    )
    return rows


def write_tables(records: List[ResultRecord], out_dir: str | Path) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers1 = [
        "case",
        "solver",
        "n_obs",
        "runs",
        "pure_success",
        "final_success",
        "fallback",
        "solved_k",
        "speedup_x",
        "time_total",
        "time_qp",
        "cost",
        "avg_rho",
        "n_cases",
    ]
    body1 = _aggregate_case_solver(records)

    text1 = []
    text1.append("Benchmark summary by case and solver")
    text1.append("")
    text1.append("pure_success = success without fallback. final_success = reported success including fallback recovery.")
    text1.append("speedup_x = STLPY end-to-end solve time divided by the learned solver end-to-end solve time.")
    text1.append("For Cases 1 and 2, metrics are averaged across the obstacle-count sweep.")
    text1.append("For Cases 3, 4, and 5, each row reflects the fixed obstacle-count setting with resampled obstacle geometries.")
    text1.append("")
    text1.append(_pad_table(headers1, body1))

    path1 = out_dir / "example_summary.txt"
    path1.write_text("\n".join(text1) + "\n", encoding="utf-8")

    headers2 = [
        "case",
        "n_obs",
        "solver",
        "setting",
        "pure_success",
        "final_success",
        "fallback",
        "solved_k",
        "speedup_x",
        "time_total",
        "time_qp",
        "cost",
    ]
    body2 = _obstacle_detail_rows(records)

    text2 = []
    text2.append("Benchmark summary by obstacle-count setting")
    text2.append("")
    text2.append("Cases 1 and 2 are obstacle-count sweeps. Cases 3, 4, and 5 use fixed obstacle counts with resampled obstacle geometries.")
    text2.append("pure_success is the primary metric for learned solvers because fallback is designed to recover feasibility.")
    text2.append("")
    text2.append(_pad_table(headers2, body2))

    path2 = out_dir / "obstacle_sweep_summary.txt"
    path2.write_text("\n".join(text2) + "\n", encoding="utf-8")

    return path1, path2


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper-ready text tables from benchmark summaries.")
    p.add_argument("--root_dir", type=str, default="results")
    args = p.parse_args()

    records = discover_result_records(args.root_dir)
    out_dir = Path(args.root_dir) / "benchmark" / "tables"
    p1, p2 = write_tables(records, out_dir)
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()