# gnn_train/eval_diag.py
from __future__ import annotations

import os
from typing import Dict, Optional, TextIO

import numpy as np
from gurobipy import GRB


class DiagLogger:
    """
    Simple line-based logger that overwrites the output file each run.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.f: TextIO = open(self.path, "w", encoding="utf-8")

    def write(self, msg: str) -> None:
        self.f.write(str(msg))

    def writeln(self, msg: str = "") -> None:
        self.f.write(str(msg) + "\n")

    def flush(self) -> None:
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.flush()
        finally:
            self.f.close()


def status_name(status: int) -> str:
    """
    Convert a Gurobi status code to a readable name.
    """
    mp = {
        int(GRB.OPTIMAL): "OPTIMAL",
        int(GRB.INFEASIBLE): "INFEASIBLE",
        int(GRB.INF_OR_UNBD): "INF_OR_UNBD",
        int(GRB.UNBOUNDED): "UNBOUNDED",
        int(GRB.TIME_LIMIT): "TIME_LIMIT",
        int(GRB.INTERRUPTED): "INTERRUPTED",
        int(GRB.SUBOPTIMAL): "SUBOPTIMAL",
    }
    return mp.get(int(status), f"STATUS_{int(status)}")


def compute_face_switch_stats(
    faces_by_pred: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, int]]:
    """
    Compute per-predicate switch statistics for face sequences.

    Args:
        faces_by_pred: pred -> faces array of shape (T+1,), dtype int.

    Returns:
        pred -> {"T": int, "switches": int, "n_segments": int}
    """
    out: Dict[str, Dict[str, int]] = {}
    if not faces_by_pred:
        return out

    for pred, faces in faces_by_pred.items():
        if not isinstance(faces, np.ndarray) or faces.ndim != 1 or faces.shape[0] < 2:
            continue

        switches = 0
        for t in range(1, int(faces.shape[0])):
            switches += int(faces[t] != faces[t - 1])

        out[str(pred)] = {
            "T": int(faces.shape[0] - 1),
            "switches": int(switches),
            "n_segments": int(switches + 1),
        }

    return out