# micp.py
"""
Data generation module for neurosymbolic path planning.

Implements:
- STL -> strategic binaries (STLBinaryExtractor)
- MICP (mixed-integer convex programming) construction for Gurobi.
- Strategy extraction (save strategic binaries)
- Dataset generation for GNN training (jsonl.gz)

Current project requirement (patched):
- Ground-truth supervision is the raw strategic binaries `strategy.binaries`
  keyed by BinaryDescriptor.id for ALL descriptors extracted from the AST.
- The dataset must store:
  (1) raw strategic binaries (id -> 0/1)  [authoritative GT for training]
  (2) binary descriptors (id semantics: role/node_tag/time/meta)
  (3) AST (dict) and sample env
  (4) optional solution trajectory (X,U) for analysis/debugging

PATCH NOTES (graph correctness):
- ast_to_dict now includes a deterministic per-node uid:
    uid = f"{node.key()}@{path}"  where root path is "0"
  This matches stl._uid_for / stl.compute_uid_maps and enables unambiguous
  operator-instance attachment in graph_builder without overwrite risk.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from geometry import Rect, rect_in_constraints, rect_outside_bigM
from dynamics import DoubleIntegrator2D
from stl import (
    STLNode,
    InRect,
    OutsideRect,
    And,
    Or,
    Eventually,
    Always,
    Until,
    BinaryDescriptor,
    STLBinaryExtractor,
    STLStrategy,
)


# ============================================================
# 1) Planning problem definition
# ============================================================

@dataclass
class PlanningProblem:
    T: int
    dynamics: DoubleIntegrator2D
    x0: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    bigM: float = 1e3


# ============================================================
# 2) Strategy extraction (ALL strategic binaries)
# ============================================================

class StrategyExtractor:
    """
    Extract ALL binary variables corresponding to ALL BinaryDescriptors.

    Rationale:
      - Training will supervise all binary nodes using BCE.
      - Exact key match: BinaryDescriptor.id <-> strategy.binaries key
    """

    @staticmethod
    def from_model(
        T: int,
        descs: List[BinaryDescriptor],
        bin_vars: Dict[str, gp.Var],
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> STLStrategy:
        binaries: Dict[str, int] = {}
        for d in descs:
            if d.id not in bin_vars:
                raise RuntimeError(f"StrategyExtractor: missing binary var for id={d.id}")
            binaries[d.id] = int(round(bin_vars[d.id].X))
        m = dict(meta) if isinstance(meta, dict) else {}
        m.setdefault("n_descs_total", int(len(descs)))
        return STLStrategy(T=int(T), binaries=binaries, meta=m)


# ============================================================
# 2.5) Descriptor serialization (dataset logging)
# ============================================================

def _descs_to_dicts(descs: List[BinaryDescriptor]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in descs:
        out.append(
            {
                "id": str(d.id),
                "role": str(d.role),
                "node_tag": str(d.node_tag),
                "time": (None if d.time is None else int(d.time)),
                "meta": dict(d.meta),
            }
        )
    return out


# ============================================================
# 3) MICP builder
# ============================================================

class MICPBuilder:
    def __init__(self, problem: PlanningProblem, ast: STLNode):
        self.p = problem
        self.ast = ast
        self.model: Optional[gp.Model] = None
        self.x = None
        self.u = None
        self.bin_vars: Dict[str, gp.Var] = {}

        self._outside_face_id: Dict[Tuple[str, int, int], str] = {}
        self._event_time_id: Dict[Tuple[str, int, Optional[int]], str] = {}
        self._always_or_id: Dict[Tuple[str, int, int], str] = {}

    def build(self, descs: List[BinaryDescriptor], *, output_flag: int = 0) -> gp.Model:
        T = int(self.p.T)
        A, B = self.p.dynamics.matrices()

        m = gp.Model("stl_micp")
        m.Params.OutputFlag = int(output_flag)
        m.Params.Threads = 16    # small integers for preventing segmentation faults (observed empirically)
        self.model = m

        self.x = m.addVars(T + 1, 4, lb=-GRB.INFINITY, name="x")
        self.u = m.addVars(T, 2, lb=-GRB.INFINITY, name="u")

        for k in range(T + 1):
            for i in range(4):
                m.addConstr(self.x[k, i] >= float(self.p.x_min[i]), name=f"xmin[{k},{i}]")
                m.addConstr(self.x[k, i] <= float(self.p.x_max[i]), name=f"xmax[{k},{i}]")
        for k in range(T):
            for i in range(2):
                m.addConstr(self.u[k, i] >= float(self.p.u_min[i]), name=f"umin[{k},{i}]")
                m.addConstr(self.u[k, i] <= float(self.p.u_max[i]), name=f"umax[{k},{i}]")

        x0 = np.asarray(self.p.x0, dtype=float).reshape(-1)
        if x0.shape[0] != 4:
            raise ValueError("x0 must have shape (4,)")
        for i in range(4):
            m.addConstr(self.x[0, i] == float(x0[i]), name=f"x0[{i}]")

        for k in range(T):
            for i in range(4):
                m.addConstr(
                    self.x[k + 1, i]
                    == gp.quicksum(float(A[i, j]) * self.x[k, j] for j in range(4))
                    + gp.quicksum(float(B[i, j]) * self.u[k, j] for j in range(2)),
                    name=f"dyn[{k},{i}]",
                )

        # Create ALL binary variables for ALL descriptors.
        for bd in descs:
            if bd.id not in self.bin_vars:
                # name=... must be unique; sanitize ":" for Gurobi's name display only
                self.bin_vars[bd.id] = m.addVar(vtype=GRB.BINARY, name=bd.id.replace(":", "_"))
        m.update()

        self._outside_face_id = {}
        self._event_time_id = {}
        self._always_or_id = {}

        for d in descs:
            if d.role == "outside_face":
                pred = d.meta.get("pred", d.meta.get("rect"))
                if pred is None:
                    raise RuntimeError(f"outside_face descriptor missing pred/rect in meta: {d.meta}")

                t = int(d.meta["t"]) if "t" in d.meta else int(d.meta["k"])
                face = int(d.meta["face"]) if "face" in d.meta else int(d.meta["h"])

                key = (str(pred), t, face)
                self._outside_face_id[key] = d.id

            elif d.role == "event_time":
                pred = d.meta.get("pred")
                k = int(d.meta["k"])
                opt = d.meta.get("opt", None)
                opt = int(opt) if opt is not None else None
                key = (str(pred), k, opt)
                self._event_time_id[key] = d.id

            elif d.role == "always_or_choice":
                pred = d.meta.get("pred")
                k = int(d.meta["k"])
                opt = int(d.meta["opt"])
                key = (str(pred), k, opt)
                self._always_or_id[key] = d.id

        self._add_stl_constraints(self.ast)

        Q = 0.5 * (np.asarray(self.p.Q, dtype=float) + np.asarray(self.p.Q, dtype=float).T)
        R = 0.5 * (np.asarray(self.p.R, dtype=float) + np.asarray(self.p.R, dtype=float).T)

        obj = gp.QuadExpr()

        for k in range(T + 1):
            for i in range(4):
                xi = self.x[k, i]
                for j in range(4):
                    qij = float(Q[i, j])
                    if qij != 0.0:
                        obj += qij * xi * self.x[k, j]

        for k in range(T):
            for i in range(2):
                ui = self.u[k, i]
                for j in range(2):
                    rij = float(R[i, j])
                    if rij != 0.0:
                        obj += rij * ui * self.u[k, j]

        m.setObjective(obj, GRB.MINIMIZE)
        return m

    def _add_stl_constraints(self, node: STLNode) -> None:
        if isinstance(node, And):
            for c in node.args:
                self._add_stl_constraints(c)
            return

        if isinstance(node, Always):
            self._add_always(node.a, node.b, node.phi)
            return

        if isinstance(node, Eventually):
            self._add_eventually(node.a, node.b, node.phi)
            return

        if isinstance(node, Until):
            self._add_until(node.a, node.b, node.phi, node.psi)
            return

        raise NotImplementedError("Top-level predicate must be under temporal operator in this pipeline.")

    def _add_always(self, a: int, b: int, phi: STLNode) -> None:
        assert self.model is not None

        if isinstance(phi, OutsideRect):
            rect = phi.rect
            name = phi.name
            for k in range(a, b + 1):
                face = []
                for h in range(4):
                    bid = self._outside_face_id.get((name, k, h))
                    if bid is None:
                        raise RuntimeError(f"Missing outside_face binary for (rect={name}, time={k}, h={h})")
                    face.append(self.bin_vars[bid])
                rect_outside_bigM(
                    self.model,
                    self.x[k, 0],
                    self.x[k, 1],
                    rect,
                    face,
                    prefix=f"always_out_{name}[{k}]",
                    M=self.p.bigM,
                    rhs=1.0,
                )
            return

        if isinstance(phi, Or) and all(isinstance(arg, InRect) for arg in phi.args):
            rects = [arg.rect for arg in phi.args]
            names = [arg.name for arg in phi.args]
            for k in range(a, b + 1):
                sel = []
                for i, nm in enumerate(names):
                    bid = self._always_or_id.get((nm, k, i))
                    if bid is None:
                        raise RuntimeError(f"Missing always_or_choice binary for (pred={nm}, time={k}, opt={i})")
                    sel.append(self.bin_vars[bid])

                self.model.addConstr(gp.quicksum(sel) >= 1, name=f"always_or_atleast1[k{k}]")

                for i, rect in enumerate(rects):
                    xmin, xmax, ymin, ymax = rect
                    self.model.addGenConstrIndicator(sel[i], True, self.x[k, 0] >= xmin, name=f"sel_in_{names[i]}[{k}]_xmin")
                    self.model.addGenConstrIndicator(sel[i], True, self.x[k, 0] <= xmax, name=f"sel_in_{names[i]}[{k}]_xmax")
                    self.model.addGenConstrIndicator(sel[i], True, self.x[k, 1] >= ymin, name=f"sel_in_{names[i]}[{k}]_ymin")
                    self.model.addGenConstrIndicator(sel[i], True, self.x[k, 1] <= ymax, name=f"sel_in_{names[i]}[{k}]_ymax")
            return

        if isinstance(phi, And):
            for c in phi.args:
                self._add_always(a, b, c)
            return

        raise NotImplementedError("Always supports OutsideRect, Or(InRect,...), or And of those.")

    def _add_eventually(self, a: int, b: int, phi: STLNode) -> None:
        assert self.model is not None

        if isinstance(phi, Or) and all(isinstance(arg, Always) and isinstance(arg.phi, InRect) for arg in phi.args):
            alw_args = list(phi.args)
            a2 = int(alw_args[0].a)
            b2 = int(alw_args[0].b)
            for arg in alw_args:
                if int(arg.a) != a2 or int(arg.b) != b2:
                    raise NotImplementedError(
                        "Eventually(Or(Always(...))) requires identical Always intervals across options."
                    )

            bvars = []
            for k in range(a, b + 1):
                for i, alw in enumerate(alw_args):
                    pred = alw.phi
                    nm = pred.name
                    rect = pred.rect

                    bid = self._event_time_id.get((nm, k, i))
                    if bid is None:
                        raise RuntimeError(f"Missing event_time binary for (pred={nm}, time={k}, opt={i})")
                    bk = self.bin_vars[bid]
                    bvars.append(bk)

                    xmin, xmax, ymin, ymax = rect
                    t_lo = k + a2
                    t_hi = k + b2
                    if t_lo < 0 or t_hi > self.p.T:
                        raise ValueError(
                            f"Dwell window out of bounds: k={k}, [a2,b2]=[{a2},{b2}], T={self.p.T}"
                        )

                    for t in range(t_lo, t_hi + 1):
                        self.model.addGenConstrIndicator(
                            bk, True, self.x[t, 0] >= xmin,
                            name=f"evt_dwell_{nm}[k{k},t{t}]_xmin"
                        )
                        self.model.addGenConstrIndicator(
                            bk, True, self.x[t, 0] <= xmax,
                            name=f"evt_dwell_{nm}[k{k},t{t}]_xmax"
                        )
                        self.model.addGenConstrIndicator(
                            bk, True, self.x[t, 1] >= ymin,
                            name=f"evt_dwell_{nm}[k{k},t{t}]_ymin"
                        )
                        self.model.addGenConstrIndicator(
                            bk, True, self.x[t, 1] <= ymax,
                            name=f"evt_dwell_{nm}[k{k},t{t}]_ymax"
                        )

            self.model.addConstr(gp.quicksum(bvars) == 1, name="evt_select_or_always")
            return

        if isinstance(phi, InRect):
            rect = phi.rect
            nm = phi.name
            bvars = []
            for k in range(a, b + 1):
                bid = self._event_time_id.get((nm, k, None))
                if bid is None:
                    raise RuntimeError(f"Missing event_time binary for (pred={nm}, time={k})")
                bk = self.bin_vars[bid]
                bvars.append(bk)

                xmin, xmax, ymin, ymax = rect
                self.model.addGenConstrIndicator(bk, True, self.x[k, 0] >= xmin, name=f"evt_in_{nm}[{k}]_xmin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 0] <= xmax, name=f"evt_in_{nm}[{k}]_xmax")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 1] >= ymin, name=f"evt_in_{nm}[{k}]_ymin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 1] <= ymax, name=f"evt_in_{nm}[{k}]_ymax")

            self.model.addConstr(gp.quicksum(bvars) == 1, name=f"evt_select_{nm}")
            return

        if isinstance(phi, Or) and all(isinstance(arg, InRect) for arg in phi.args):
            rects = [arg.rect for arg in phi.args]
            names = [arg.name for arg in phi.args]
            bvars = []
            for k in range(a, b + 1):
                for i, nm in enumerate(names):
                    bid = self._event_time_id.get((nm, k, i))
                    if bid is None:
                        raise RuntimeError(f"Missing event_time binary for (pred={nm}, time={k}, opt={i})")
                    bk = self.bin_vars[bid]
                    bvars.append(bk)

                    xmin, xmax, ymin, ymax = rects[i]
                    self.model.addGenConstrIndicator(bk, True, self.x[k, 0] >= xmin, name=f"evt_or_in_{nm}[k{k}]_xmin")
                    self.model.addGenConstrIndicator(bk, True, self.x[k, 0] <= xmax, name=f"evt_or_in_{nm}[k{k}]_xmax")
                    self.model.addGenConstrIndicator(bk, True, self.x[k, 1] >= ymin, name=f"evt_or_in_{nm}[k{k}]_ymin")
                    self.model.addGenConstrIndicator(bk, True, self.x[k, 1] <= ymax, name=f"evt_or_in_{nm}[k{k}]_ymax")

            self.model.addConstr(gp.quicksum(bvars) == 1, name="evt_select_or_inrect")
            return

        raise NotImplementedError("Eventually supports InRect, Or(InRect,...), or Or(Always(InRect),...).")

    def _add_until(self, a: int, b: int, phi: STLNode, psi: STLNode) -> None:
        assert self.model is not None

        if not isinstance(phi, OutsideRect):
            raise NotImplementedError("Until left side must be OutsideRect(D) in this project.")
        if not isinstance(psi, InRect):
            raise NotImplementedError("Until right side must be InRect(K) in this project.")

        d_rect = phi.rect
        d_name = phi.name
        k_rect = psi.rect
        k_name = psi.name

        # ------------------------------------------------------------
        # 1) Trigger time selection: choose exactly one t in [a,b]
        #    such that x[t] ∈ K (psi).
        # ------------------------------------------------------------
        b_by_t: Dict[int, gp.Var] = {}
        for t in range(a, b + 1):
            bid = self._event_time_id.get((k_name, t, None))
            if bid is None:
                raise RuntimeError(f"Missing event_time binary for (pred={k_name}, time={t})")
            bt = self.bin_vars[bid]
            b_by_t[int(t)] = bt

            xmin, xmax, ymin, ymax = k_rect
            self.model.addGenConstrIndicator(bt, True, self.x[t, 0] >= xmin, name=f"until_goal_{k_name}[{t}]_xmin")
            self.model.addGenConstrIndicator(bt, True, self.x[t, 0] <= xmax, name=f"until_goal_{k_name}[{t}]_xmax")
            self.model.addGenConstrIndicator(bt, True, self.x[t, 1] >= ymin, name=f"until_goal_{k_name}[{t}]_ymin")
            self.model.addGenConstrIndicator(bt, True, self.x[t, 1] <= ymax, name=f"until_goal_{k_name}[{t}]_ymax")

        self.model.addConstr(gp.quicksum(b_by_t.values()) == 1, name=f"until_select_trigger[{k_name}]")

        # ------------------------------------------------------------
        # 2) Safety BEFORE trigger: enforce x[t] outside D only for t < t*
        #
        #    Let bt indicate trigger time t*.
        #    Define pre_t = sum_{tau=t+1..b} b_tau.
        #
        #    If trigger is at t*:
        #      - for t < t*: pre_t = 1  -> enforce outside(D) at time t
        #      - for t >= t*: pre_t = 0 -> no constraint (door may be crossed)
        #
        #    This matches the intended "¬D U K" behavior in this project.
        # ------------------------------------------------------------
        for t in range(0, self.p.T + 1):
            face: List[gp.Var] = []
            for h in range(4):
                bid = self._outside_face_id.get((d_name, t, h))
                if bid is None:
                    raise RuntimeError(f"Missing outside_face binary for (rect={d_name}, time={t}, h={h})")
                face.append(self.bin_vars[bid])

            # pre_t is 1 iff the chosen trigger time tau satisfies tau > t (i.e., t is strictly before trigger).
            if t < a:
                pre_t = 1.0
            elif t >= b:
                pre_t = 0.0
            else:
                pre_t = gp.quicksum(b_by_t[tau] for tau in range(t + 1, b + 1))

            rect_outside_bigM(
                self.model,
                self.x[t, 0],
                self.x[t, 1],
                d_rect,
                face,
                prefix=f"until_safe_{d_name}[{t}]",
                M=self.p.bigM,
                rhs=pre_t,
            )


def solve_micp_with_strategy(ast: STLNode, prob: PlanningProblem, *, output_flag: int = 0) -> Tuple[np.ndarray, np.ndarray, STLStrategy, float]:
    T = int(prob.T)

    descs = STLBinaryExtractor(T).extract(ast)

    builder = MICPBuilder(prob, ast)
    model = builder.build(descs, output_flag=output_flag)

    t0 = time.time()
    model.optimize()
    t_sec = time.time() - t0

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status={model.Status}, not optimal.")

    X = np.zeros((T + 1, 4), dtype=float)
    U = np.zeros((T, 2), dtype=float)

    for k in range(T + 1):
        for i in range(4):
            X[k, i] = float(builder.x[k, i].X)
    for k in range(T):
        for i in range(2):
            U[k, i] = float(builder.u[k, i].X)

    strategy = StrategyExtractor.from_model(T, descs, builder.bin_vars, meta={"gurobi_status": int(model.Status)})
    return X, U, strategy, float(t_sec)


# ============================================================
# 5) AST serialization
# ============================================================

def ast_to_dict(node: STLNode, *, _path: str = "0") -> Dict[str, Any]:
    """Serialize STL AST to a JSON-friendly dict, including deterministic per-node uid.

    uid scheme matches stl.compute_uid_maps / stl._uid_for:
      uid = f"{node.key()}@{path}"
    where `path` is a structural path (root "0").
    """
    uid = f"{node.key()}@{_path}"

    if isinstance(node, InRect):
        return {"type": "InRect", "uid": uid, "name": node.name, "rect": list(node.rect)}
    if isinstance(node, OutsideRect):
        return {"type": "OutsideRect", "uid": uid, "name": node.name, "rect": list(node.rect)}
    if isinstance(node, And):
        return {"type": "And", "uid": uid, "args": [ast_to_dict(a, _path=f"{_path}/{i}") for i, a in enumerate(node.args)]}
    if isinstance(node, Or):
        return {"type": "Or", "uid": uid, "args": [ast_to_dict(a, _path=f"{_path}/{i}") for i, a in enumerate(node.args)]}
    if isinstance(node, Eventually):
        return {"type": "Eventually", "uid": uid, "a": int(node.a), "b": int(node.b), "phi": ast_to_dict(node.phi, _path=f"{_path}/0")}
    if isinstance(node, Always):
        return {"type": "Always", "uid": uid, "a": int(node.a), "b": int(node.b), "phi": ast_to_dict(node.phi, _path=f"{_path}/0")}
    if isinstance(node, Until):
        return {
            "type": "Until",
            "uid": uid,
            "a": int(node.a),
            "b": int(node.b),
            "phi": ast_to_dict(node.phi, _path=f"{_path}/0"),
            "psi": ast_to_dict(node.psi, _path=f"{_path}/1"),
        }
    raise NotImplementedError(f"ast_to_dict: unsupported node type {type(node)}")


# ============================================================
# 6) Dataset writer (jsonl.gz)
# ============================================================

def write_jsonl_gz(path: str, records: List[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _infer_T(sample: Dict[str, Any], X: Any) -> int:
    if "T" in sample:
        return int(sample["T"])
    return int(len(X) - 1)


def generate_dataset(
    *,
    out_path: str,
    samples: List[Dict[str, Any]],
    build_ast_fn,
    build_problem_fn,
    output_flag: int = 0,
) -> None:
    """
    Dataset schema (patched):

    Each record stores:
      - id: int
      - sample: original sample dict (includes env)
      - ast: ast_to_dict(ast)     (now includes per-node uid)
      - micp:
          - time_sec
          - X, U
          - strategy: Dict[str,int]  (keyed by BinaryDescriptor.id)
          - meta
          - descs: List[dict]        (serialized BinaryDescriptor list)

    No legacy "labels" are written.
    """
    records: List[Dict[str, Any]] = []

    for idx, s in enumerate(samples):
        ast = build_ast_fn(s)
        prob = build_problem_fn(s)

        try:
            X, U, strategy, t_sec = solve_micp_with_strategy(
                ast, prob, output_flag=output_flag
            )
        except RuntimeError as e:
            print(f"[WARN] MICP failed (skipping sample): {e}")
            continue
        except Exception as e:
            print(f"[WARN] Unexpected error (skipping sample): {type(e).__name__}: {e}")
            continue

        T = _infer_T(s, X)

        # Extract descriptors again (must match the ids in strategy.binaries exactly)
        descs = STLBinaryExtractor(int(T)).extract(ast)

        strategy_binaries: Dict[str, int] = strategy.binaries

        # Optional sanity: ensure every desc id exists in strategy dict
        # (strict, because training will match ids exactly)
        for d in descs:
            if d.id not in strategy_binaries:
                raise RuntimeError(f"Dataset generation: strategy missing id={d.id} (role={d.role})")

        records.append(
            {
                "id": int(idx),
                "sample": s,
                "ast": ast_to_dict(ast),
                "micp": {
                    "time_sec": float(t_sec),
                    "X": X.tolist(),
                    "U": U.tolist(),
                    "strategy": strategy_binaries,
                    "meta": strategy.meta,
                    "descs": _descs_to_dicts(descs),
                },
            }
        )

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(samples)}] last_solve={t_sec:.3f}s")

    write_jsonl_gz(out_path, records)
    print(f"Wrote {len(records)} samples to: {out_path}")


__all__ = [
    "PlanningProblem",
    "MICPBuilder",
    "StrategyExtractor",
    "solve_micp_with_strategy",
    "generate_dataset",
    "ast_to_dict",
]