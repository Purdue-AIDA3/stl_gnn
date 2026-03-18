# gnn_train/qp_core.py
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_qp_replay(
    *,
    prob,
    in_rect: List[Tuple[int, Tuple[float, float, float, float], str]],
    outside_face: List[Tuple[int, Tuple[float, float, float, float], int, str]],
    output_flag: int = 0,
    compute_iis: bool = False,
    iis_limit: int = 40,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, int, List[str]]:
    """
    Solve the replay QP with named constraints and optional IIS extraction.

    Returns:
        X, U, solve_time, status_code, iis_constraint_names
    """

    T = int(prob.T)
    A, B = prob.dynamics.matrices()

    m = gp.Model("stl_qp_replay")
    m.Params.OutputFlag = int(output_flag)
    m.Params.DualReductions = 0
    m.Params.InfUnbdInfo = 1

    # State and control variables
    x = m.addVars(T + 1, 4, lb=-GRB.INFINITY, name="x")
    u = m.addVars(T, 2, lb=-GRB.INFINITY, name="u")

    # State bounds
    for k in range(T + 1):
        for i in range(4):
            m.addConstr(
                x[k, i] >= float(prob.x_min[i]),
                name=f"bnd_x_lb_k{k}_i{i}",
            )
            m.addConstr(
                x[k, i] <= float(prob.x_max[i]),
                name=f"bnd_x_ub_k{k}_i{i}",
            )

    # Control bounds
    for k in range(T):
        for i in range(2):
            m.addConstr(
                u[k, i] >= float(prob.u_min[i]),
                name=f"bnd_u_lb_k{k}_i{i}",
            )
            m.addConstr(
                u[k, i] <= float(prob.u_max[i]),
                name=f"bnd_u_ub_k{k}_i{i}",
            )

    # Initial condition
    for i in range(4):
        m.addConstr(
            x[0, i] == float(prob.x0[i]),
            name=f"init_x0_i{i}",
        )

    # Dynamics
    for k in range(T):
        for i in range(4):
            m.addConstr(
                x[k + 1, i]
                == gp.quicksum(float(A[i, j]) * x[k, j] for j in range(4))
                + gp.quicksum(float(B[i, j]) * u[k, j] for j in range(2)),
                name=f"dyn_k{k}_i{i}",
            )

    # In-rectangle constraints
    for (k, rect, tag) in in_rect:
        x1, x2, y1, y2 = rect
        base = tag if tag else f"inrect_k{k}"
        m.addConstr(x[k, 0] >= float(x1), name=f"{base}_x_lb")
        m.addConstr(x[k, 0] <= float(x2), name=f"{base}_x_ub")
        m.addConstr(x[k, 1] >= float(y1), name=f"{base}_y_lb")
        m.addConstr(x[k, 1] <= float(y2), name=f"{base}_y_ub")

    # Outside-face halfspace constraints
    for (t, rect, h, tag) in outside_face:
        x1, x2, y1, y2 = rect
        base = tag if tag else f"outside_t{t}_h{int(h)}"

        if int(h) == 0:
            m.addConstr(x[t, 0] <= float(x1), name=f"{base}_x_le_x1")
        elif int(h) == 1:
            m.addConstr(x[t, 0] >= float(x2), name=f"{base}_x_ge_x2")
        elif int(h) == 2:
            m.addConstr(x[t, 1] <= float(y1), name=f"{base}_y_le_y1")
        elif int(h) == 3:
            m.addConstr(x[t, 1] >= float(y2), name=f"{base}_y_ge_y2")
        else:
            raise ValueError(f"Invalid face index h={h} at t={t}")

    # Objective (control effort)
    obj = gp.quicksum(u[k, i] * u[k, i] for k in range(T) for i in range(2))
    m.setObjective(obj, GRB.MINIMIZE)

    # Optimize
    t0 = time.time()
    m.optimize()
    solve_time = float(time.time() - t0)

    status = int(m.Status)

    # If not optimal, optionally extract IIS
    if status != int(GRB.OPTIMAL):
        iis_names: List[str] = []

        if compute_iis and status in (int(GRB.INFEASIBLE), int(GRB.INF_OR_UNBD)):
            try:
                m.computeIIS()
                count = 0
                for c in m.getConstrs():
                    if getattr(c, "IISConstr", 0) == 1:
                        iis_names.append(str(c.ConstrName))
                        count += 1
                        if count >= int(iis_limit):
                            break
            except Exception:
                iis_names = ["<IIS computation failed>"]

        return None, None, solve_time, status, iis_names

    # Extract solution
    X = np.zeros((T + 1, 4), dtype=np.float64)
    U = np.zeros((T, 2), dtype=np.float64)

    for k in range(T + 1):
        for i in range(4):
            X[k, i] = float(x[k, i].X)

    for k in range(T):
        for i in range(2):
            U[k, i] = float(u[k, i].X)

    return X, U, solve_time, status, []