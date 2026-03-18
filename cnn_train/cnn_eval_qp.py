from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gurobipy import GRB

import matplotlib.pyplot as plt

from scripts import build_samples, build_ast_fn, build_problem_fn
from stl import STLBinaryExtractor, STLNode
from micp import solve_micp_with_strategy
from examples import GeometryConstraints

from gnn_train.qp_core import solve_qp_replay
from gnn_train.retry_policy import solve_with_local_joint_topk
from gnn_train.eval_diag import DiagLogger, status_name, compute_face_switch_stats
from gnn_train.eval_vis import (
    save_case_png,
    EX3_GOAL_PALETTE,
    EX5_PAIR_PALETTE,
    _pick_from_palette,
    _color_for_type_name,
)
from gnn_train.eval_shared import (
    BenchmarkStats,
    _get_obstacles,
    _get_regions,
    _start_consistent_face,
    _force_start_consistent_faces_inplace,
    compute_solution_cost,
    decode_strategy_from_logits,
    _extract_selected_event_times,
    _extract_selected_faces_by_pred,
    check_example1,
    check_example2,
    check_example3,
    check_example4,
    check_example5,
    solve_with_stlpy_fallback,
    start_timer,
    stop_timer,
    write_summary_json,
)

from .cnn_rasterizer import CNNRasterConfig, rasterize_env_to_image
from .cnn_model import CNNBinaryNet, CNNModelConfig


# ==============================================================================
# Checkpoint loader
# ==============================================================================

def load_cnn_from_ckpt(path: str, device: torch.device) -> Tuple[CNNBinaryNet, Dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("CNN checkpoint must be a dict.")

    cfg_dict = ckpt.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        raise ValueError("CNN checkpoint missing dict cfg.")

    cfg = CNNModelConfig(**cfg_dict)
    model = CNNBinaryNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


# ==============================================================================
# CNN inference
# ==============================================================================

@torch.no_grad()
def cnn_predict_logits_by_id(
    *,
    model: CNNBinaryNet,
    sample: Dict[str, Any],
    ast: STLNode,
    device: torch.device,
    raster_cfg: CNNRasterConfig,
) -> Dict[str, torch.Tensor]:
    T = int(sample["T"])
    descs = STLBinaryExtractor(int(T)).extract(ast)

    img = rasterize_env_to_image(sample=sample, cfg=raster_cfg, device=device).unsqueeze(0)

    desc_batch_idx = torch.zeros((len(descs),), device=device, dtype=torch.long)
    desc_meta: List[Dict[str, Any]] = []
    desc_ids: List[str] = []

    for d in descs:
        m = dict(d.meta) if isinstance(d.meta, dict) else {}
        m["role"] = str(d.role)
        m["node_tag"] = str(d.node_tag)
        if "pred" not in m:
            m["pred"] = str(m.get("ent_name", ""))
        if "op_uid" not in m:
            m["op_uid"] = str(d.node_tag)
        if "pred_uid" not in m:
            m["pred_uid"] = str(m.get("pred", ""))
        if "time" not in m:
            m["time"] = int(getattr(d, "time", 0))
        if "face" not in m and "h" in m:
            m["face"] = int(m["h"])
        desc_meta.append(m)
        desc_ids.append(str(d.id))

    logits = model(images=img, desc_batch_idx=desc_batch_idx, desc_meta=desc_meta)

    out: Dict[str, torch.Tensor] = {}
    for did, v in zip(desc_ids, logits):
        out[str(did)] = v
    return out


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--example_id", type=int, default=1)

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

    p.add_argument("--gurobi_out", type=int, default=0)
    p.add_argument("--run_micp_baseline", type=int, default=0)

    p.add_argument("--save_png", type=int, default=1)
    p.add_argument("--png_dir", type=str, default="results/cnn_eval_results")

    p.add_argument("--topk", type=int, default=1, help="Top-K local joint retry (1 disables retry)")
    p.add_argument(
        "--success_metric",
        type=str,
        default="QP_opt",
        choices=["QP_opt", "SAT", "QP_opt_AND_SAT"],
        help="Metric used for success_rate in diagnostics and summary.json.",
    )

    p.add_argument("--diag_log", type=str, default="")
    p.add_argument("--diag_iis", type=int, default=1)
    p.add_argument("--diag_iis_limit", type=int, default=40)

    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)

    args = p.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model, ckpt = load_cnn_from_ckpt(args.ckpt, device=device)

    ex_eval = int(args.example_id)
    if int(ex_eval) not in (1, 2, 3, 4, 5):
        raise ValueError(f"cnn_eval_qp supports example_id 1..5. Got ex_eval={ex_eval}")

    raster_cfg_dict = ckpt.get("raster_cfg", {})
    if not isinstance(raster_cfg_dict, dict):
        raster_cfg_dict = {}
    raster_cfg = CNNRasterConfig(**{**raster_cfg_dict, "H": int(args.H), "W": int(args.W)})

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
        example_id=int(ex_eval),
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

    os.makedirs(args.png_dir, exist_ok=True)

    diag_path = str(args.diag_log).strip()
    if diag_path == "":
        diag_path = os.path.join(str(args.png_dir), "diagnostics.txt")
    diag = DiagLogger(diag_path)
    diag.writeln("==== cnn_eval_qp diagnostics (overwrite) ====")
    diag.writeln(f"ckpt={str(args.ckpt)}")
    diag.writeln(f"example_id={int(ex_eval)}  n={int(args.n)}  seed={int(args.seed)}")
    diag.writeln(f"device={str(device)}")
    diag.writeln("")

    n_qp_opt = 0
    n_sat = 0
    n_avoid = 0
    n_reach = 0
    n_extra = 0

    n_opt_sat = 0
    n_opt_unsat = 0
    n_nonopt = 0

    t_qp_list: List[float] = []
    t_micp_list: List[float] = []
    n_micp_opt = 0
    n_eval = 0
    n_skip_baseline = 0
    qp_nonopt: List[Tuple[int, int]] = []
    bench_stats = BenchmarkStats()
    n_success_metric = 0

    for i, s in enumerate(samples):
        if int(args.run_micp_baseline) == 1:
            ast_base = build_ast_fn(s)
            prob_base = build_problem_fn(s)
            X_m = U_m = strat_m = None
            t_m = None
            try:
                X_m, U_m, strat_m, t_m = solve_micp_with_strategy(
                    ast_base, prob_base, output_flag=int(args.gurobi_out)
                )
            except RuntimeError as e:
                print(f"[WARN] MICP baseline failed (skipping sample): {e}")
            except Exception as e:
                print(f"[WARN] Unexpected baseline error (skipping sample): {type(e).__name__}: {e}")

            if X_m is None:
                n_skip_baseline += 1
                continue

            n_micp_opt += 1
            if t_m is not None:
                t_micp_list.append(float(t_m))

        n_eval += 1
        t_case0 = start_timer()

        ast = build_ast_fn(s)
        prob = build_problem_fn(s)
        T = int(prob.T)
        t_predict = 0.0
        t_decode = 0.0
        t_qp_total = 0.0
        t_fallback = 0.0
        t_total = 0.0
        solved_k = -1
        used_fallback = False
        cost_val = float("nan")

        t0_predict = start_timer()
        bin_logits_by_id = cnn_predict_logits_by_id(
            model=model,
            sample=s,
            ast=ast,
            device=device,
            raster_cfg=raster_cfg,
        )
        t_predict = stop_timer(t0_predict)

        t0_decode = start_timer()
        strategy_base, descs_check = decode_strategy_from_logits(
            ast=ast,
            T=T,
            bin_logits_by_id=bin_logits_by_id,
        )
        t_decode = stop_timer(t0_decode)

        env = s["env"]
        obstacles = _get_obstacles(env)
        regions = _get_regions(env)
        meta = env.get("meta", {}) or {}

        faces_fixed = _extract_selected_faces_by_pred(descs=descs_check, strategy=strategy_base, T=T)

        obstacle_rect_by_pred: Dict[str, Tuple[float, float, float, float]] = {}
        outside_face_obstacles_index: Dict[Tuple[str, int], int] = {}

        outside_face_obstacles: List[Tuple[int, Tuple[float, float, float, float], int, str]] = []
        forced_start_changed: Dict[str, int] = {}

        for oi, O_rect in enumerate(obstacles):
            pred = f"O{oi}"
            obstacle_rect_by_pred[pred] = O_rect
            if pred not in faces_fixed:
                raise RuntimeError(f"Missing faces for obstacle pred={pred}. Have: {list(faces_fixed.keys())[:10]}")
            faces = faces_fixed[pred]

            prev0 = int(faces[0]) if isinstance(faces, np.ndarray) and faces.ndim == 1 and faces.shape[0] >= 1 else 0
            _force_start_consistent_faces_inplace(prob=prob, rect=O_rect, faces=faces)
            forced_start_changed[pred] = int(int(faces[0]) != int(prev0))

            for t in range(T + 1):
                tag = f"outside_{pred}_t{int(t)}_h{int(faces[t])}"
                outside_face_obstacles_index[(pred, int(t))] = len(outside_face_obstacles)
                outside_face_obstacles.append((int(t), O_rect, int(faces[t]), tag))

        face_stats = compute_face_switch_stats(faces_fixed)

        def _build_constraints_from_strategy(strat_local: Dict[str, int]):
            events_local = _extract_selected_event_times(descs=descs_check, strategy=strat_local)

            in_rect_local: List[Tuple[int, Tuple[float, float, float, float], str]] = []
            outside_face_local: List[Tuple[int, Tuple[float, float, float, float], int, str]] = list(outside_face_obstacles)

            faces_override = strat_local.get("__faces_override__", None)
            if isinstance(faces_override, dict):
                for pred, mp in faces_override.items():
                    if not isinstance(mp, dict):
                        continue
                    for t_str, h_new in mp.items():
                        try:
                            t = int(t_str)
                            h = int(h_new)
                        except Exception:
                            continue
                        if h not in (0, 1, 2, 3):
                            continue

                        key = (str(pred), int(t))
                        if key in outside_face_obstacles_index:
                            idx = outside_face_obstacles_index[key]
                            if int(t) == 0 and str(pred) in obstacle_rect_by_pred:
                                r = obstacle_rect_by_pred[str(pred)]
                                h = int(_start_consistent_face(x0=float(prob.x0[0]), y0=float(prob.x0[1]), rect=r))

                            rect = obstacle_rect_by_pred.get(str(pred), outside_face_local[idx][1])
                            tag = f"outside_{str(pred)}_t{int(t)}_h{int(h)}"
                            outside_face_local[idx] = (int(t), rect, int(h), tag)

            filled_rects_local: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None
            edgecolor_by_name_local: Optional[Dict[str, Any]] = None
            fillcolor_by_name_local: Optional[Dict[str, Any]] = None
            highlight_names_local: Optional[set] = None
            linestyle_by_name_local: Optional[Dict[str, str]] = None

            G_rect_local: Optional[Tuple[float, float, float, float]] = None
            extra_rects_local: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None

            if int(ex_eval) == 1:
                sel = [(pred, k, m) for (pred, k, m) in events_local if pred == "G" and not bool(m.get("until", False))]
                if len(sel) != 1:
                    raise RuntimeError(f"Example1: expected exactly 1 selected goal event_time for pred='G', got {sel}")
                _, k_goal, _ = sel[0]
                if "G" not in regions:
                    raise RuntimeError("Example1: env.regions missing 'G'")
                G_rect_local = regions["G"]
                in_rect_local.append((int(k_goal), G_rect_local, f"in_G_t{int(k_goal)}"))

            elif int(ex_eval) == 2:
                selG = [(pred, k, m) for (pred, k, m) in events_local if pred == "G" and not bool(m.get("until", False))]
                if len(selG) != 1:
                    raise RuntimeError(f"Example2: expected exactly 1 selected goal event_time for pred='G', got {selG}")
                _, k_goal, _ = selG[0]
                G_rect_local = regions["G"]
                in_rect_local.append((int(k_goal), G_rect_local, f"in_G_t{int(k_goal)}"))

                selD = [(pred, k, m) for (pred, k, m) in events_local if bool(m.get("dwell_a", False)) or bool(m.get("dwell_b", False))]
                if len(selD) != 1:
                    raise RuntimeError(f"Example2: expected exactly 1 selected dwell event_time, got {selD}")
                pred_d, k_mid, _ = selD[0]

                tau = int(meta.get("tau", env.get("tau", None)))
                if tau is None:
                    raise RuntimeError("Example2: tau missing in env['meta']['tau'] (or legacy env['tau']).")
                if pred_d not in regions:
                    raise RuntimeError(f"Example2: dwell pred {pred_d} missing in regions.")
                R = regions[pred_d]
                for t in range(int(k_mid), int(k_mid) + int(tau) + 1):
                    in_rect_local.append((t, R, f"in_{str(pred_d)}_t{int(t)}_dwell"))

                extra_rects_local = [(regions["T1"], "T1"), (regions["T2"], "T2")]

            elif int(ex_eval) == 3:
                goals = [k for k in sorted(regions.keys()) if k.startswith("G")]
                if len(goals) < 1:
                    raise RuntimeError("Example3: expected at least one goal region named G0,G1,...")
                sel = [(pred, k, m) for (pred, k, m) in events_local if pred.startswith("G")]
                if len(sel) != 1:
                    raise RuntimeError(f"Example3: expected exactly 1 selected goal event_time among {goals}, got {sel}")
                pred_g, k_goal, _ = sel[0]
                if pred_g not in regions:
                    raise RuntimeError(f"Example3: selected goal pred {pred_g} not in regions.")

                edgecolor_by_name_local = {}
                fillcolor_by_name_local = {}
                for idx, gname in enumerate(goals):
                    c = _pick_from_palette(idx, EX3_GOAL_PALETTE)
                    edgecolor_by_name_local[gname] = c
                    fillcolor_by_name_local[gname] = c

                extra_rects_local = [(regions[g], g) for g in goals]
                filled_rects_local = [(regions[pred_g], str(pred_g))]
                highlight_names_local = {str(pred_g)}

                in_rect_local.append((int(k_goal), regions[pred_g], f"in_{str(pred_g)}_t{int(k_goal)}"))
                G_rect_local = None

            elif int(ex_eval) == 4:
                types = meta.get("types", None)
                if not isinstance(types, list) or len(types) < 1:
                    infer_types = set()
                    for rk in regions.keys():
                        if rk.startswith("T_"):
                            parts = rk.split("_", 2)
                            if len(parts) == 3:
                                infer_types.add(parts[1])
                    types = sorted(list(infer_types))
                if not isinstance(types, list) or len(types) < 1:
                    raise RuntimeError("Example4: could not determine types.")

                type_to_color: Dict[str, Any] = {}
                for typ in types:
                    type_to_color[str(typ)] = _color_for_type_name(str(typ))

                edgecolor_by_name_local = {}
                fillcolor_by_name_local = {}
                for rk in regions.keys():
                    if not rk.startswith("T_"):
                        continue
                    parts = rk.split("_", 2)
                    if len(parts) != 3:
                        continue
                    typ = parts[1]
                    c = type_to_color.get(str(typ), (0, 0, 0, 1))
                    edgecolor_by_name_local[str(rk)] = c
                    fillcolor_by_name_local[str(rk)] = c

                picked: Dict[str, Tuple[str, int]] = {}
                for (pred, k, _) in events_local:
                    if not pred.startswith("T_"):
                        continue
                    parts = pred.split("_", 2)
                    if len(parts) != 3:
                        continue
                    typ = parts[1]
                    picked[str(typ)] = (str(pred), int(k))

                selected_names = set()
                filled_rects_local = []
                for typ in types:
                    st = str(typ)
                    if st not in picked:
                        raise RuntimeError(f"Example4: missing selected event_time for type={st}. picked={picked}")
                    pred_t, k_t = picked[st]
                    if pred_t not in regions:
                        raise RuntimeError(f"Example4: selected target {pred_t} not in regions.")
                    in_rect_local.append((int(k_t), regions[pred_t], f"in_{str(pred_t)}_t{int(k_t)}"))
                    selected_names.add(pred_t)
                    filled_rects_local.append((regions[pred_t], pred_t))

                highlight_names_local = selected_names
                extra_rects_local = [(regions[k], k) for k in sorted(regions.keys()) if k.startswith("T_")]
                G_rect_local = None

            elif int(ex_eval) == 5:
                door_keys = sorted([k for k in regions.keys() if k.startswith("D")])
                key_keys = sorted([k for k in regions.keys() if k.startswith("K")])
                if "G" not in regions:
                    raise RuntimeError("Example5: missing regions['G']")
                G_rect_local = regions["G"]

                n_pairs = min(len(door_keys), len(key_keys))
                if n_pairs < 1:
                    raise RuntimeError("Example5: expected at least one door/key pair named D0/K0,...")

                selG = [(pred, k, m) for (pred, k, m) in events_local if pred == "G" and not bool(m.get("until", False))]
                if len(selG) != 1:
                    raise RuntimeError(f"Example5: expected exactly 1 selected goal event_time for pred='G', got {selG}")
                _, k_goal, _ = selG[0]
                in_rect_local.append((int(k_goal), G_rect_local, f"in_G_t{int(k_goal)}"))

                for idx in range(n_pairs):
                    Di = f"D{idx}"
                    Ki = f"K{idx}"
                    if Di not in regions or Ki not in regions:
                        raise RuntimeError(f"Example5: missing region {Di} or {Ki}")
                    D_rect = regions[Di]
                    K_rect = regions[Ki]

                    selK = [(pred, k, m) for (pred, k, m) in events_local if pred == Ki and bool(m.get("until", False))]
                    if len(selK) != 1:
                        raise RuntimeError(f"Example5: expected exactly 1 selected until event_time for pred={Ki}, got {selK}")
                    _, k_key, _ = selK[0]
                    k_key = int(k_key)

                    in_rect_local.append((k_key, K_rect, f"in_{Ki}_t{int(k_key)}"))

                    if Di not in faces_fixed:
                        raise RuntimeError(f"Example5: missing faces for door pred={Di}. Have: {list(faces_fixed.keys())[:10]}")
                    facesD = faces_fixed[Di]

                    prev0 = int(facesD[0]) if isinstance(facesD, np.ndarray) and facesD.ndim == 1 and facesD.shape[0] >= 1 else 0
                    _force_start_consistent_faces_inplace(prob=prob, rect=D_rect, faces=facesD)
                    forced_start_changed[Di] = int(int(facesD[0]) != int(prev0))

                    for t in range(0, k_key):
                        tag = f"outside_{Di}_t{int(t)}_h{int(facesD[t])}"
                        outside_face_local.append((t, D_rect, int(facesD[t]), tag))

                faces_override2 = strat_local.get("__faces_override__", None)
                if isinstance(faces_override2, dict):
                    for pred, mp in faces_override2.items():
                        if not (isinstance(pred, str) and pred.startswith("D")):
                            continue
                        if not isinstance(mp, dict):
                            continue
                        for t_str, h_new in mp.items():
                            try:
                                t = int(t_str)
                                h = int(h_new)
                            except Exception:
                                continue
                            if h not in (0, 1, 2, 3):
                                continue
                            if int(t) < 0:
                                continue
                            for j in range(len(outside_face_local)):
                                tj, rectj, hj, tagj = outside_face_local[j]
                                if int(tj) != int(t):
                                    continue
                                if not str(tagj).startswith(f"outside_{pred}_t{int(t)}_"):
                                    continue
                                if int(t) == 0:
                                    h = int(_start_consistent_face(x0=float(prob.x0[0]), y0=float(prob.x0[1]), rect=rectj))
                                outside_face_local[j] = (int(t), rectj, int(h), f"outside_{pred}_t{int(t)}_h{int(h)}")
                                break

                edgecolor_by_name_local = {}
                fillcolor_by_name_local = (fillcolor_by_name_local or {})
                linestyle_by_name_local = {}

                for idx in range(n_pairs):
                    Di = f"D{idx}"
                    Ki = f"K{idx}"
                    c = _pick_from_palette(idx, EX5_PAIR_PALETTE)
                    edgecolor_by_name_local[Di] = c
                    edgecolor_by_name_local[Ki] = c
                    linestyle_by_name_local[Di] = "-"
                    linestyle_by_name_local[Ki] = "--"

                extra_rects_local = (
                    [(regions[k], k) for k in sorted(regions.keys()) if k.startswith("D") or k.startswith("K")]
                    + [(regions["G"], "G")]
                )

            else:
                raise ValueError(f"Unsupported ex_eval={ex_eval}")

            aux_local = {
                "events": events_local,
                "faces_by_pred": faces_fixed,
                "G_rect": G_rect_local,
                "extra_rects": extra_rects_local,
                "filled_rects": filled_rects_local,
                "edgecolor_by_name": edgecolor_by_name_local,
                "fillcolor_by_name": fillcolor_by_name_local,
                "highlight_names": highlight_names_local,
                "linestyle_by_name": linestyle_by_name_local,
            }
            return in_rect_local, outside_face_local, aux_local

        topk = int(getattr(args, "topk", 1))
        if topk <= 1:
            strategy = dict(strategy_base)
            in_rect, outside_face, aux = _build_constraints_from_strategy(strategy)

            X, U, t_qp, status_qp, iis_names = solve_qp_replay(
                prob=prob,
                in_rect=in_rect,
                outside_face=outside_face,
                output_flag=int(args.gurobi_out),
                compute_iis=(int(args.diag_iis) == 1),
                iis_limit=int(args.diag_iis_limit),
            )
            t_qp_total += float(t_qp)
            solved_k = 1 if int(status_qp) == int(GRB.OPTIMAL) and X is not None else -1
        else:
            def _accept_sat(Xcand: np.ndarray) -> bool:
                if int(ex_eval) == 1:
                    _, _, sat_c = check_example1(X=Xcand, obstacles=obstacles, G_rect=regions["G"])
                    return bool(sat_c)
                if int(ex_eval) == 2:
                    _, _, _, sat_c = check_example2(
                        X=Xcand,
                        obstacles=obstacles,
                        G_rect=regions["G"],
                        T1_rect=regions["T1"],
                        T2_rect=regions["T2"],
                        tau=int(meta.get("tau", env.get("tau", 0))),
                    )
                    return bool(sat_c)
                if int(ex_eval) == 3:
                    goals = [regions[k] for k in sorted(regions.keys()) if k.startswith("G")]
                    _, _, sat_c = check_example3(X=Xcand, obstacles=obstacles, goals=goals)
                    return bool(sat_c)
                if int(ex_eval) == 4:
                    targets_by_type: Dict[str, List[Tuple[float, float, float, float]]] = {}
                    for rk in regions.keys():
                        if not rk.startswith("T_"):
                            continue
                        parts = rk.split("_", 2)
                        if len(parts) == 3:
                            targets_by_type.setdefault(parts[1], []).append(regions[rk])
                    _, _, sat_c = check_example4(X=Xcand, obstacles=obstacles, targets_by_type=targets_by_type)
                    return bool(sat_c)
                if int(ex_eval) == 5:
                    doors = [regions[k] for k in sorted(regions.keys()) if k.startswith("D")]
                    keys = [regions[k] for k in sorted(regions.keys()) if k.startswith("K")]
                    _, _, _, sat_c = check_example5(X=Xcand, obstacles=obstacles, doors=doors, keys=keys, G_rect=regions["G"])
                    return bool(sat_c)
                return False

            X, U, t_qp, status_qp, iis_names, strategy, aux = solve_with_local_joint_topk(
                prob=prob,
                K=topk,
                base_strategy=strategy_base,
                descs=descs_check,
                bin_logits_by_id=bin_logits_by_id,
                build_constraints_fn=_build_constraints_from_strategy,
                output_flag=int(args.gurobi_out),
                compute_iis=(int(args.diag_iis) == 1),
                iis_limit=int(args.diag_iis_limit),
                accept_fn=_accept_sat,
                accept_name="SAT",
            )
            t_qp_total += float(t_qp)
            if isinstance(aux, dict):
                solved_k = int(aux.get("solved_k", -1))

        if (int(status_qp) != int(GRB.OPTIMAL)) or (X is None):
            X_fb, U_fb, t_fb = solve_with_stlpy_fallback(
                sample={"T": T, "env": s["env"]},
                example_id=int(ex_eval),
            )
            t_fallback = float(t_fb)
            if X_fb is not None and U_fb is not None:
                X = X_fb
                U = U_fb
                status_qp = int(GRB.OPTIMAL)
                used_fallback = True
                if solved_k < 0:
                    solved_k = int(topk) + 1

        events = aux.get("events", [])
        faces_by_pred = aux.get("faces_by_pred", faces_fixed)

        G_rect = aux.get("G_rect", None)
        extra_rects = aux.get("extra_rects", None)
        filled_rects = aux.get("filled_rects", None)
        edgecolor_by_name = aux.get("edgecolor_by_name", None)
        fillcolor_by_name = aux.get("fillcolor_by_name", None)
        highlight_names = aux.get("highlight_names", None)
        linestyle_by_name = aux.get("linestyle_by_name", None)

        t_total = stop_timer(t_case0)
        t_qp_list.append(float(t_qp_total))

        sat = False
        avoid_ok = False
        reach_ok = False
        extra_ok: Optional[bool] = None

        case_events = [(str(pred), int(k)) for (pred, k, _) in events]
        status_str = status_name(int(status_qp))

        if int(status_qp) == int(GRB.OPTIMAL) and X is not None:
            n_qp_opt += 1
            if int(ex_eval) == 1:
                avoid_ok, reach_ok, sat = check_example1(X=X, obstacles=obstacles, G_rect=regions["G"])
            elif int(ex_eval) == 2:
                avoid_ok, reach_ok, mid_ok, sat = check_example2(
                    X=X,
                    obstacles=obstacles,
                    G_rect=regions["G"],
                    T1_rect=regions["T1"],
                    T2_rect=regions["T2"],
                    tau=int(meta.get("tau", env.get("tau", 0))),
                )
                extra_ok = bool(mid_ok)
            elif int(ex_eval) == 3:
                goals = [regions[k] for k in sorted(regions.keys()) if k.startswith("G")]
                avoid_ok, reach_ok, sat = check_example3(X=X, obstacles=obstacles, goals=goals)
            elif int(ex_eval) == 4:
                targets_by_type: Dict[str, List[Tuple[float, float, float, float]]] = {}
                for rk in regions.keys():
                    if not rk.startswith("T_"):
                        continue
                    parts = rk.split("_", 2)
                    if len(parts) == 3:
                        targets_by_type.setdefault(parts[1], []).append(regions[rk])
                avoid_ok, reach_all, sat = check_example4(X=X, obstacles=obstacles, targets_by_type=targets_by_type)
                reach_ok = bool(reach_all)
                extra_ok = bool(reach_all)
            elif int(ex_eval) == 5:
                doors = [regions[k] for k in sorted(regions.keys()) if k.startswith("D")]
                keys = [regions[k] for k in sorted(regions.keys()) if k.startswith("K")]
                avoid_ok, reach_ok, doors_ok, sat = check_example5(X=X, obstacles=obstacles, doors=doors, keys=keys, G_rect=regions["G"])
                extra_ok = bool(doors_ok)

            n_avoid += int(avoid_ok)
            n_reach += int(reach_ok)
            n_sat += int(sat)
            if extra_ok is not None:
                n_extra += int(extra_ok)

            if bool(sat):
                n_opt_sat += 1
            else:
                n_opt_unsat += 1
            if U is not None:
                cost_val = compute_solution_cost(prob=prob, X=X, U=U)
        else:
            qp_nonopt.append((int(i), int(status_qp)))
            n_nonopt += 1

        if str(args.success_metric) == "QP_opt":
            success_case = bool(int(status_qp) == int(GRB.OPTIMAL) and X is not None)
        elif str(args.success_metric) == "SAT":
            success_case = bool(sat)
        else:
            success_case = bool(int(status_qp) == int(GRB.OPTIMAL) and X is not None and sat)
        n_success_metric += int(success_case)

        if int(status_qp) == int(GRB.OPTIMAL) and X is not None:
            bench_stats.add_case(
                time_total=float(t_total),
                time_qp=float(t_qp_total),
                solved_k=int(solved_k if solved_k >= 0 else 1),
                used_fallback=bool(used_fallback),
                cost=float(cost_val),
            )

        diag.writeln(f"--- case {int(i):04d} ---")
        diag.writeln(
            f"status={status_str} ({int(status_qp)})  T={int(T)}  "
            f"t_predict={float(t_predict):.6f}s  "
            f"t_decode={float(t_decode):.6f}s  "
            f"t_qp_total={float(t_qp_total):.6f}s  "
            f"t_fallback={float(t_fallback):.6f}s  "
            f"t_total={float(t_total):.6f}s"
        )
        if int(status_qp) == int(GRB.OPTIMAL):
            diag.writeln(
                f"sat={int(bool(sat))}  avoid_ok={int(bool(avoid_ok))}  reach_ok={int(bool(reach_ok))}  extra_ok={(None if extra_ok is None else int(bool(extra_ok)))}"
            )
        else:
            diag.writeln("sat=NA  avoid_ok=NA  reach_ok=NA  extra_ok=NA")
        diag.writeln(
            f"success_metric={str(args.success_metric)}  success={int(bool(success_case))}  "
            f"solved_k={int(solved_k)}  fallback={int(bool(used_fallback))}  cost={float(cost_val):.6f}"
        )

        if len(case_events) > 0:
            diag.writeln("event_time_selections:")
            for (pred, kk) in case_events:
                diag.writeln(f"  - pred={pred}  k={int(kk)}")
        else:
            diag.writeln("event_time_selections: <none>")

        if int(ex_eval) == 5:
            try:
                selG2 = [(pred, k, m) for (pred, k, m) in events if pred == "G" and not bool(m.get("until", False))]
                if len(selG2) == 1:
                    diag.writeln(f"ex5_k_goal={int(selG2[0][1])}")
                selK2 = [(pred, k, m) for (pred, k, m) in events if pred.startswith("K") and bool(m.get("until", False))]
                if len(selK2) > 0:
                    diag.writeln("ex5_k_keys:")
                    for (pred, k, _) in selK2:
                        diag.writeln(f"  - {str(pred)}: k={int(k)}")
            except Exception:
                diag.writeln("ex5_k_goal/ex5_k_keys: <failed to parse>")

        if len(face_stats) > 0:
            diag.writeln("face_switch_stats:")
            for pred in sorted(face_stats.keys()):
                st = face_stats[pred]
                ch = int(forced_start_changed.get(pred, 0))
                diag.writeln(f"  - pred={pred}  switches={int(st['switches'])}  n_segments={int(st['n_segments'])}  forced_start_changed={ch}")
        else:
            diag.writeln("face_switch_stats: <none>")

        if int(status_qp) in (int(GRB.INFEASIBLE), int(GRB.INF_OR_UNBD)) and len(iis_names) > 0:
            diag.writeln(f"IIS (first {min(len(iis_names), int(args.diag_iis_limit))} constraints):")
            for nm in iis_names:
                diag.writeln(f"  - {str(nm)}")
        diag.writeln("")

        save_case_png(
            save_png=int(args.save_png),
            png_dir=str(args.png_dir),
            i=int(i),
            X=X,
            x0_env=np.array(env["x0"], dtype=float),
            obstacles=obstacles,
            G_rect=G_rect,
            sat=bool(sat),
            avoid_ok=bool(avoid_ok),
            reach_ok=bool(reach_ok),
            status_qp=int(status_qp),
            extra_ok=extra_ok,
            extra_rects=extra_rects,
            faces_by_pred=faces_by_pred,
            filled_rects=filled_rects,
            edgecolor_by_name=edgecolor_by_name,
            highlight_names=highlight_names,
            fillcolor_by_name=fillcolor_by_name,
            linestyle_by_name=linestyle_by_name,
        )

        if (i + 1) % 10 == 0 or (i + 1) == int(args.n):
            avg_qp = (sum(t_qp_list) / max(1, len(t_qp_list)))
            print(
                f"[{i+1:4d}/{int(args.n):4d}] "
                f"QP_opt={n_qp_opt} SAT={n_sat} avoid={n_avoid} reach={n_reach}"
                + (f" extra={n_extra}" if int(ex_eval) in (2, 4, 5) else "")
                + f" avg_qp={avg_qp:.3f}s"
            )
            diag.flush()

    avg_qp = (sum(t_qp_list) / max(1, len(t_qp_list))) if len(t_qp_list) > 0 else float("nan")
    avg_m = (sum(t_micp_list) / max(1, len(t_micp_list))) if len(t_micp_list) > 0 else float("nan")
    agg = bench_stats.compute()

    print("\n==== Summary ====")
    denom = int(n_eval) if int(args.run_micp_baseline) == 1 else int(args.n)
    if int(args.run_micp_baseline) == 1:
        print(f"[INFO] Baseline-gated eval: skipped={int(n_skip_baseline)} kept={int(n_eval)}")
    print(f"example_id={int(ex_eval)}  n={denom}  seed={int(args.seed)}")
    print(f"QP optimal: {n_qp_opt}/{denom}  avg_qp={avg_qp:.4f}s")
    print(f"SAT: {n_sat}/{denom}")
    print(f"avoid_ok: {n_avoid}/{denom}  reach_ok: {n_reach}/{denom}")
    if int(ex_eval) in (2, 4, 5):
        print(f"extra_ok: {n_extra}/{denom}")
    print(f"OPTIMAL+SAT: {n_opt_sat}/{denom}  OPTIMAL+UNSAT: {n_opt_unsat}/{denom}  NON-OPT: {n_nonopt}/{denom}")
    print(f"success_metric={str(args.success_metric)}  success={n_success_metric}/{denom}")
    print(
        f"time_total_mean={float(agg['time_total_mean']):.6f}s  "
        f"time_total_std={float(agg['time_total_std']):.6f}s"
    )
    print(
        f"cost_mean={float(agg['cost_mean']):.6f}  "
        f"cost_std={float(agg['cost_std']):.6f}"
    )
    if int(args.run_micp_baseline) == 1:
        print(f"MICP optimal: {n_micp_opt}/{denom}  avg_micp={avg_m:.4f}s")

    if len(qp_nonopt) > 0:
        print("\nNon-optimal QP statuses (idx,status):")
        print(qp_nonopt[:50])
        if len(qp_nonopt) > 50:
            print(f"... ({len(qp_nonopt)} total)")

    diag.writeln("==== aggregate diagnostics ====")
    diag.writeln(f"QP_opt={int(n_qp_opt)}  SAT={int(n_sat)}  avoid_ok={int(n_avoid)}  reach_ok={int(n_reach)}")
    if int(ex_eval) in (2, 4, 5):
        diag.writeln(f"extra_ok={int(n_extra)}")
    diag.writeln(f"OPTIMAL+SAT={int(n_opt_sat)}  OPTIMAL+UNSAT={int(n_opt_unsat)}  NON-OPT={int(n_nonopt)}")
    diag.writeln(f"success_metric={str(args.success_metric)}  success={int(n_success_metric)}  n_total={int(denom)}")
    diag.writeln(
        f"time_total_mean={float(agg['time_total_mean']):.6f}  "
        f"time_total_std={float(agg['time_total_std']):.6f}"
    )
    diag.writeln(
        f"time_qp_mean={float(agg['time_qp_mean']):.6f}  "
        f"time_qp_std={float(agg['time_qp_std']):.6f}"
    )
    diag.writeln(
        f"solved_k_mean={float(agg['solved_k_mean']):.6f}  "
        f"solved_k_std={float(agg['solved_k_std']):.6f}"
    )
    diag.writeln(
        f"cost_mean={float(agg['cost_mean']):.6f}  "
        f"cost_std={float(agg['cost_std']):.6f}"
    )
    diag.writeln(
        f"fallback_count={int(bench_stats.fallback_count)}  "
        f"fallback_rate={float(bench_stats.fallback_count / max(1, denom)):.6f}"
    )
    write_summary_json(
        out_dir=str(args.png_dir),
        solver_name="cnn",
        example_id=int(ex_eval),
        n_cases=int(denom),
        success_rate=float(n_success_metric / max(1, denom)),
        stats=bench_stats,
    )
    diag.flush()
    diag.close()


if __name__ == "__main__":
    main()