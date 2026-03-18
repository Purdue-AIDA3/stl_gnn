from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data, HeteroData


# ----------------------------
# Feature schema (fixed-length)
# ----------------------------
# We use a single dense feature layout for all node types.
#
# Feature vector layout (length = 64):
#   kind_onehot(4):      [OP, PRED, ENT, B]
#   op_onehot(5):        [And, Or, Always, Eventually, Until]
#   pred_onehot(2):      [InRect, OutsideRect]
#   ent_super_onehot(2): [Obstacle, Region]
#   ent_role_onehot(6):  [Obstacle, Goal, Target, Door, Key, OtherRegion]
#   bin_role_onehot(3):  [event_time, outside_face, always_or_choice]
#   geom(12):
#       [xmin, xmax, ymin, ymax, cx, cy, w, h, area, dx0, dy0, dist0]
#   temporal(6):
#       [a_or_t, b_or_aux, span, midpoint, slack_to_T, tau_norm]
#   binary(8):
#       [t0, opt, dwell_a, dwell_b, face, until_flag, pred_sign, has_pred_uid]
#   global(8):
#       [T, tau, n_obs, n_regions, x0x, x0y, x0vx, x0vy]
#   name_hash(8):
#       stable numeric signature for node identity
#
# Notes:
# - Geometry is explicit rather than overloading a generic rect slot.
# - Temporal information is explicit for temporal operators and binary nodes.
# - Binary descriptor semantics are expanded using meta fields already present in rec["micp"]["descs"].
# - Entity role splits goals / targets / doors / keys / generic regions without changing graph structure.


X_DIM = 64


def _zero_rect() -> List[float]:
    return [0.0, 0.0, 0.0, 0.0]


def _rect_to_list(r: Any) -> List[float]:
    if isinstance(r, (list, tuple)) and len(r) == 4:
        return [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
    return _zero_rect()


def _get_env(sample: Dict[str, Any]) -> Dict[str, Any]:
    env = sample.get("env", {})
    return env if isinstance(env, dict) else {}


def _get_x0(env: Dict[str, Any]) -> List[float]:
    x0 = env.get("x0", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(x0, (list, tuple)) or len(x0) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(x0[0]), float(x0[1]), float(x0[2]), float(x0[3])]


def _get_obstacles(env: Dict[str, Any]) -> List[List[float]]:
    if "obstacles" in env and isinstance(env["obstacles"], list):
        out: List[List[float]] = []
        for r in env["obstacles"]:
            if isinstance(r, (list, tuple)) and len(r) == 4:
                out.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        return out

    if "O" in env and isinstance(env["O"], (list, tuple)) and len(env["O"]) == 4:
        r = env["O"]
        return [[float(r[0]), float(r[1]), float(r[2]), float(r[3])]]

    return []


def _get_regions(env: Dict[str, Any]) -> Dict[str, List[float]]:
    regions = env.get("regions", None)
    if isinstance(regions, dict):
        out: Dict[str, List[float]] = {}
        for k, v in regions.items():
            if isinstance(v, (list, tuple)) and len(v) == 4:
                out[str(k)] = [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
        return out
    return {}


def _get_tau(env: Dict[str, Any]) -> int:
    meta = env.get("meta", {})
    if isinstance(meta, dict) and "tau" in meta:
        try:
            return int(meta["tau"])
        except Exception:
            return 0

    try:
        return int(env.get("tau", 0))
    except Exception:
        return 0


def _stable_hash8(s: str) -> List[float]:
    import hashlib

    h = hashlib.sha256(s.encode("utf-8")).digest()
    out: List[float] = []
    for i in range(8):
        v = int.from_bytes(h[2 * i : 2 * i + 2], "big")
        out.append(float(v) / 65536.0)
    return out


def _to_int_or_none(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    return None


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _edge_index_from_pairs(pairs: List[Tuple[int, int]]) -> torch.Tensor:
    if len(pairs) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _dedup_pairs(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen: Set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for a, b in pairs:
        key = (int(a), int(b))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _add_bidirectional_clique(edges: List[Tuple[int, int]], nodes: List[int]) -> None:
    uniq = sorted(set(int(v) for v in nodes))
    n = len(uniq)
    if n < 2:
        return
    for i in range(n):
        a = uniq[i]
        for j in range(i + 1, n):
            b = uniq[j]
            edges.append((a, b))
            edges.append((b, a))


def _rect_stats(rect: Optional[List[float]], x0: List[float]) -> List[float]:
    r = rect if isinstance(rect, list) and len(rect) == 4 else _zero_rect()
    xmin, xmax, ymin, ymax = [float(v) for v in r]
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    area = w * h
    dx0 = cx - float(x0[0])
    dy0 = cy - float(x0[1])
    dist0 = (dx0 * dx0 + dy0 * dy0) ** 0.5
    return [xmin, xmax, ymin, ymax, cx, cy, w, h, area, dx0, dy0, dist0]


def _entity_role_onehot(ent: Optional[str], ent_name: str) -> List[float]:
    role_oh = [0.0] * 6
    name = str(ent_name)

    idx = 5
    if str(ent) == "Obstacle":
        idx = 0
    elif name == "G" or name.startswith("G_"):
        idx = 1
    elif name.startswith("T"):
        idx = 2
    elif name.startswith("D"):
        idx = 3
    elif name.startswith("K"):
        idx = 4

    role_oh[idx] = 1.0
    return role_oh


def _binary_role_onehot(role: str) -> List[float]:
    out = [0.0, 0.0, 0.0]
    role_map = {"event_time": 0, "outside_face": 1, "always_or_choice": 2}
    if role in role_map:
        out[role_map[role]] = 1.0
    return out


def _pred_sign(pred: Optional[str]) -> float:
    if pred == "InRect":
        return 1.0
    if pred == "OutsideRect":
        return -1.0
    return 0.0


def _binary_scalar_block(*, role: str, pred: Optional[str], meta: Dict[str, Any], time_field: Any) -> List[float]:
    t0 = -1.0
    if "t" in meta:
        t0 = _to_float(meta.get("t", -1), -1.0)
    elif "k" in meta:
        t0 = _to_float(meta.get("k", -1), -1.0)
    elif time_field is not None:
        t0 = _to_float(time_field, -1.0)

    opt = _to_float(meta.get("opt", 0), 0.0)
    dwell_a = _to_float(meta.get("dwell_a", 0), 0.0)
    dwell_b = _to_float(meta.get("dwell_b", 0), 0.0)
    face = _to_float(meta.get("face", meta.get("h", 0)), 0.0)
    until_flag = 1.0 if bool(meta.get("until", False)) else 0.0
    pred_sign = _pred_sign(pred)
    has_pred_uid = 1.0 if isinstance(meta.get("pred_uid", None), str) else 0.0

    if role != "outside_face":
        face = 0.0
    if role != "event_time":
        dwell_a = 0.0
        dwell_b = 0.0
    if role not in ("event_time", "always_or_choice"):
        opt = 0.0

    return [t0, opt, dwell_a, dwell_b, face, until_flag, pred_sign, has_pred_uid]


def _temporal_block(
    *,
    T: int,
    tau: int,
    a_or_t: float,
    b_or_aux: float,
    span: float,
    midpoint: float,
) -> List[float]:
    slack_to_T = float(T) - float(b_or_aux)
    tau_norm = (float(tau) / float(max(T, 1))) if int(T) > 0 else 0.0
    return [float(a_or_t), float(b_or_aux), float(span), float(midpoint), float(slack_to_T), float(tau_norm)]


def _feat(
    *,
    kind: str,
    op: Optional[str],
    pred: Optional[str],
    ent: Optional[str],
    ent_name: str,
    rect: Optional[List[float]],
    x0: List[float],
    T: int,
    tau: int,
    n_obs: int,
    n_regions: int,
    name: str,
    role: str = "",
    time_field: Any = None,
    meta: Optional[Dict[str, Any]] = None,
) -> List[float]:
    meta = meta if isinstance(meta, dict) else {}

    kind_oh = [0.0, 0.0, 0.0, 0.0]
    kind_map = {"op": 0, "pred": 1, "ent": 2, "b": 3}
    if kind not in kind_map:
        raise ValueError(f"Unknown kind={kind}")
    kind_oh[kind_map[kind]] = 1.0

    op_oh = [0.0] * 5
    op_map = {"And": 0, "Or": 1, "Always": 2, "Eventually": 3, "Until": 4}
    if op in op_map:
        op_oh[op_map[op]] = 1.0

    pred_oh = [0.0] * 2
    pred_map = {"InRect": 0, "OutsideRect": 1}
    if pred in pred_map:
        pred_oh[pred_map[pred]] = 1.0

    ent_super_oh = [0.0] * 2
    ent_map = {"Obstacle": 0, "Region": 1}
    if ent in ent_map:
        ent_super_oh[ent_map[ent]] = 1.0

    ent_role_oh = _entity_role_onehot(ent, ent_name)
    bin_role_oh = _binary_role_onehot(str(role))

    geom = _rect_stats(rect, x0)

    if kind == "op":
        a_v = _to_float(meta.get("a", 0), 0.0)
        b_v = _to_float(meta.get("b", 0), 0.0)
        span = max(0.0, b_v - a_v)
        midpoint = 0.5 * (a_v + b_v)
        temporal = _temporal_block(T=T, tau=tau, a_or_t=a_v, b_or_aux=b_v, span=span, midpoint=midpoint)
    elif kind == "b":
        t0 = _to_float(meta.get("t", meta.get("k", time_field if time_field is not None else -1)), -1.0)
        aux = _to_float(meta.get("opt", meta.get("face", meta.get("h", 0))), 0.0)
        span = _to_float(meta.get("dwell_b", 0), 0.0) - _to_float(meta.get("dwell_a", 0), 0.0)
        midpoint = 0.5 * (
            _to_float(meta.get("dwell_a", 0), 0.0) + _to_float(meta.get("dwell_b", 0), 0.0)
        )
        temporal = _temporal_block(T=T, tau=tau, a_or_t=t0, b_or_aux=aux, span=span, midpoint=midpoint)
    else:
        temporal = _temporal_block(T=T, tau=tau, a_or_t=0.0, b_or_aux=0.0, span=0.0, midpoint=0.0)

    binary = _binary_scalar_block(role=str(role), pred=pred, meta=meta, time_field=time_field) if kind == "b" else [0.0] * 8

    global_block = [
        float(T),
        float(tau),
        float(n_obs),
        float(n_regions),
        float(x0[0]),
        float(x0[1]),
        float(x0[2]),
        float(x0[3]),
    ]

    nh = _stable_hash8(name)

    feat = kind_oh + op_oh + pred_oh + ent_super_oh + ent_role_oh + bin_role_oh + geom + temporal + binary + global_block + nh
    if len(feat) != X_DIM:
        raise RuntimeError(f"Feature length mismatch: got {len(feat)} expected {X_DIM}")
    return feat


def _get_desc_t0(meta: Dict[str, Any], time_field: Any) -> int:
    if "t" in meta:
        try:
            return int(meta["t"])
        except Exception:
            return -1
    if "k" in meta:
        try:
            return int(meta["k"])
        except Exception:
            return -1
    if time_field is not None:
        try:
            return int(time_field)
        except Exception:
            return -1
    return -1


def _entity_name_from_binary_meta(meta: Dict[str, Any]) -> str:
    ent_name = meta.get("ent_name", None)
    pred_name = meta.get("pred", None)
    if isinstance(ent_name, str):
        return ent_name
    if isinstance(pred_name, str):
        return pred_name
    return ""


def _compute_knn_edges_from_rects(rects_by_idx: Dict[int, List[float]], knn_k: int) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    if int(knn_k) <= 0:
        return edges

    node_ids = sorted(rects_by_idx.keys())
    if len(node_ids) < 2:
        return edges

    for i_idx in node_ids:
        r_i = rects_by_idx[i_idx]
        cx_i = 0.5 * (float(r_i[0]) + float(r_i[1]))
        cy_i = 0.5 * (float(r_i[2]) + float(r_i[3]))

        dists: List[Tuple[float, int]] = []
        for j_idx in node_ids:
            if j_idx == i_idx:
                continue
            r_j = rects_by_idx[j_idx]
            cx_j = 0.5 * (float(r_j[0]) + float(r_j[1]))
            cy_j = 0.5 * (float(r_j[2]) + float(r_j[3]))
            d = (cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2
            dists.append((float(d), j_idx))

        dists.sort(key=lambda t: t[0])
        k_eff = min(int(knn_k), len(dists))
        for _, j_idx in dists[:k_eff]:
            edges.append((i_idx, j_idx))
            edges.append((j_idx, i_idx))

    return _dedup_pairs(edges)


@dataclass(frozen=True)
class VarGraphSpec:
    x_dim: int = X_DIM
    knn_k: int = 7


def build_hetero_graph_from_record(rec: Dict[str, Any], spec: VarGraphSpec = VarGraphSpec()):
    """
    Build a heterogeneous PyG HeteroData graph with explicit node/edge types.

      Node types: "op", "pred", "ent", "b"

      Edge types:
        ("op",   "ast",          "op")
        ("op",   "ast",          "pred")
        ("pred", "refers_to",    "ent")

        ("b",    "to_op",        "op")
        ("b",    "to_pred",      "pred")
        ("b",    "to_ent",       "ent")

        ("op",   "to_b",         "b")
        ("pred", "to_b",         "b")
        ("ent",  "to_b",         "b")

        ("ent",  "spatial",      "ent")
        ("b",    "time_next",    "b")

        ("pred", "ast_rev",      "op")
        ("ent",  "rev_refers_to","pred")

      Added graph-strengthening edges:
        ("b",    "same_context", "b")
        ("b",    "same_pred",    "b")
        ("b",    "same_ent",     "b")
        ("b",    "to_root_op",   "op")
        ("op",   "root_to_b",    "b")
    """
    sample = rec.get("sample", {})
    if not isinstance(sample, dict):
        raise ValueError("Record missing 'sample' dict.")
    T = int(sample.get("T", 0))
    env = _get_env(sample)

    x0 = _get_x0(env)
    tau = _get_tau(env)

    obstacles = _get_obstacles(env)
    regions = _get_regions(env)
    n_obs = int(len(obstacles))
    n_regions = int(len(regions))

    x_op: List[List[float]] = []
    x_pred: List[List[float]] = []
    x_ent: List[List[float]] = []
    x_b: List[List[float]] = []

    op_uid_to_i: Dict[str, int] = {}
    pred_uid_to_i: Dict[str, int] = {}
    pred_name_to_is: Dict[str, List[int]] = {}
    ent_name_to_i: Dict[str, int] = {}
    b_descid_to_i: Dict[str, int] = {}

    ent_rect_by_i: Dict[int, List[float]] = {}
    b_records: List[Dict[str, Any]] = []

    def _add_op(*, uid: Optional[str], op_type: str, a_val: float, b_val: float, name: str) -> int:
        i = len(x_op)
        x_op.append(
            _feat(
                kind="op",
                op=op_type,
                pred=None,
                ent=None,
                ent_name="",
                rect=None,
                x0=x0,
                T=T,
                tau=tau,
                n_obs=n_obs,
                n_regions=n_regions,
                name=name,
                meta={"a": float(a_val), "b": float(b_val)},
            )
        )
        if isinstance(uid, str):
            if uid in op_uid_to_i:
                raise ValueError(f"Duplicate op uid in AST dict (unexpected): {uid}")
            op_uid_to_i[uid] = i
        return i

    def _add_pred(*, uid: Optional[str], pred_type: str, pred_name: str, rect: Any, name: str) -> int:
        i = len(x_pred)
        x_pred.append(
            _feat(
                kind="pred",
                op=None,
                pred=pred_type,
                ent=None,
                ent_name=str(pred_name),
                rect=_rect_to_list(rect),
                x0=x0,
                T=T,
                tau=tau,
                n_obs=n_obs,
                n_regions=n_regions,
                name=name,
            )
        )
        if isinstance(uid, str):
            if uid in pred_uid_to_i:
                raise ValueError(f"Duplicate predicate uid in AST dict (unexpected): {uid}")
            pred_uid_to_i[uid] = i
        pred_name_to_is.setdefault(str(pred_name), []).append(i)
        return i

    def _add_ent(*, ent_kind: str, ent_name: str, rect: Any, name: str) -> int:
        rect_list = _rect_to_list(rect)
        i = len(x_ent)
        x_ent.append(
            _feat(
                kind="ent",
                op=None,
                pred=None,
                ent=ent_kind,
                ent_name=str(ent_name),
                rect=rect_list,
                x0=x0,
                T=T,
                tau=tau,
                n_obs=n_obs,
                n_regions=n_regions,
                name=name,
            )
        )
        if ent_name in ent_name_to_i:
            raise ValueError(f"Duplicate entity name in env (unexpected): {ent_name}")
        ent_name_to_i[str(ent_name)] = i
        ent_rect_by_i[i] = rect_list
        return i

    def _add_b(*, desc_id: str, role: str, time_field: Any, node_tag: Any, meta: Dict[str, Any]) -> int:
        if str(desc_id) in b_descid_to_i:
            raise ValueError(f"Duplicate binary descriptor id encountered (would overwrite): {desc_id}")

        ent_name = _entity_name_from_binary_meta(meta)
        pred_name = meta.get("pred", None)
        pred_type = None
        if isinstance(pred_name, str) and pred_name.startswith("O"):
            pred_type = "OutsideRect"

        i = len(x_b)
        x_b.append(
            _feat(
                kind="b",
                op=None,
                pred=pred_type,
                ent=None,
                ent_name=ent_name,
                rect=None,
                x0=x0,
                T=T,
                tau=tau,
                n_obs=n_obs,
                n_regions=n_regions,
                name=f"Bin:{desc_id}",
                role=str(role),
                time_field=time_field,
                meta=meta,
            )
        )
        b_descid_to_i[str(desc_id)] = i
        b_records.append(
            {
                "id": str(desc_id),
                "role": str(role),
                "time": time_field,
                "t0": int(_get_desc_t0(meta, time_field)),
                "node_tag": node_tag,
                "meta": meta if isinstance(meta, dict) else {},
            }
        )
        return i

    for j, r in enumerate(obstacles):
        name = f"O{j}"
        _add_ent(ent_kind="Obstacle", ent_name=name, rect=r, name=f"Obstacle:{name}")

    for k in sorted(regions.keys()):
        _add_ent(ent_kind="Region", ent_name=str(k), rect=regions[k], name=f"Region:{k}")

    ast = rec.get("ast", None)
    if not isinstance(ast, dict):
        raise ValueError("Record missing 'ast' dict (need ast_to_dict output).")

    e_op_ast_op: List[Tuple[int, int]] = []
    e_op_ast_pred: List[Tuple[int, int]] = []
    e_pred_ref_ent: List[Tuple[int, int]] = []
    e_pred_ast_rev_op: List[Tuple[int, int]] = []
    e_ent_rev_ref_pred: List[Tuple[int, int]] = []

    ast_uid_to_kind: Dict[str, str] = {}
    ast_uid_to_idx: Dict[str, int] = {}
    root_op_uid: Optional[str] = None

    def walk(node: Dict[str, Any], parent_uid: Optional[str]) -> str:
        nonlocal root_op_uid

        ntype = node.get("type", None)
        if not isinstance(ntype, str):
            raise ValueError(f"Bad AST node type: {node}")

        uid = node.get("uid", None)
        if not isinstance(uid, str):
            raise ValueError(f"AST node missing deterministic uid: type={ntype}, node={node}")

        if ntype in ("And", "Or"):
            me_i = _add_op(uid=uid, op_type=ntype, a_val=0.0, b_val=0.0, name=f"Op:{ntype}:{uid}")
            ast_uid_to_kind[uid] = "op"
            ast_uid_to_idx[uid] = me_i

            if parent_uid is None and root_op_uid is None:
                root_op_uid = uid

            if isinstance(parent_uid, str):
                if ast_uid_to_kind.get(parent_uid, None) != "op":
                    raise ValueError(f"AST parent is not op (unexpected): parent_uid={parent_uid}")
                pi = ast_uid_to_idx[parent_uid]
                e_op_ast_op.append((pi, me_i))
                e_op_ast_op.append((me_i, pi))

            args = node.get("args", [])
            if not isinstance(args, list):
                raise ValueError(f"{ntype}.args must be list, got {type(args)}")
            for c in args:
                if isinstance(c, dict):
                    walk(c, uid)
            return uid

        if ntype in ("Always", "Eventually", "Until"):
            a_i = _to_int_or_none(node.get("a", None))
            b_i = _to_int_or_none(node.get("b", None))
            me_i = _add_op(
                uid=uid,
                op_type=ntype,
                a_val=float(a_i if a_i is not None else 0),
                b_val=float(b_i if b_i is not None else 0),
                name=f"Op:{ntype}:{a_i}:{b_i}:{uid}",
            )
            ast_uid_to_kind[uid] = "op"
            ast_uid_to_idx[uid] = me_i

            if parent_uid is None and root_op_uid is None:
                root_op_uid = uid

            if isinstance(parent_uid, str):
                if ast_uid_to_kind.get(parent_uid, None) != "op":
                    raise ValueError(f"AST parent is not op (unexpected): parent_uid={parent_uid}")
                pi = ast_uid_to_idx[parent_uid]
                e_op_ast_op.append((pi, me_i))
                e_op_ast_op.append((me_i, pi))

            if ntype == "Until":
                phi = node.get("phi", None)
                psi = node.get("psi", None)
                if isinstance(phi, dict):
                    walk(phi, uid)
                if isinstance(psi, dict):
                    walk(psi, uid)
            else:
                phi = node.get("phi", None)
                if isinstance(phi, dict):
                    walk(phi, uid)
            return uid

        if ntype in ("InRect", "OutsideRect"):
            pred_name = str(node.get("name", ""))
            rect = node.get("rect", None)
            me_i = _add_pred(
                uid=uid,
                pred_type=ntype,
                pred_name=pred_name,
                rect=rect,
                name=f"Pred:{ntype}:{pred_name}",
            )
            ast_uid_to_kind[uid] = "pred"
            ast_uid_to_idx[uid] = me_i

            if isinstance(parent_uid, str):
                if ast_uid_to_kind.get(parent_uid, None) != "op":
                    raise ValueError(f"AST parent is not op (unexpected): parent_uid={parent_uid}")
                pi = ast_uid_to_idx[parent_uid]
                e_op_ast_pred.append((pi, me_i))
                e_pred_ast_rev_op.append((me_i, pi))

            if pred_name in ent_name_to_i:
                ei = ent_name_to_i[pred_name]
                e_pred_ref_ent.append((me_i, ei))
                e_ent_rev_ref_pred.append((ei, me_i))

            return uid

        raise NotImplementedError(f"Unsupported AST node type: {ntype}")

    _ = walk(ast, parent_uid=None)

    e_b_to_op: List[Tuple[int, int]] = []
    e_b_to_pred: List[Tuple[int, int]] = []
    e_b_to_ent: List[Tuple[int, int]] = []
    e_b_time_next_b: List[Tuple[int, int]] = []

    e_op_to_b: List[Tuple[int, int]] = []
    e_pred_to_b: List[Tuple[int, int]] = []
    e_ent_to_b: List[Tuple[int, int]] = []

    e_b_same_context_b: List[Tuple[int, int]] = []
    e_b_same_pred_b: List[Tuple[int, int]] = []
    e_b_same_ent_b: List[Tuple[int, int]] = []
    e_b_to_root_op: List[Tuple[int, int]] = []
    e_op_root_to_b: List[Tuple[int, int]] = []

    micp = rec.get("micp", {})
    descs = micp.get("descs", None) if isinstance(micp, dict) else None
    if descs is not None:
        if not isinstance(descs, list):
            raise ValueError("rec['micp']['descs'] must be a list of descriptor dicts.")

        for d in descs:
            if not isinstance(d, dict):
                continue
            did = d.get("id", "")
            role = d.get("role", "")
            node_tag = d.get("node_tag", "")
            meta = d.get("meta", {})
            time_field = d.get("time", None)
            if not isinstance(meta, dict):
                meta = {}
            _add_b(desc_id=str(did), role=str(role), time_field=time_field, node_tag=node_tag, meta=meta)

        groups_time: Dict[Tuple[str, str, str, str], List[Tuple[int, int]]] = {}
        groups_context: Dict[Tuple[str, str, str, str], List[int]] = {}
        groups_pred: Dict[str, List[int]] = {}
        groups_ent: Dict[str, List[int]] = {}

        root_op_i: Optional[int] = None
        if isinstance(root_op_uid, str):
            root_op_i = op_uid_to_i.get(root_op_uid, None)

        for b in b_records:
            did = b["id"]
            node_tag = b["node_tag"]
            meta = b["meta"] if isinstance(b["meta"], dict) else {}
            b_i = b_descid_to_i[did]

            if isinstance(node_tag, str) and node_tag in op_uid_to_i:
                op_i = op_uid_to_i[node_tag]
                e_b_to_op.append((b_i, op_i))
                e_op_to_b.append((op_i, b_i))

            pred_uid = meta.get("pred_uid", None)
            pred_name = meta.get("pred", None)
            ent_name = meta.get("ent_name", None)

            pred_i: Optional[int] = None
            if isinstance(pred_uid, str):
                if pred_uid not in pred_uid_to_i:
                    raise ValueError(f"Binary {did} references pred_uid not in AST: {pred_uid}")
                pred_i = pred_uid_to_i[pred_uid]
                e_b_to_pred.append((b_i, pred_i))
                e_pred_to_b.append((pred_i, b_i))
            elif isinstance(pred_name, str):
                cands = pred_name_to_is.get(pred_name, [])
                if len(cands) == 0:
                    raise ValueError(f"Binary {did} references pred name not in AST: {pred_name}")
                if len(cands) > 1:
                    raise ValueError(
                        f"Ambiguous binary-to-predicate attachment for {did}: pred='{pred_name}' matches {len(cands)} predicate nodes. "
                        "Provide meta['pred_uid'] in descriptor."
                    )
                pred_i = cands[0]
                e_b_to_pred.append((b_i, pred_i))
                e_pred_to_b.append((pred_i, b_i))

            target_ent = _entity_name_from_binary_meta(meta)
            if target_ent != "":
                if target_ent not in ent_name_to_i:
                    raise ValueError(f"Binary {did} references entity name not in env: {target_ent}")
                ent_i = ent_name_to_i[target_ent]
                e_b_to_ent.append((b_i, ent_i))
                e_ent_to_b.append((ent_i, b_i))

            if root_op_i is not None:
                e_b_to_root_op.append((b_i, root_op_i))
                e_op_root_to_b.append((root_op_i, b_i))

            role = str(b.get("role", ""))
            t0 = int(b.get("t0", -1))

            pred_key = ""
            if isinstance(pred_uid, str):
                pred_key = str(pred_uid)
            elif isinstance(pred_name, str):
                pred_key = str(pred_name)

            ent_key = ""
            if isinstance(ent_name, str):
                ent_key = str(ent_name)
            elif isinstance(pred_name, str):
                ent_key = str(pred_name)

            op_key = str(node_tag) if isinstance(node_tag, str) else ""
            gk = (role, op_key, pred_key, ent_key)

            if t0 >= 0:
                groups_time.setdefault(gk, []).append((t0, b_i))

            groups_context.setdefault(gk, []).append(b_i)

            if pred_key != "":
                groups_pred.setdefault(pred_key, []).append(b_i)

            if ent_key != "":
                groups_ent.setdefault(ent_key, []).append(b_i)

        for _, items in groups_time.items():
            items.sort(key=lambda p: p[0])
            for (_, a), (_, b) in zip(items[:-1], items[1:]):
                e_b_time_next_b.append((a, b))
                e_b_time_next_b.append((b, a))

        for _, nodes in groups_context.items():
            _add_bidirectional_clique(e_b_same_context_b, nodes)

        for _, nodes in groups_pred.items():
            _add_bidirectional_clique(e_b_same_pred_b, nodes)

        for _, nodes in groups_ent.items():
            _add_bidirectional_clique(e_b_same_ent_b, nodes)

    e_ent_spatial_ent = _compute_knn_edges_from_rects(ent_rect_by_i, int(spec.knn_k))

    e_op_ast_op = _dedup_pairs(e_op_ast_op)
    e_op_ast_pred = _dedup_pairs(e_op_ast_pred)
    e_pred_ref_ent = _dedup_pairs(e_pred_ref_ent)
    e_pred_ast_rev_op = _dedup_pairs(e_pred_ast_rev_op)
    e_ent_rev_ref_pred = _dedup_pairs(e_ent_rev_ref_pred)

    e_b_to_op = _dedup_pairs(e_b_to_op)
    e_b_to_pred = _dedup_pairs(e_b_to_pred)
    e_b_to_ent = _dedup_pairs(e_b_to_ent)
    e_b_time_next_b = _dedup_pairs(e_b_time_next_b)

    e_op_to_b = _dedup_pairs(e_op_to_b)
    e_pred_to_b = _dedup_pairs(e_pred_to_b)
    e_ent_to_b = _dedup_pairs(e_ent_to_b)

    e_b_same_context_b = _dedup_pairs(e_b_same_context_b)
    e_b_same_pred_b = _dedup_pairs(e_b_same_pred_b)
    e_b_same_ent_b = _dedup_pairs(e_b_same_ent_b)
    e_b_to_root_op = _dedup_pairs(e_b_to_root_op)
    e_op_root_to_b = _dedup_pairs(e_op_root_to_b)

    data = HeteroData()

    data["op"].x = torch.tensor(x_op, dtype=torch.float) if len(x_op) > 0 else torch.zeros((0, X_DIM), dtype=torch.float)
    data["pred"].x = torch.tensor(x_pred, dtype=torch.float) if len(x_pred) > 0 else torch.zeros((0, X_DIM), dtype=torch.float)
    data["ent"].x = torch.tensor(x_ent, dtype=torch.float) if len(x_ent) > 0 else torch.zeros((0, X_DIM), dtype=torch.float)
    data["b"].x = torch.tensor(x_b, dtype=torch.float) if len(x_b) > 0 else torch.zeros((0, X_DIM), dtype=torch.float)

    data[("op", "ast", "op")].edge_index = _edge_index_from_pairs(e_op_ast_op)
    data[("op", "ast", "pred")].edge_index = _edge_index_from_pairs(e_op_ast_pred)
    data[("pred", "refers_to", "ent")].edge_index = _edge_index_from_pairs(e_pred_ref_ent)

    data[("pred", "ast_rev", "op")].edge_index = _edge_index_from_pairs(e_pred_ast_rev_op)
    data[("ent", "rev_refers_to", "pred")].edge_index = _edge_index_from_pairs(e_ent_rev_ref_pred)

    data[("b", "to_op", "op")].edge_index = _edge_index_from_pairs(e_b_to_op)
    data[("b", "to_pred", "pred")].edge_index = _edge_index_from_pairs(e_b_to_pred)
    data[("b", "to_ent", "ent")].edge_index = _edge_index_from_pairs(e_b_to_ent)

    data[("op", "to_b", "b")].edge_index = _edge_index_from_pairs(e_op_to_b)
    data[("pred", "to_b", "b")].edge_index = _edge_index_from_pairs(e_pred_to_b)
    data[("ent", "to_b", "b")].edge_index = _edge_index_from_pairs(e_ent_to_b)

    data[("ent", "spatial", "ent")].edge_index = _edge_index_from_pairs(e_ent_spatial_ent)
    data[("b", "time_next", "b")].edge_index = _edge_index_from_pairs(e_b_time_next_b)

    data[("b", "same_context", "b")].edge_index = _edge_index_from_pairs(e_b_same_context_b)
    data[("b", "same_pred", "b")].edge_index = _edge_index_from_pairs(e_b_same_pred_b)
    data[("b", "same_ent", "b")].edge_index = _edge_index_from_pairs(e_b_same_ent_b)
    data[("b", "to_root_op", "op")].edge_index = _edge_index_from_pairs(e_b_to_root_op)
    data[("op", "root_to_b", "b")].edge_index = _edge_index_from_pairs(e_op_root_to_b)

    data.T = torch.tensor([T], dtype=torch.long)
    data.tau = torch.tensor([tau], dtype=torch.long)
    data.n_obs = torch.tensor([n_obs], dtype=torch.long)

    data["b"].node_desc_id_str = [br["id"] for br in b_records]
    data["b"].role_str = [br["role"] for br in b_records]

    return data


def validate_hetero_graph(data, *, strict: bool = True, max_print: int = 5, verbose: bool = False) -> None:
    """
    Validation checks:
      - feature dims per type
      - edge_index shape/range per relation
      - time_next symmetry
      - b role distribution + time_next breakdown
      - quick degree sanity for attachments
    """
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData, got {type(data)}")

    for ntype in ("op", "pred", "ent", "b"):
        if ntype not in data.node_types:
            raise ValueError(f"Missing node type: {ntype}")
        x = data[ntype].x
        if x.dim() != 2 or int(x.size(1)) != int(X_DIM):
            raise ValueError(f"{ntype}.x must be (N,{X_DIM}), got {tuple(x.shape)}")

    n_op = int(data["op"].x.size(0))
    n_pred = int(data["pred"].x.size(0))
    n_ent = int(data["ent"].x.size(0))
    n_b = int(data["b"].x.size(0))

    rels = [
        (("op", "ast", "op"), n_op, n_op),
        (("op", "ast", "pred"), n_op, n_pred),
        (("pred", "refers_to", "ent"), n_pred, n_ent),
        (("b", "to_op", "op"), n_b, n_op),
        (("b", "to_pred", "pred"), n_b, n_pred),
        (("b", "to_ent", "ent"), n_b, n_ent),
        (("ent", "spatial", "ent"), n_ent, n_ent),
        (("b", "time_next", "b"), n_b, n_b),
        (("op", "to_b", "b"), n_op, n_b),
        (("pred", "to_b", "b"), n_pred, n_b),
        (("ent", "to_b", "b"), n_ent, n_b),
        (("pred", "ast_rev", "op"), n_pred, n_op),
        (("ent", "rev_refers_to", "pred"), n_ent, n_pred),
        (("b", "same_context", "b"), n_b, n_b),
        (("b", "same_pred", "b"), n_b, n_b),
        (("b", "same_ent", "b"), n_b, n_b),
        (("b", "to_root_op", "op"), n_b, n_op),
        (("op", "root_to_b", "b"), n_op, n_b),
    ]

    def _check_edge(etype, n_src, n_dst):
        if etype not in data.edge_types:
            raise ValueError(f"Missing edge type: {etype}")
        ei = data[etype].edge_index
        if ei.dim() != 2 or ei.size(0) != 2:
            raise ValueError(f"{etype}.edge_index must be (2,E), got {tuple(ei.shape)}")
        if ei.numel() == 0:
            return 0
        s = ei[0]
        t = ei[1]
        if int(torch.min(s)) < 0 or int(torch.max(s)) >= int(n_src):
            raise ValueError(f"{etype}: src index out of range [0,{n_src})")
        if int(torch.min(t)) < 0 or int(torch.max(t)) >= int(n_dst):
            raise ValueError(f"{etype}: dst index out of range [0,{n_dst})")
        return int(ei.size(1))

    counts = {}
    for etype, ns, nd in rels:
        counts[etype] = _check_edge(etype, ns, nd)

    tn = data[("b", "time_next", "b")].edge_index
    pairs = set()
    if tn.numel() > 0:
        pairs = set((int(a), int(b)) for a, b in zip(tn[0].tolist(), tn[1].tolist()))
        bad = []
        for a, b in list(pairs)[: max_print * 100]:
            if (b, a) not in pairs:
                bad.append((a, b))
                if len(bad) >= max_print:
                    break
        if len(bad) > 0:
            raise ValueError(f"time_next is not symmetric for some edges (showing up to {max_print}): {bad}")

    def _min_out_degree(etype) -> int:
        ei = data[etype].edge_index
        if ei.numel() == 0:
            return 0
        src = ei[0]
        deg = torch.bincount(src, minlength=n_b)
        return int(torch.min(deg).item()) if deg.numel() > 0 else 0

    if strict and n_b > 0:
        for et in [
            ("b", "to_op", "op"),
            ("b", "to_pred", "pred"),
            ("b", "to_ent", "ent"),
        ]:
            mdeg = _min_out_degree(et)
            if mdeg < 1:
                raise ValueError(f"Some b nodes have no outgoing edges for {et}. min_out_degree={mdeg}")

    ids = []
    if hasattr(data["b"], "node_desc_id_str"):
        ids = list(data["b"].node_desc_id_str)

    role_counts: Dict[str, int] = {}
    if hasattr(data["b"], "role_str"):
        rs = list(data["b"].role_str)
        if len(rs) == n_b:
            for r in rs:
                r = str(r)
                role_counts[r] = role_counts.get(r, 0) + 1

    if verbose:
        print("[hetero] node counts:", {"op": n_op, "pred": n_pred, "ent": n_ent, "b": n_b})
        print("[hetero] edge counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")

        if len(role_counts) > 0:
            print("[hetero] b role counts:", role_counts)

        if len(ids) > 0:
            print("[hetero] b.node_desc_id_str sample:", ids[: min(max_print, len(ids))])

        if tn.numel() > 0:
            undirected = len(pairs) // 2
            print(f"[hetero] time_next directed edges={len(pairs)} (~undirected={undirected})")

            deg = torch.bincount(tn[0], minlength=n_b)
            n0 = int((deg == 0).sum().item())
            n1 = int((deg == 1).sum().item())
            n2 = int((deg == 2).sum().item())
            n3p = int((deg >= 3).sum().item())
            print("[hetero] time_next out-degree histogram:", {"0": n0, "1": n1, "2": n2, ">=3": n3p})


__all__ = ["X_DIM", "VarGraphSpec", "build_hetero_graph_from_record", "validate_hetero_graph"]