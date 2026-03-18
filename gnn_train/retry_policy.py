# gnn_train/retry_policy.py
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from gnn_train.topk_strategy import LocalJointTopKConfig, generate_local_joint_strategies
from gnn_train.qp_core import solve_qp_replay


_IN_TAG_RE = re.compile(r"^in_(?P<pred>.+)_t(?P<k>\d+)_")
_OUT_TAG_RE = re.compile(r"^outside_(?P<pred>.+)_t(?P<t>\d+)_h(?P<h>\d+)_")
_DWELL_IN_TAG_RE = re.compile(r"^in_(?P<pred>.+)_t(?P<k>\d+)_dwell(?:_|$)")


def _extract_infeasible_inrect_targets_from_iis(iis_names: List[str]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    seen = set()
    for nm in iis_names:
        m = _IN_TAG_RE.match(str(nm))
        if not m:
            continue
        pred = str(m.group("pred"))
        k = int(m.group("k"))
        key = (pred, k)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_infeasible_outside_targets_from_iis(iis_names: List[str]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    seen = set()
    for nm in iis_names:
        m = _OUT_TAG_RE.match(str(nm))
        if not m:
            continue
        pred = str(m.group("pred"))
        t = int(m.group("t"))
        h = int(m.group("h"))
        key = (pred, t, h)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_dwell_windows_from_iis(iis_names: List[str]) -> Dict[str, int]:
    ts_by_pred: Dict[str, List[int]] = {}
    for nm in iis_names:
        m = _DWELL_IN_TAG_RE.match(str(nm))
        if not m:
            continue
        pred = str(m.group("pred"))
        k = int(m.group("k"))
        ts_by_pred.setdefault(pred, []).append(int(k))

    out: Dict[str, int] = {}
    for pred, ts in ts_by_pred.items():
        if len(ts) == 0:
            continue
        out[str(pred)] = max(0, int(max(ts) - min(ts)))
    return out


def _copy_faces_override(strategy: Dict[str, int]) -> Dict[str, Dict[int, int]]:
    raw = strategy.get("__faces_override__", None)
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Dict[int, int]] = {}
    for pred, mp in raw.items():
        if not isinstance(mp, dict):
            continue
        out[str(pred)] = {}
        for t, h in mp.items():
            try:
                out[str(pred)][int(t)] = int(h)
            except Exception:
                continue
    return out


def _canonical_strategy_key(
    strategy: Dict[str, int],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...]]:
    active_ids = tuple(sorted(str(k) for k, v in strategy.items() if k != "__faces_override__" and int(v) == 1))

    ov = _copy_faces_override(strategy)
    ov_key: List[Tuple[str, Tuple[Tuple[int, int], ...]]] = []
    for pred in sorted(ov.keys()):
        items = tuple(sorted((int(t), int(h)) for t, h in ov[pred].items()))
        ov_key.append((str(pred), items))

    return active_ids, tuple(ov_key)


def solve_with_local_joint_topk(
    *,
    prob,
    K: int,
    base_strategy: Dict[str, int],
    descs: List[Any],
    bin_logits_by_id: Dict[str, torch.Tensor],
    build_constraints_fn: Callable[[Dict[str, int]], Tuple[List[Any], List[Any], Dict[str, Any]]],
    output_flag: int,
    compute_iis: bool,
    iis_limit: int,
    accept_fn: Optional[Callable[[Any], bool]] = None,
    accept_name: str = "ACCEPT",
    max_optimal_without_accept: int = 999999,
    repair_window: int = 2,
    max_repair_trials_per_attempt: int = 12,
    max_face_repairs_per_attempt: int = 16,
) -> Tuple[Any, Any, float, int, List[str], Dict[str, int], Dict[str, Any]]:
    if int(K) < 1:
        raise ValueError("K must be >= 1.")

    def _run_once(strategy: Dict[str, int]):
        in_rect, outside_face, aux = build_constraints_fn(strategy)
        X, U, t_qp, status_qp, iis_names = solve_qp_replay(
            prob=prob,
            in_rect=in_rect,
            outside_face=outside_face,
            output_flag=int(output_flag),
            compute_iis=bool(compute_iis),
            iis_limit=int(iis_limit),
        )
        return X, U, float(t_qp), int(status_qp), list(iis_names), aux

    def _is_accepted(status_qp: int, X) -> Optional[bool]:
        if accept_fn is None:
            return None
        if int(status_qp) != 2 or X is None:
            return False
        try:
            return bool(accept_fn(X))
        except Exception:
            return False

    if int(K) == 1:
        X, U, t_qp, status_qp, iis_names, aux = _run_once(dict(base_strategy))
        aux2 = dict(aux) if isinstance(aux, dict) else {}
        aux2["retry_log"] = [
            {
                "attempt": 0,
                "status": int(status_qp),
                "t_qp": float(t_qp),
                "n_iis": int(len(iis_names)),
                "accepted": _is_accepted(status_qp, X),
                "local_candidates_added": 0,
            }
        ]
        return X, U, float(t_qp), int(status_qp), list(iis_names), dict(base_strategy), aux2

    pending: List[Dict[str, int]] = [dict(base_strategy)]
    seen: Set[Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...]]] = {
        _canonical_strategy_key(dict(base_strategy))
    }

    retry_log: List[Dict[str, Any]] = []
    best_optimal = None
    last = None
    n_optimal_not_accepted = 0
    attempts = 0

    while len(pending) > 0 and int(attempts) < int(K):
        cand = pending.pop(0)

        X, U, t_qp, status_qp, iis_names, aux = _run_once(cand)
        accepted = _is_accepted(status_qp, X)

        rec: Dict[str, Any] = {
            "attempt": int(attempts),
            "status": int(status_qp),
            "t_qp": float(t_qp),
            "n_iis": int(len(iis_names)),
            "accepted": accepted,
            "local_candidates_added": 0,
        }
        retry_log.append(rec)

        last = (X, U, float(t_qp), int(status_qp), list(iis_names), cand, aux)

        if int(status_qp) == 2 and X is not None and best_optimal is None:
            best_optimal = last

        if accept_fn is not None:
            if int(status_qp) == 2 and X is not None:
                if bool(accepted):
                    aux2 = dict(aux) if isinstance(aux, dict) else {}
                    aux2["retry_log"] = retry_log
                    aux2["retry_accept_name"] = str(accept_name)
                    return X, U, float(t_qp), int(status_qp), list(iis_names), cand, aux2

                n_optimal_not_accepted += 1
                if int(n_optimal_not_accepted) >= int(max_optimal_without_accept):
                    break
        else:
            if int(status_qp) == 2 and X is not None:
                aux2 = dict(aux) if isinstance(aux, dict) else {}
                aux2["retry_log"] = retry_log
                return X, U, float(t_qp), int(status_qp), list(iis_names), cand, aux2

        remaining_budget = int(K) - int(attempts) - 1
        if remaining_budget <= 0:
            attempts += 1
            continue

        if int(status_qp) in (3, 4) and len(iis_names) > 0:
            culprit_events = _extract_infeasible_inrect_targets_from_iis(iis_names)
            culprit_faces = _extract_infeasible_outside_targets_from_iis(iis_names)

            only_in_conflict = len(culprit_events) > 0 and len(culprit_faces) == 0
            multi_event_in_conflict = bool(only_in_conflict and len(culprit_events) >= 2)

            dwell_windows = _extract_dwell_windows_from_iis(iis_names)
            has_dwell_conflict = len(dwell_windows) > 0

            effective_repair_window = int(repair_window)
            if multi_event_in_conflict:
                effective_repair_window = max(int(repair_window), min(8, 3 * int(repair_window)))

            local_budget = min(
                int(remaining_budget),
                max(1, int(max_repair_trials_per_attempt) + int(max_face_repairs_per_attempt)),
            )

            cfg = LocalJointTopKConfig(
                repair_window=int(effective_repair_window),
                max_event_alts=max(2, int(max_repair_trials_per_attempt)),
                max_face_alts=max(2, min(4, int(max_face_repairs_per_attempt))),
                beam_size=max(2, int(local_budget)),
                interval_conflict_penalty=12.0 if multi_event_in_conflict else 8.0,
                point_conflict_penalty=3.0 if multi_event_in_conflict else 2.0,
                forbid_interval_overlap=bool(multi_event_in_conflict and has_dwell_conflict),
                forbid_point_conflict=False,
                dwell_default_width=max(1, int(effective_repair_window)) if has_dwell_conflict else 1,
                event_window_by_pred=dict(dwell_windows),
            )

            local_candidates = generate_local_joint_strategies(
                descs=descs,
                bin_logits_by_id=bin_logits_by_id,
                base_strategy=cand,
                K=int(local_budget) + 1,
                culprit_event_targets=culprit_events,
                culprit_outside_targets=culprit_faces,
                config=cfg,
            )

            added_front: List[Dict[str, int]] = []
            for alt in local_candidates[1:]:
                key = _canonical_strategy_key(alt)
                if key in seen:
                    continue
                seen.add(key)
                added_front.append(alt)

            if len(added_front) > 0:
                pending = added_front + pending
                rec["local_candidates_added"] = int(len(added_front))

            rec["effective_repair_window"] = int(effective_repair_window)
            rec["multi_event_in_conflict"] = int(multi_event_in_conflict)
            rec["has_dwell_conflict"] = int(has_dwell_conflict)

        attempts += 1

    if best_optimal is not None:
        X, U, t_qp, status_qp, iis_names, strategy_best, aux = best_optimal
        aux2 = dict(aux) if isinstance(aux, dict) else {}
        aux2["retry_log"] = retry_log
        if accept_fn is not None:
            aux2["retry_accept_name"] = str(accept_name)
        return X, U, float(t_qp), int(status_qp), list(iis_names), strategy_best, aux2

    if last is not None:
        X, U, t_qp, status_qp, iis_names, strategy_last, aux = last
        aux2 = dict(aux) if isinstance(aux, dict) else {}
        aux2["retry_log"] = retry_log
        if accept_fn is not None:
            aux2["retry_accept_name"] = str(accept_name)
        return X, U, float(t_qp), int(status_qp), list(iis_names), strategy_last, aux2

    X, U, t_qp, status_qp, iis_names, aux = _run_once(dict(base_strategy))
    aux2 = dict(aux) if isinstance(aux, dict) else {}
    aux2["retry_log"] = retry_log
    if accept_fn is not None:
        aux2["retry_accept_name"] = str(accept_name)
    return X, U, float(t_qp), int(status_qp), list(iis_names), dict(base_strategy), aux2