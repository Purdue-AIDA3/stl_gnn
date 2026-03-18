# gnn_train/topk_strategy.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


@dataclass(frozen=True)
class LocalJointTopKConfig:
    repair_window: int = 2
    max_event_alts: Optional[int] = None
    max_face_alts: Optional[int] = 4
    beam_size: Optional[int] = None

    interval_conflict_penalty: float = 8.0
    point_conflict_penalty: float = 2.0
    forbid_interval_overlap: bool = False
    forbid_point_conflict: bool = False

    dwell_default_width: int = 1
    event_window_by_pred: Dict[str, int] = field(default_factory=dict)


def _to_logit_map(bin_logits_by_id: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for did, v in bin_logits_by_id.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"bin_logits_by_id[{did}] is not a Tensor.")
        out[str(did)] = float(v.detach().cpu().item())
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


def _canonical_key(
    strategy: Dict[str, int],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...]]:
    active_ids = tuple(sorted(str(k) for k, v in strategy.items() if k != "__faces_override__" and int(v) == 1))

    ov = _copy_faces_override(strategy)
    ov_key: List[Tuple[str, Tuple[Tuple[int, int], ...]]] = []
    for pred in sorted(ov.keys()):
        items = tuple(sorted((int(t), int(h)) for t, h in ov[pred].items()))
        ov_key.append((str(pred), items))
    return active_ids, tuple(ov_key)


def _set_event_choice(
    *,
    strategy: Dict[str, int],
    pred_to_ids: Dict[str, List[str]],
    pred: str,
    sid_new: str,
) -> Dict[str, int]:
    out = dict(strategy)
    for sid in pred_to_ids.get(str(pred), []):
        out[str(sid)] = 0
    out[str(sid_new)] = 1
    return out


def _set_face_override(
    *,
    strategy: Dict[str, int],
    pred: str,
    t: int,
    h_new: int,
) -> Dict[str, int]:
    out = dict(strategy)
    ov = _copy_faces_override(out)
    ov.setdefault(str(pred), {})
    ov[str(pred)][int(t)] = int(h_new)
    out["__faces_override__"] = ov
    return out


def _get_current_face(
    *,
    base_face_by_pred_t: Dict[Tuple[str, int], int],
    strategy: Dict[str, int],
    pred: str,
    t: int,
) -> Optional[int]:
    ov = _copy_faces_override(strategy)
    if str(pred) in ov and int(t) in ov[str(pred)]:
        return int(ov[str(pred)][int(t)])
    key = (str(pred), int(t))
    if key in base_face_by_pred_t:
        return int(base_face_by_pred_t[key])
    return None


def _event_kind(meta: Dict[str, Any]) -> str:
    if bool(meta.get("dwell_a", False)) or bool(meta.get("dwell_b", False)):
        return "interval"
    if bool(meta.get("until", False)):
        return "until"
    return "point"


def _event_width(*, pred: str, meta: Dict[str, Any], cfg: LocalJointTopKConfig) -> int:
    if str(pred) in cfg.event_window_by_pred:
        return max(0, int(cfg.event_window_by_pred[str(pred)]))

    if "window" in meta:
        try:
            return max(0, int(meta.get("window", 0)))
        except Exception:
            pass

    kind = _event_kind(meta)
    if kind == "interval":
        return max(0, int(cfg.dwell_default_width))
    return 0


def _event_interval(k: int, width: int) -> Tuple[int, int]:
    a = int(k)
    b = int(k) + max(0, int(width))
    return a, b


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _pair_event_penalty(
    *,
    k_new: int,
    width_new: int,
    pred_new: str,
    events_current: Dict[str, Tuple[int, int]],
    cfg: LocalJointTopKConfig,
) -> Optional[float]:
    a0, a1 = _event_interval(int(k_new), int(width_new))
    penalty = 0.0

    for pred_old, (k_old, width_old) in events_current.items():
        if str(pred_old) == str(pred_new):
            continue

        b0, b1 = _event_interval(int(k_old), int(width_old))
        if not _intervals_overlap(a0, a1, b0, b1):
            continue

        if max(int(width_new), int(width_old)) > 0:
            if bool(cfg.forbid_interval_overlap):
                return None
            penalty += float(cfg.interval_conflict_penalty)
        else:
            if bool(cfg.forbid_point_conflict):
                return None
            penalty += float(cfg.point_conflict_penalty)

    return float(penalty)


def generate_local_joint_strategies(
    *,
    descs: List[Any],
    bin_logits_by_id: Dict[str, torch.Tensor],
    base_strategy: Dict[str, int],
    K: int,
    culprit_event_targets: Optional[List[Tuple[str, int]]] = None,
    culprit_outside_targets: Optional[List[Tuple[str, int, int]]] = None,
    config: Optional[LocalJointTopKConfig] = None,
) -> List[Dict[str, int]]:
    if int(K) < 1:
        raise ValueError("K must be >= 1.")

    cfg = config if config is not None else LocalJointTopKConfig()
    beam_size = int(cfg.beam_size) if cfg.beam_size is not None else int(K)
    logit = _to_logit_map(bin_logits_by_id)

    pred_to_event: Dict[str, List[Tuple[int, str]]] = {}
    pred_to_ids: Dict[str, List[str]] = {}
    sid_to_meta: Dict[str, Dict[str, Any]] = {}
    sid_to_pred: Dict[str, str] = {}
    sid_to_k: Dict[str, int] = {}

    face_options: Dict[Tuple[str, int], List[Tuple[int, str]]] = {}
    base_face_by_pred_t: Dict[Tuple[str, int], int] = {}

    base_selected_events: Dict[str, Tuple[int, int]] = {}

    for d in descs:
        role = str(getattr(d, "role", ""))
        did = str(getattr(d, "id"))
        meta = dict(getattr(d, "meta", {}) or {})

        if role == "event_time":
            pred = str(meta.get("pred", ""))
            if pred == "":
                continue
            k = int(meta.get("k", 0))
            pred_to_event.setdefault(pred, []).append((int(k), did))
            pred_to_ids.setdefault(pred, []).append(did)
            sid_to_meta[did] = meta
            sid_to_pred[did] = pred
            sid_to_k[did] = int(k)

            if int(base_strategy.get(did, 0)) == 1:
                width = _event_width(pred=pred, meta=meta, cfg=cfg)
                base_selected_events[str(pred)] = (int(k), int(width))

        elif role == "outside_face":
            pred = str(meta.get("pred", ""))
            if pred == "":
                continue
            t = int(meta.get("t", meta.get("k", 0)))
            h = int(meta.get("face", meta.get("h", 0)))
            key = (str(pred), int(t))
            face_options.setdefault(key, []).append((int(h), did))
            if int(base_strategy.get(did, 0)) == 1:
                base_face_by_pred_t[key] = int(h)

    for pred in pred_to_event.keys():
        pred_to_event[pred].sort(key=lambda x: int(x[0]))

    event_targets = list(culprit_event_targets or [])
    outside_targets = list(culprit_outside_targets or [])

    if len(event_targets) == 0 and len(outside_targets) == 0:
        return [dict(base_strategy)]

    group_specs: List[Tuple[str, List[Tuple[float, Tuple[str, Any]]]]] = []

    seen_event_preds: Set[str] = set()
    for pred, k_cur in event_targets:
        pred = str(pred)
        if pred in seen_event_preds:
            continue
        seen_event_preds.add(pred)

        if pred not in pred_to_event:
            continue

        cur_sid = None
        cur_k = int(k_cur)
        for k_val, sid in pred_to_event[pred]:
            if int(base_strategy.get(str(sid), 0)) == 1:
                cur_sid = str(sid)
                cur_k = int(k_val)
                break

        cand: List[Tuple[float, Tuple[str, Any]]] = []
        for k_val, sid in pred_to_event[pred]:
            if abs(int(k_val) - int(cur_k)) > int(cfg.repair_window):
                continue

            meta = sid_to_meta.get(str(sid), {})
            width = _event_width(pred=pred, meta=meta, cfg=cfg)

            current_events = dict(base_selected_events)
            current_events.pop(str(pred), None)

            extra_pen = _pair_event_penalty(
                k_new=int(k_val),
                width_new=int(width),
                pred_new=str(pred),
                events_current=current_events,
                cfg=cfg,
            )
            if extra_pen is None:
                continue

            score = float(logit.get(str(sid), float("-inf"))) - float(extra_pen)
            cand.append((score, ("event", (pred, str(sid)))))

        if cur_sid is not None and all(op[1][1] != cur_sid for op in cand):
            meta = sid_to_meta.get(str(cur_sid), {})
            width = _event_width(pred=pred, meta=meta, cfg=cfg)

            current_events = dict(base_selected_events)
            current_events.pop(str(pred), None)

            extra_pen = _pair_event_penalty(
                k_new=int(cur_k),
                width_new=int(width),
                pred_new=str(pred),
                events_current=current_events,
                cfg=cfg,
            )
            if extra_pen is not None:
                score = float(logit.get(str(cur_sid), float("-inf"))) - float(extra_pen)
                cand.append((score, ("event", (pred, str(cur_sid)))))

        if len(cand) == 0:
            for k_val, sid in pred_to_event[pred]:
                meta = sid_to_meta.get(str(sid), {})
                width = _event_width(pred=pred, meta=meta, cfg=cfg)

                current_events = dict(base_selected_events)
                current_events.pop(str(pred), None)

                extra_pen = _pair_event_penalty(
                    k_new=int(k_val),
                    width_new=int(width),
                    pred_new=str(pred),
                    events_current=current_events,
                    cfg=cfg,
                )
                if extra_pen is None:
                    continue

                score = float(logit.get(str(sid), float("-inf"))) - float(extra_pen)
                cand.append((score, ("event", (pred, str(sid)))))

        cand.sort(key=lambda x: float(x[0]), reverse=True)

        max_event_alts = int(cfg.max_event_alts) if cfg.max_event_alts is not None else max(1, min(int(K), len(cand)))
        cand = cand[: max(1, min(max_event_alts, len(cand)))]
        if len(cand) > 0:
            group_specs.append((f"event:{pred}", cand))

    seen_face_keys: Set[Tuple[str, int]] = set()
    for pred, t, _h_cur in outside_targets:
        key = (str(pred), int(t))
        if key in seen_face_keys:
            continue
        seen_face_keys.add(key)

        if key not in face_options:
            continue

        cur_h = _get_current_face(
            base_face_by_pred_t=base_face_by_pred_t,
            strategy=base_strategy,
            pred=str(pred),
            t=int(t),
        )

        cand: List[Tuple[float, Tuple[str, Any]]] = []
        for h_val, sid in face_options[key]:
            cand.append((float(logit.get(str(sid), float("-inf"))), ("face", (str(pred), int(t), int(h_val)))))

        cand.sort(key=lambda x: float(x[0]), reverse=True)

        if cur_h is not None:
            has_cur = any(int(op[1][1][2]) == int(cur_h) for op in cand)
            if not has_cur:
                cand.append((0.0, ("face", (str(pred), int(t), int(cur_h)))))

        max_face_alts = int(cfg.max_face_alts) if cfg.max_face_alts is not None else len(cand)
        cand = cand[: max(1, min(max_face_alts, len(cand)))]
        if len(cand) > 0:
            group_specs.append((f"face:{pred}:{int(t)}", cand))

    if len(group_specs) == 0:
        return [dict(base_strategy)]

    @dataclass
    class _BeamState:
        score: float
        ops: List[Tuple[str, Any]]
        events_selected: Dict[str, Tuple[int, int]]

    beam: List[_BeamState] = [
        _BeamState(
            score=0.0,
            ops=[],
            events_selected=dict(base_selected_events),
        )
    ]

    for _group_name, options in group_specs:
        next_beam: List[_BeamState] = []

        for st in beam:
            for sc, op in options:
                op_kind, payload = op

                if str(op_kind) == "event":
                    pred, sid_new = payload
                    pred = str(pred)
                    sid_new = str(sid_new)

                    meta = sid_to_meta.get(sid_new, {})
                    k_new = int(sid_to_k.get(sid_new, 0))
                    width_new = _event_width(pred=pred, meta=meta, cfg=cfg)

                    current_events = dict(st.events_selected)
                    current_events.pop(pred, None)

                    extra_pen = _pair_event_penalty(
                        k_new=int(k_new),
                        width_new=int(width_new),
                        pred_new=pred,
                        events_current=current_events,
                        cfg=cfg,
                    )
                    if extra_pen is None:
                        continue

                    events2 = dict(st.events_selected)
                    events2[pred] = (int(k_new), int(width_new))

                    next_beam.append(
                        _BeamState(
                            score=float(st.score) + float(sc) - float(extra_pen),
                            ops=list(st.ops) + [op],
                            events_selected=events2,
                        )
                    )
                else:
                    next_beam.append(
                        _BeamState(
                            score=float(st.score) + float(sc),
                            ops=list(st.ops) + [op],
                            events_selected=dict(st.events_selected),
                        )
                    )

        next_beam.sort(key=lambda s: float(s.score), reverse=True)
        beam = next_beam[: max(1, int(beam_size))]

    candidates: List[Dict[str, int]] = [dict(base_strategy)]
    seen: Set[Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...]]] = {
        _canonical_key(dict(base_strategy))
    }

    for st in beam:
        strat = dict(base_strategy)
        for op_kind, payload in st.ops:
            if str(op_kind) == "event":
                pred, sid_new = payload
                strat = _set_event_choice(
                    strategy=strat,
                    pred_to_ids=pred_to_ids,
                    pred=str(pred),
                    sid_new=str(sid_new),
                )
            elif str(op_kind) == "face":
                pred, t, h_new = payload
                strat = _set_face_override(
                    strategy=strat,
                    pred=str(pred),
                    t=int(t),
                    h_new=int(h_new),
                )

        key = _canonical_key(strat)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(strat)
        if len(candidates) >= int(K):
            break

    return candidates