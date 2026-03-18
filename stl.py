# stl.py
"""
Signal Temporal Logic (STL) utilities for a restricted fragment:
- Predicates: InRect, OutsideRect
- Boolean: And, Or
- Temporal: Eventually (F), Always (G), Until (U)

Also includes:
- Plotting helpers for rectangles and trajectories.
- Strategy represention (strategic binaries only) and JSON persistence.
- Binary extractor for supported fragment.

PATCH NOTES (semantic + graph correctness):
- Added deterministic per-AST-node unique identity (uid) based on structural path:
    uid = f"{node.key()}@{path}"  where root path is "0"
- STLBinaryExtractor now attaches BinaryDescriptor.node_tag to the EXACT enclosing
  temporal operator instance uid (or predicate uid if no enclosing op exists).
- BinaryDescriptor.meta now includes:
    op_uid, pred_uid, ent_name
  so graph_builder can attach binaries deterministically without ambiguous name matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

from geometry import Rect


# ============================================================
# 1) STL AST (Abstract Syntax Tree) nodes
# ============================================================

class STLNode:
    def children(self) -> List["STLNode"]:
        return []

    def key(self) -> str:
        raise NotImplementedError


# ============================================================
# AST unique identity helpers
# ============================================================

def _uid_for(node: "STLNode", path: str) -> str:
    '''
    Deterministic unique identity for an AST node instance.

    - `path` is a structural path such as "0/1/2" (root is "0").
    - We include node.key() so uids remain readable while still unique per instance.
    - Stable as long as AST structure (child ordering) is stable.
    '''
    return f"{node.key()}@{path}"


def compute_uid_maps(root: "STLNode") -> Tuple[Dict[int, str], Dict[int, str]]:
    '''
    Returns:
      - id(node) -> uid for ALL nodes
      - id(node) -> path for ALL nodes (path only, without key)

    This is useful for deterministic attachment of BinaryDescriptors to exact AST instances.
    '''
    uid_map: Dict[int, str] = {}
    path_map: Dict[int, str] = {}

    def walk(n: STLNode, path: str) -> None:
        uid_map[id(n)] = _uid_for(n, path)
        path_map[id(n)] = path

        if isinstance(n, (And, Or)):
            for j, c in enumerate(n.args):
                walk(c, f"{path}/{j}")
            return
        if isinstance(n, (Always, Eventually)):
            walk(n.phi, f"{path}/0")
            return
        if isinstance(n, Until):
            walk(n.phi, f"{path}/0")
            walk(n.psi, f"{path}/1")
            return
        # predicates have no children

    walk(root, "0")
    return uid_map, path_map


@dataclass(frozen=True)
class InRect(STLNode):
    name: str
    rect: Rect

    def key(self) -> str:
        return f"InRect({self.name})"


@dataclass(frozen=True)
class OutsideRect(STLNode):
    name: str
    rect: Rect

    def key(self) -> str:
        return f"OutsideRect({self.name})"


@dataclass(frozen=True)
class And(STLNode):
    args: Tuple[STLNode, ...]

    def children(self) -> List[STLNode]:
        return list(self.args)

    def key(self) -> str:
        return "And"


@dataclass(frozen=True)
class Or(STLNode):
    args: Tuple[STLNode, ...]

    def children(self) -> List[STLNode]:
        return list(self.args)

    def key(self) -> str:
        return "Or"


@dataclass(frozen=True)
class Eventually(STLNode):
    a: int
    b: int
    phi: STLNode

    def children(self) -> List[STLNode]:
        return [self.phi]

    def key(self) -> str:
        return f"F[{self.a},{self.b}]"


@dataclass(frozen=True)
class Always(STLNode):
    a: int
    b: int
    phi: STLNode

    def children(self) -> List[STLNode]:
        return [self.phi]

    def key(self) -> str:
        return f"G[{self.a},{self.b}]"


@dataclass(frozen=True)
class Until(STLNode):
    a: int
    b: int
    phi: STLNode    # Left
    psi: STLNode    # Right (trigger)

    def children(self) -> List[STLNode]:
        return [self.phi, self.psi]

    def key(self) -> str:
        return f"U[{self.a},{self.b}]"


# ============================================================
# 2) Plot helpers
# ============================================================

def plot_environment(ax, ast: STLNode) -> None:
    """Draw rectangles referenced in the AST (obstacles + regions)."""

    def walk(node: STLNode) -> None:
        if isinstance(node, (InRect, OutsideRect)):
            xmin, xmax, ymin, ymax = node.rect
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                "k--" if isinstance(node, InRect) else "k-",
                linewidth=1.5,
            )
        for c in node.children():
            walk(c)

    walk(ast)


def plot_trajectory(
    X,
    ast: STLNode,
    filename: str,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_environment(ax, ast)
    ax.plot(X[:, 0], X[:, 1], "-o", linewidth=2)
    ax.plot(X[0, 0], X[0, 1], "go", label="start")
    ax.plot(X[-1, 0], X[-1, 1], "ro", label="end")

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ============================================================
# 3) Binary descriptors + strategy container
# ============================================================

@dataclass
class BinaryDescriptor:
    id: str
    role: str               # "event_time", "outside_face", "always_or_choice"
    node_tag: str           # uid of enclosing temporal op instance (or predicate uid if none)
    time: Optional[int]     # time index if applicable
    meta: Dict[str, Any]


@dataclass
class STLStrategy:
    T: int
    binaries: Dict[str, int]  # binary id -> value (0 or 1)
    meta: Dict[str, Any]      # optional extra info

    def save(self, path: str) -> None:
        obj = {"T": int(self.T), "binaries": self.binaries, "meta": self.meta}
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def load(path: str) -> "STLStrategy":
        with open(path, "r") as f:
            obj = json.load(f)
        return STLStrategy(
            T=int(obj["T"]),
            binaries={k: int(v) for k, v in obj["binaries"].items()},
            meta=obj.get("meta", {}),
        )


# ============================================================
# 4) Strategic binary extractor
# ============================================================

class STLBinaryExtractor:
    """
    Extract strategic binaries that must be saved to replay as QP:
    - Eventually / Until trigger times: event_time binaries
    - OutsideRect disjunction faces: outside_face binaries
    - Always(Or(InRect(...))) per-time region choices: always_or_choice binaries
    - Until requires outside-face binaries for the "unsafe" predicate before trigger time

    Patched requirements:
      - Each AST operator instance is identified by a deterministic uid.
      - BinaryDescriptor.node_tag is the uid of the EXACT temporal operator instance
        the descriptor belongs to (or, if there is no enclosing temporal operator,
        the uid of the predicate instance).
      - BinaryDescriptor.meta carries:
          * op_uid: enclosing temporal operator uid (if any)
          * pred_uid: uid of the predicate instance (if any)
          * ent_name: entity/region name the predicate refers to (if any)
    """

    def __init__(self, T: int):
        self.T = int(T)
        self.descs: List[BinaryDescriptor] = []
        self._uid_map: Dict[int, str] = {}
        self._path_map: Dict[int, str] = {}

    def extract(self, node: STLNode) -> List[BinaryDescriptor]:
        self.descs = []
        self._uid_map, self._path_map = compute_uid_maps(node)
        self._walk(node, context={"enclosing_op_uid": None}, path="0")
        return self.descs

    def _uid(self, n: STLNode, path: str) -> str:
        # Prefer the precomputed map (by object identity). Fall back to recomputation if needed.
        return self._uid_map.get(id(n), _uid_for(n, path))

    def _walk(self, node: STLNode, context: Dict[str, Any], path: str) -> None:
        if isinstance(node, And):
            for j, c in enumerate(node.args):
                self._walk(c, context, f"{path}/{j}")
            return

        if isinstance(node, Or):
            for j, c in enumerate(node.args):
                self._walk(c, context, f"{path}/{j}")
            return

        if isinstance(node, Always):
            op_uid = self._uid(node, path)
            phi = node.phi

            # Under Always, if child is OR of InRect, we need to save per-time choices.
            if isinstance(phi, Or) and all(isinstance(a, InRect) for a in phi.args):
                or_path = f"{path}/0"
                for k in range(int(node.a), int(node.b) + 1):
                    for i, pred in enumerate(phi.args):
                        pred_path = f"{or_path}/{i}"
                        pred_uid = self._uid(pred, pred_path)
                        ent_name = str(pred.name)

                        bid = f"always_or:{op_uid}:k{k}:opt{i}:{ent_name}"
                        self.descs.append(
                            BinaryDescriptor(
                                id=bid,
                                role="always_or_choice",
                                node_tag=op_uid,
                                time=k,
                                meta={
                                    "k": int(k),
                                    "opt": int(i),
                                    "pred": ent_name,
                                    "pred_uid": pred_uid,
                                    "ent_name": ent_name,
                                    "op_uid": op_uid,
                                },
                            )
                        )
                return

            # Otherwise recurse (OutsideRect faces are discovered inside OutsideRect).
            self._walk(phi, context={"enclosing_op_uid": op_uid}, path=f"{path}/0")
            return

        if isinstance(node, Eventually):
            op_uid = self._uid(node, path)
            a, b = int(node.a), int(node.b)

            # Support: F[a,b] ( Or( Always(a2,b2, InRect(T1)), Always(a2,b2, InRect(T2)), ... ) )
            # Used for "Either-Or" dwell tasks where the Always interval is relative to the trigger time.
            if isinstance(node.phi, Or) and all(isinstance(c, Always) and isinstance(c.phi, InRect) for c in node.phi.args):
                a2 = int(node.phi.args[0].a)
                b2 = int(node.phi.args[0].b)
                for c in node.phi.args:
                    if int(c.a) != a2 or int(c.b) != b2:
                        raise NotImplementedError("Eventually(Or(Always(...))) requires identical Always intervals across options.")

                or_path = f"{path}/0"
                for k in range(a, b + 1):
                    for i, alw in enumerate(node.phi.args):
                        # alw is Always; its child is InRect
                        pred = alw.phi
                        pred_path = f"{or_path}/{i}/0"
                        pred_uid = self._uid(pred, pred_path)
                        ent_name = str(pred.name)

                        bid = f"event:{op_uid}:k{k}:opt{i}:{ent_name}"
                        self.descs.append(
                            BinaryDescriptor(
                                id=bid,
                                role="event_time",
                                node_tag=op_uid,
                                time=k,
                                meta={
                                    "k": int(k),
                                    "opt": int(i),
                                    "pred": ent_name,
                                    "pred_uid": pred_uid,
                                    "ent_name": ent_name,
                                    "dwell_a": int(a2),
                                    "dwell_b": int(b2),
                                    "op_uid": op_uid,
                                },
                            )
                        )
                return

            if isinstance(node.phi, Or) and all(isinstance(c, InRect) for c in node.phi.args):
                or_path = f"{path}/0"
                for k in range(a, b + 1):
                    for i, pred in enumerate(node.phi.args):
                        pred_path = f"{or_path}/{i}"
                        pred_uid = self._uid(pred, pred_path)
                        ent_name = str(pred.name)

                        bid = f"event:{op_uid}:k{k}:opt{i}:{ent_name}"
                        self.descs.append(
                            BinaryDescriptor(
                                id=bid,
                                role="event_time",
                                node_tag=op_uid,
                                time=k,
                                meta={
                                    "k": int(k),
                                    "opt": int(i),
                                    "pred": ent_name,
                                    "pred_uid": pred_uid,
                                    "ent_name": ent_name,
                                    "op_uid": op_uid,
                                },
                            )
                        )
                return

            if isinstance(node.phi, InRect):
                pred = node.phi
                pred_path = f"{path}/0"
                pred_uid = self._uid(pred, pred_path)
                ent_name = str(pred.name)

                for k in range(a, b + 1):
                    bid = f"event:{op_uid}:k{k}:{ent_name}"
                    self.descs.append(
                        BinaryDescriptor(
                            id=bid,
                            role="event_time",
                            node_tag=op_uid,
                            time=k,
                            meta={
                                "k": int(k),
                                "pred": ent_name,
                                "pred_uid": pred_uid,
                                "ent_name": ent_name,
                                "op_uid": op_uid,
                            },
                        )
                    )
                return

            raise NotImplementedError("Eventually supports InRect, Or(InRect,...), or Or(Always(InRect),...).")

        if isinstance(node, Until):
            if not isinstance(node.phi, OutsideRect):
                raise NotImplementedError("Until left side must be OutsideRect(D) in this project.")
            if not isinstance(node.psi, InRect):
                raise NotImplementedError("Until right side must be InRect(K) in this project.")

            op_uid = self._uid(node, path)

            # Trigger selection for psi
            psi = node.psi
            psi_path = f"{path}/1"
            psi_uid = self._uid(psi, psi_path)
            ent_name_psi = str(psi.name)

            for k in range(int(node.a), int(node.b) + 1):
                bid = f"event:{op_uid}:k{k}:{ent_name_psi}"
                self.descs.append(
                    BinaryDescriptor(
                        id=bid,
                        role="event_time",
                        node_tag=op_uid,
                        time=k,
                        meta={
                            "k": int(k),
                            "pred": ent_name_psi,
                            "pred_uid": psi_uid,
                            "ent_name": ent_name_psi,
                            "until": True,
                            "op_uid": op_uid,
                        },
                    )
                )

            # Outside-face binaries needed for pre-trigger safety certification
            phi = node.phi
            phi_path = f"{path}/0"
            phi_uid = self._uid(phi, phi_path)
            ent_name_phi = str(phi.name)

            for t in range(0, self.T + 1):
                for h in range(4):
                    bid = f"outside_face:{op_uid}:t{t}:h{h}:{ent_name_phi}"
                    self.descs.append(
                        BinaryDescriptor(
                            id=bid,
                            role="outside_face",
                            node_tag=op_uid,
                            time=t,
                            meta={
                                "t": int(t),
                                "face": int(h),
                                "pred": ent_name_phi,
                                "pred_uid": phi_uid,
                                "ent_name": ent_name_phi,
                                "until_support": True,
                                "op_uid": op_uid,
                            },
                        )
                    )
            return

        if isinstance(node, OutsideRect):
            pred_uid = self._uid(node, path)
            ent_name = str(node.name)
            enclosing_op_uid = context.get("enclosing_op_uid", None)
            node_tag = str(enclosing_op_uid) if enclosing_op_uid is not None else pred_uid

            for k in range(0, self.T + 1):
                for h in range(4):
                    bid = f"outside_face:{node_tag}:t{k}:h{h}:{ent_name}"
                    self.descs.append(
                        BinaryDescriptor(
                            id=bid,
                            role="outside_face",
                            node_tag=node_tag,
                            time=k,
                            meta={
                                "t": int(k),
                                "face": int(h),
                                "pred": ent_name,
                                "pred_uid": pred_uid,
                                "ent_name": ent_name,
                                "op_uid": (str(enclosing_op_uid) if enclosing_op_uid is not None else None),
                            },
                        )
                    )
            return

        if isinstance(node, InRect):
            # InRect alone does not require strategic binaries.
            return

        raise NotImplementedError(f"Unsupported STL node type: {type(node)}")


__all__ = [
    # AST
    "STLNode",
    "InRect",
    "OutsideRect",
    "And",
    "Or",
    "Eventually",
    "Always",
    "Until",
    # Plot
    "plot_environment",
    "plot_trajectory",
    # Strategy
    "BinaryDescriptor",
    "STLStrategy",
    "STLBinaryExtractor",
]