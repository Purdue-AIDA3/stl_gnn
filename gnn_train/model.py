from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv


@dataclass(frozen=True)
class ModelDefaults:
    hidden_dim: int = 256
    layers: int = 3
    dropout_p: float = 0.05


MODEL_DEFAULTS = ModelDefaults()


class STLBinaryNet(nn.Module):
    """
    Heterogeneous GNN for STL binary prediction.
    Outputs logits ONLY for 'b' node type.

    Patch:
    - Configurable number of hetero conv layers (layers >= 1).
    - Per-layer per-node-type LayerNorm.
    - Residual connections (lazy projections).
    - Feature dropout between layers.
    - Graph-strengthening relations for binary context propagation.
    """

    def __init__(
        self,
        hidden_dim: int = MODEL_DEFAULTS.hidden_dim,
        dropout_p: float = MODEL_DEFAULTS.dropout_p,
        layers: int = MODEL_DEFAULTS.layers,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout_p)
        self.layers = int(layers)
        if self.layers < 1:
            raise ValueError(f"layers must be >= 1, got {self.layers}")

        self.dropout = nn.Dropout(p=self.dropout_p)

        # Keep original relations and add only graph-strengthening relations.
        self._rel_specs: List[Tuple[str, str, str]] = [
            ("op", "ast", "op"),
            ("op", "ast", "pred"),
            ("pred", "refers_to", "ent"),
            ("b", "to_op", "op"),
            ("b", "to_pred", "pred"),
            ("b", "to_ent", "ent"),
            ("op", "to_b", "b"),
            ("pred", "to_b", "b"),
            ("ent", "to_b", "b"),
            ("ent", "spatial", "ent"),
            ("b", "time_next", "b"),
            ("pred", "ast_rev", "op"),
            ("ent", "rev_refers_to", "pred"),
            ("b", "same_context", "b"),
            ("b", "same_pred", "b"),
            ("b", "same_ent", "b"),
            ("b", "to_root_op", "op"),
            ("op", "root_to_b", "b"),
        ]

        def make_hetero_conv() -> HeteroConv:
            conv_dict = {}
            for et in self._rel_specs:
                conv_dict[et] = SAGEConv((-1, -1), self.hidden_dim)
            return HeteroConv(conv_dict, aggr="sum")

        self.convs = nn.ModuleList([make_hetero_conv() for _ in range(self.layers)])

        def make_norm() -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    "op": nn.LayerNorm(self.hidden_dim),
                    "pred": nn.LayerNorm(self.hidden_dim),
                    "ent": nn.LayerNorm(self.hidden_dim),
                    "b": nn.LayerNorm(self.hidden_dim),
                }
            )

        def make_res_proj() -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    "op": nn.LazyLinear(self.hidden_dim),
                    "pred": nn.LazyLinear(self.hidden_dim),
                    "ent": nn.LazyLinear(self.hidden_dim),
                    "b": nn.LazyLinear(self.hidden_dim),
                }
            )

        self.norms = nn.ModuleList([make_norm() for _ in range(self.layers)])
        self.res_projs = nn.ModuleList([make_res_proj() for _ in range(self.layers)])

        self.b_classifier = nn.Linear(self.hidden_dim, 1)

    def _norm_res_relu(
        self,
        x_new: Dict[str, torch.Tensor],
        x_in: Dict[str, torch.Tensor],
        norm: nn.ModuleDict,
        proj: nn.ModuleDict,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply: LayerNorm -> Residual add (projected) -> ReLU.
        If a node type is missing from x_new, keep projected residual only.
        """
        out: Dict[str, torch.Tensor] = {}
        keys = set(x_in.keys()) | set(x_new.keys())
        for k in keys:
            res = x_in.get(k, None)
            msg = x_new.get(k, None)

            if msg is None:
                if res is None:
                    continue
                out[k] = F.relu(proj[k](res)) if k in proj else F.relu(res)
                continue

            if k in norm:
                msg = norm[k](msg)

            if res is not None and k in proj:
                msg = msg + proj[k](res)

            out[k] = F.relu(msg)
        return out

    def forward(self, data):
        x_dict: Dict[str, torch.Tensor] = data.x_dict
        edge_index_dict = data.edge_index_dict

        for li in range(self.layers):
            x_in = x_dict
            x_dict = self.convs[li](x_dict, edge_index_dict)
            x_dict = self._norm_res_relu(x_dict, x_in, self.norms[li], self.res_projs[li])

            if li != self.layers - 1:
                x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        b_emb = x_dict["b"]
        logits = self.b_classifier(b_emb).view(-1)
        return logits