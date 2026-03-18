from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _stable_hash_to_int(s: str, mod: int) -> int:
    # Deterministic across runs/machines; do not use Python's built-in hash().
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    v = int(h[:8], 16)
    return int(v % int(mod))


@dataclass(frozen=True)
class CNNModelConfig:
    # Descriptor embedding sizes
    str_vocab: int = 65536
    str_emb_dim: int = 64

    role_vocab: int = 8
    role_emb_dim: int = 16

    # Numeric index vocab
    max_t: int = 256
    max_k: int = 256
    max_face: int = 8
    t_emb_dim: int = 16
    k_emb_dim: int = 16
    face_emb_dim: int = 8

    # CNN encoder
    img_channels: int = 6  # default channels from rasterizer cfg (can differ; checked at runtime)
    enc_dim: int = 256

    # Head
    head_hidden: int = 256


class CNNImageEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int) -> None:
        super().__init__()
        # Simple, stable conv stack + global avg pool
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        h = self.net(x)  # (B,256,h,w)
        h = h.mean(dim=(2, 3))  # global avg pool -> (B,256)
        z = self.proj(h)  # (B,out_dim)
        return z


class CNNDescriptorScorer(nn.Module):
    """
    Scores variable-length descriptor sets conditioned on image embedding.
    Descriptor identity/metadata is embedded using:
      - role string
      - pred string (meta['pred'] or meta['ent_name'])
      - op_uid string (meta['op_uid'])
      - node_tag string (desc['node_tag'])
      - numeric t/k/face (if present)
    """

    def __init__(self, cfg: CNNModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.role_emb = nn.Embedding(int(cfg.role_vocab), int(cfg.role_emb_dim))
        self.str_emb = nn.Embedding(int(cfg.str_vocab), int(cfg.str_emb_dim))

        self.t_emb = nn.Embedding(int(cfg.max_t), int(cfg.t_emb_dim))
        self.k_emb = nn.Embedding(int(cfg.max_k), int(cfg.k_emb_dim))
        self.face_emb = nn.Embedding(int(cfg.max_face), int(cfg.face_emb_dim))

        in_dim = int(cfg.enc_dim) + int(cfg.role_emb_dim) + 4 * int(cfg.str_emb_dim) + int(cfg.t_emb_dim) + int(cfg.k_emb_dim) + int(cfg.face_emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(cfg.head_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(cfg.head_hidden), 1),
        )

    def _role_to_idx(self, role: str) -> int:
        # Map known roles to small indices; keep stable.
        r = str(role)
        if r == "outside_face":
            return 1
        if r == "event_time":
            return 2
        if r == "always_or_choice":
            return 3
        return 0

    def _str_to_idx(self, s: str) -> int:
        return _stable_hash_to_int(str(s), int(self.cfg.str_vocab))

    def forward(
        self,
        *,
        img_z: torch.Tensor,  # (B,enc_dim)
        desc_batch_idx: torch.Tensor,  # (M_total,) long
        desc_meta: List[Dict[str, Any]],  # length M_total
    ) -> torch.Tensor:
        """
        Returns logits (M_total,).
        """
        if desc_batch_idx.ndim != 1:
            raise ValueError("desc_batch_idx must be 1D.")
        M = int(desc_batch_idx.numel())
        if len(desc_meta) != M:
            raise ValueError(f"desc_meta length mismatch: {len(desc_meta)} vs M={M}")

        z = img_z.index_select(0, desc_batch_idx)  # (M,enc_dim)

        role_idx = torch.tensor([self._role_to_idx(m.get("role", "")) for m in desc_meta], device=z.device, dtype=torch.long)
        role_e = self.role_emb(role_idx)  # (M,role_emb_dim)

        # String embeddings
        pred_s = [str(m.get("pred", m.get("ent_name", ""))) for m in desc_meta]
        op_s = [str(m.get("op_uid", "")) for m in desc_meta]
        node_tag_s = [str(m.get("node_tag", "")) for m in desc_meta]
        pred_uid_s = [str(m.get("pred_uid", "")) for m in desc_meta]

        pred_i = torch.tensor([self._str_to_idx(s) for s in pred_s], device=z.device, dtype=torch.long)
        op_i = torch.tensor([self._str_to_idx(s) for s in op_s], device=z.device, dtype=torch.long)
        nt_i = torch.tensor([self._str_to_idx(s) for s in node_tag_s], device=z.device, dtype=torch.long)
        pu_i = torch.tensor([self._str_to_idx(s) for s in pred_uid_s], device=z.device, dtype=torch.long)

        pred_e = self.str_emb(pred_i)
        op_e = self.str_emb(op_i)
        nt_e = self.str_emb(nt_i)
        pu_e = self.str_emb(pu_i)

        # Numeric embeddings (clamped)
        t_raw = [int(m.get("t", m.get("time", 0))) for m in desc_meta]
        k_raw = [int(m.get("k", m.get("time", 0))) for m in desc_meta]
        face_raw = [int(m.get("face", 0)) for m in desc_meta]

        t_idx = torch.tensor([max(0, min(int(self.cfg.max_t) - 1, v)) for v in t_raw], device=z.device, dtype=torch.long)
        k_idx = torch.tensor([max(0, min(int(self.cfg.max_k) - 1, v)) for v in k_raw], device=z.device, dtype=torch.long)
        f_idx = torch.tensor([max(0, min(int(self.cfg.max_face) - 1, v)) for v in face_raw], device=z.device, dtype=torch.long)

        t_e = self.t_emb(t_idx)
        k_e = self.k_emb(k_idx)
        f_e = self.face_emb(f_idx)

        feat = torch.cat([z, role_e, pred_e, op_e, nt_e, pu_e, t_e, k_e, f_e], dim=1)  # (M,in_dim)
        logit = self.mlp(feat).squeeze(1)  # (M,)
        return logit


class CNNBinaryNet(nn.Module):
    def __init__(self, cfg: CNNModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = CNNImageEncoder(int(cfg.img_channels), int(cfg.enc_dim))
        self.scorer = CNNDescriptorScorer(cfg)

    def forward(
        self,
        *,
        images: torch.Tensor,  # (B,C,H,W)
        desc_batch_idx: torch.Tensor,  # (M,)
        desc_meta: List[Dict[str, Any]],  # length M
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must be (B,C,H,W), got {tuple(images.shape)}")
        if images.size(1) != int(self.cfg.img_channels):
            raise ValueError(
                f"images has C={int(images.size(1))}, but cfg.img_channels={int(self.cfg.img_channels)}. "
                "Update CNNModelConfig.img_channels to match rasterizer output."
            )
        z = self.encoder(images)
        logit = self.scorer(img_z=z, desc_batch_idx=desc_batch_idx, desc_meta=desc_meta)
        return logit