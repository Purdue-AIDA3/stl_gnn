from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from cnn_train.cnn_model import CNNDescriptorScorer


@dataclass(frozen=True)
class MLPModelConfig:
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

    # Image specification
    img_channels: int = 6

    # Adaptive pooling before MLP
    pool_h: int = 16
    pool_w: int = 16

    # MLP encoder
    mlp_hidden1: int = 512
    mlp_hidden2: int = 512
    enc_dim: int = 256

    # Head
    head_hidden: int = 256


class MLPImageEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int,
        pool_h: int,
        pool_w: int,
        hidden1: int,
        hidden2: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.pool_h = int(pool_h)
        self.pool_w = int(pool_w)

        flat_dim = int(in_ch) * int(pool_h) * int(pool_w)

        self.pool = nn.AdaptiveAvgPool2d((int(pool_h), int(pool_w)))
        self.net = nn.Sequential(
            nn.Linear(flat_dim, int(hidden1)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden1), int(hidden2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden2), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be (B,C,H,W), got {tuple(x.shape)}")
        if int(x.size(1)) != int(self.in_ch):
            raise ValueError(
                f"x has C={int(x.size(1))}, but encoder expects in_ch={int(self.in_ch)}"
            )

        h = self.pool(x)
        h = h.reshape(int(h.size(0)), -1)
        z = self.net(h)
        return z


class MLPBinaryNet(nn.Module):
    def __init__(self, cfg: MLPModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = MLPImageEncoder(
            in_ch=int(cfg.img_channels),
            pool_h=int(cfg.pool_h),
            pool_w=int(cfg.pool_w),
            hidden1=int(cfg.mlp_hidden1),
            hidden2=int(cfg.mlp_hidden2),
            out_dim=int(cfg.enc_dim),
        )
        self.scorer = CNNDescriptorScorer(cfg)

    def forward(
        self,
        *,
        images: torch.Tensor,
        desc_batch_idx: torch.Tensor,
        desc_meta: List[Dict[str, Any]],
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must be (B,C,H,W), got {tuple(images.shape)}")
        if int(images.size(1)) != int(self.cfg.img_channels):
            raise ValueError(
                f"images has C={int(images.size(1))}, but cfg.img_channels={int(self.cfg.img_channels)}"
            )

        z = self.encoder(images)
        logit = self.scorer(
            img_z=z,
            desc_batch_idx=desc_batch_idx,
            desc_meta=desc_meta,
        )
        return logit