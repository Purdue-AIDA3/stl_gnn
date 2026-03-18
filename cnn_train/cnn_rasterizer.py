from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch


Rect = Tuple[float, float, float, float]


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _rect_to_pixel_bbox(
    rect: Rect,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    W: int,
    H: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1, x2, y1, y2 = rect
    xmin, xmax = xlim
    ymin, ymax = ylim
    if xmax <= xmin or ymax <= ymin:
        return None

    # Normalize to [0,1]
    u1 = (x1 - xmin) / (xmax - xmin)
    u2 = (x2 - xmin) / (xmax - xmin)
    v1 = (y1 - ymin) / (ymax - ymin)
    v2 = (y2 - ymin) / (ymax - ymin)

    u_lo = _clip01(min(u1, u2))
    u_hi = _clip01(max(u1, u2))
    v_lo = _clip01(min(v1, v2))
    v_hi = _clip01(max(v1, v2))

    # Convert to pixel indices [0..W-1], [0..H-1]
    px1 = int(np.floor(u_lo * float(W)))
    px2 = int(np.ceil(u_hi * float(W))) - 1
    py1 = int(np.floor(v_lo * float(H)))
    py2 = int(np.ceil(v_hi * float(H))) - 1

    # Clamp
    px1 = max(0, min(W - 1, px1))
    px2 = max(0, min(W - 1, px2))
    py1 = max(0, min(H - 1, py1))
    py2 = max(0, min(H - 1, py2))

    if px2 < px1 or py2 < py1:
        return None
    return px1, px2, py1, py2


def _draw_rect(mask: torch.Tensor, bbox: Tuple[int, int, int, int], value: float = 1.0) -> None:
    px1, px2, py1, py2 = bbox
    # mask: (H,W) in row-major => y is row.
    mask[py1 : py2 + 1, px1 : px2 + 1] = float(value)


def _draw_gaussian_point(
    img: torch.Tensor,
    *,
    x: float,
    y: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    sigma_px: float,
) -> None:
    # img: (H,W)
    H, W = int(img.shape[0]), int(img.shape[1])
    xmin, xmax = xlim
    ymin, ymax = ylim
    if xmax <= xmin or ymax <= ymin:
        return

    u = (x - xmin) / (xmax - xmin)
    v = (y - ymin) / (ymax - ymin)
    u = _clip01(u)
    v = _clip01(v)

    cx = u * float(W - 1)
    cy = v * float(H - 1)

    yy = torch.arange(H, device=img.device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=img.device, dtype=torch.float32).view(1, W)

    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    s2 = float(max(1e-6, sigma_px * sigma_px))
    g = torch.exp(-0.5 * d2 / s2)
    img[:] = torch.maximum(img, g)


@dataclass(frozen=True)
class CNNRasterConfig:
    H: int = 64
    W: int = 64

    # Channels:
    # 0: obstacles occupancy
    # 1: goal regions occupancy (union of all env["regions"])
    # 2: start position (gaussian)
    # 3: vx0 constant channel
    # 4: vy0 constant channel
    # 5: T normalized constant channel
    use_velocity_channels: bool = True
    use_T_channel: bool = True
    start_sigma_px: float = 2.0


def rasterize_env_to_image(
    *,
    sample: Dict[str, Any],
    cfg: CNNRasterConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Builds a raster image tensor for CNN input from a sample dict.

    Expects:
      sample["T"] : int
      sample["bounds"]["x_min"][0:2], sample["bounds"]["x_max"][0:2] define x/y limits
      sample["env"]["x0"] : [x, y, vx, vy]
      sample["env"]["obstacles"] : list of rects [x1,x2,y1,y2]
      sample["env"]["regions"] : dict name -> rect [x1,x2,y1,y2]
    """
    T = int(sample.get("T", 0))
    bounds = sample.get("bounds", {})
    x_min = bounds.get("x_min", None)
    x_max = bounds.get("x_max", None)
    if not (isinstance(x_min, list) and isinstance(x_max, list) and len(x_min) >= 2 and len(x_max) >= 2):
        raise ValueError("sample['bounds'] must contain x_min/x_max with at least 2 entries (x,y).")

    xlim = (float(x_min[0]), float(x_max[0]))
    ylim = (float(x_min[1]), float(x_max[1]))

    env = sample.get("env", {})
    x0 = env.get("x0", None)
    if not (isinstance(x0, list) and len(x0) >= 4):
        raise ValueError("sample['env']['x0'] must be a list [x,y,vx,vy].")

    obstacles = env.get("obstacles", [])
    regions = env.get("regions", {})

    C = 3
    if bool(cfg.use_velocity_channels):
        C += 2
    if bool(cfg.use_T_channel):
        C += 1

    img = torch.zeros((C, int(cfg.H), int(cfg.W)), device=device, dtype=torch.float32)

    # Obstacles occupancy
    obs = img[0]
    for r in obstacles:
        if not (isinstance(r, list) and len(r) == 4):
            continue
        rect = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
        bbox = _rect_to_pixel_bbox(rect, xlim=xlim, ylim=ylim, W=int(cfg.W), H=int(cfg.H))
        if bbox is None:
            continue
        _draw_rect(obs, bbox, value=1.0)

    # Goal/region occupancy (union)
    reg = img[1]
    if isinstance(regions, dict):
        for _, r in regions.items():
            if not (isinstance(r, list) and len(r) == 4):
                continue
            rect = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            bbox = _rect_to_pixel_bbox(rect, xlim=xlim, ylim=ylim, W=int(cfg.W), H=int(cfg.H))
            if bbox is None:
                continue
            _draw_rect(reg, bbox, value=1.0)

    # Start position gaussian
    st = img[2]
    _draw_gaussian_point(
        st,
        x=float(x0[0]),
        y=float(x0[1]),
        xlim=xlim,
        ylim=ylim,
        sigma_px=float(cfg.start_sigma_px),
    )

    c = 3
    if bool(cfg.use_velocity_channels):
        vx0 = float(x0[2])
        vy0 = float(x0[3])
        img[c].fill_(vx0)
        img[c + 1].fill_(vy0)
        c += 2

    if bool(cfg.use_T_channel):
        # Normalize T to a reasonable scale. Keep simple and stable.
        # If T is typically <= 50, this keeps values in [0,1] for common cases.
        img[c].fill_(float(T) / 50.0)
        c += 1

    return img