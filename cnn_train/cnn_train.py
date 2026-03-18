from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .cnn_dataset import (
    CNNJsonlGzSTLDataset,
    cnn_collate_records,
    decompress_gz_to_temp_jsonl,
)
from .cnn_model import CNNBinaryNet, CNNModelConfig


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


@torch.no_grad()
def compute_pos_weight(ds: CNNJsonlGzSTLDataset, *, max_items: Optional[int] = None) -> float:
    n_pos = 0
    n_neg = 0
    N = len(ds) if max_items is None else min(len(ds), int(max_items))
    for i in range(N):
        rec = ds[i]
        y = rec.y
        n_pos += int((y > 0.5).sum().item())
        n_neg += int((y <= 0.5).sum().item())
    if n_pos <= 0:
        return 1.0
    return float(n_neg / max(1, n_pos))


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str, required=True)
    p.add_argument("--val_data", type=str, default="")

    p.add_argument("--example_id", type=int, default=None)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)

    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)

    p.add_argument("--enc_dim", type=int, default=256)
    p.add_argument("--head_hidden", type=int, default=256)

    p.add_argument("--ckpt_out", type=str, default="results/cnn_ckpt.pt")

    args = p.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    from .cnn_rasterizer import CNNRasterConfig

    raster_cfg = CNNRasterConfig(H=int(args.H), W=int(args.W))

    train_plain_path: Optional[str] = None
    val_plain_path: Optional[str] = None

    try:
        train_plain_path = decompress_gz_to_temp_jsonl(str(args.data))
        if str(args.val_data).strip() != "":
            val_plain_path = decompress_gz_to_temp_jsonl(str(args.val_data))

        ds_train = CNNJsonlGzSTLDataset(
            str(train_plain_path),
            example_id=(int(args.example_id) if args.example_id is not None else None),
            raster_cfg=raster_cfg,
        )

        ds_val = None
        if val_plain_path is not None:
            ds_val = CNNJsonlGzSTLDataset(
                str(val_plain_path),
                example_id=(int(args.example_id) if args.example_id is not None else None),
                raster_cfg=raster_cfg,
            )

        c_img = int(ds_train[0].img.size(0))

        cfg = CNNModelConfig(
            img_channels=int(c_img),
            enc_dim=int(args.enc_dim),
            head_hidden=int(args.head_hidden),
        )
        model = CNNBinaryNet(cfg).to(device)

        pos_weight = compute_pos_weight(ds_train, max_items=2000)
        bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32)
        )

        opt = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

        dl_train = DataLoader(
            ds_train,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=cnn_collate_records,
        )

        dl_val = None
        if ds_val is not None:
            dl_val = DataLoader(
                ds_val,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=cnn_collate_records,
            )

        for ep in range(int(args.epochs)):
            model.train()
            total_loss = 0.0
            total_n = 0
            total_correct = 0

            for batch in dl_train:
                images = batch["images"].to(device)
                y = batch["flat_y"].to(device)
                desc_batch_idx = batch["desc_batch_idx"].to(device)
                desc_meta = batch["flat_desc_meta"]

                logit = model(
                    images=images,
                    desc_batch_idx=desc_batch_idx,
                    desc_meta=desc_meta,
                )
                loss = bce(logit, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_loss += float(loss.item()) * int(y.numel())
                total_n += int(y.numel())

                pred = (torch.sigmoid(logit) >= 0.5).float()
                total_correct += int((pred == y).sum().item())

            tr_loss = total_loss / max(1, total_n)
            tr_acc = float(total_correct) / max(1, total_n)

            va_loss = float("nan")
            va_acc = float("nan")
            if dl_val is not None:
                model.eval()
                vtot = 0.0
                vn = 0
                vcorr = 0
                with torch.no_grad():
                    for batch in dl_val:
                        images = batch["images"].to(device)
                        y = batch["flat_y"].to(device)
                        desc_batch_idx = batch["desc_batch_idx"].to(device)
                        desc_meta = batch["flat_desc_meta"]

                        logit = model(
                            images=images,
                            desc_batch_idx=desc_batch_idx,
                            desc_meta=desc_meta,
                        )
                        loss = bce(logit, y)

                        vtot += float(loss.item()) * int(y.numel())
                        vn += int(y.numel())

                        pred = (torch.sigmoid(logit) >= 0.5).float()
                        vcorr += int((pred == y).sum().item())

                va_loss = vtot / max(1, vn)
                va_acc = float(vcorr) / max(1, vn)

            print(
                f"[cnn_train] epoch={ep+1}/{int(args.epochs)} "
                f"train_loss={tr_loss:.6f} train_acc={tr_acc:.4f} "
                f"val_loss={va_loss:.6f} val_acc={va_acc:.4f}"
            )

        out_ckpt = str(args.ckpt_out)
        _ensure_dir(out_ckpt)
        torch.save(
            {
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "raster_cfg": raster_cfg.__dict__,
                "pos_weight": float(pos_weight),
            },
            out_ckpt,
        )
        print(f"[cnn_train] saved ckpt to: {out_ckpt}")

    finally:
        for pth in [train_plain_path, val_plain_path]:
            if pth is None:
                continue
            if str(pth).endswith(".jsonl") and os.path.isfile(pth):
                try:
                    os.remove(pth)
                    print(f"[cnn_train] removed temp file: {pth}")
                except OSError:
                    pass


if __name__ == "__main__":
    main()