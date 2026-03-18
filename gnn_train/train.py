# gnn_train/train.py
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from .dataset import JsonlGzSTLDataset
from .model import STLBinaryNet, MODEL_DEFAULTS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def binary_bce_unweighted(
    bin_logits: torch.Tensor,
    bin_y: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> torch.Tensor:
    if bin_logits.ndim != 1:
        raise ValueError(f"bin_logits must be 1D (sum_nodes,), got shape={tuple(bin_logits.shape)}")
    if bin_y.ndim != 1:
        raise ValueError(f"bin_y must be 1D (sum_nodes,), got shape={tuple(bin_y.shape)}")
    if supervised_mask.ndim != 1:
        raise ValueError(f"supervised_mask must be 1D (sum_nodes,), got shape={tuple(supervised_mask.shape)}")
    if bin_logits.shape[0] != bin_y.shape[0] or bin_logits.shape[0] != supervised_mask.shape[0]:
        raise ValueError(
            f"Shape mismatch: bin_logits={tuple(bin_logits.shape)} bin_y={tuple(bin_y.shape)} "
            f"mask={tuple(supervised_mask.shape)}"
        )

    per = nn.functional.binary_cross_entropy_with_logits(bin_logits, bin_y, reduction="none")
    per = per * supervised_mask.float()
    denom = supervised_mask.sum().clamp_min(1).float()
    return per.sum() / denom


@torch.no_grad()
def binary_accuracy(
    bin_logits: torch.Tensor,
    bin_y: torch.Tensor,
    supervised_mask: torch.Tensor,
    thr: float = 0.5,
) -> float:
    n = int(supervised_mask.sum().item())
    if n <= 0:
        return float("nan")
    pred = (torch.sigmoid(bin_logits) >= float(thr)).float()
    correct = (pred == bin_y).float() * supervised_mask.float()
    return float(correct.sum().item() / max(1.0, float(n)))


def _parse_outside_face_id(s: str) -> Optional[Tuple[str, int, int]]:
    parts = str(s).split(":")
    if len(parts) < 4:
        return None

    t = None
    h = None
    pred = parts[-1]

    for p in reversed(parts[:-1]):
        if t is None and len(p) >= 2 and p[0] == "t" and p[1:].isdigit():
            t = int(p[1:])
            continue
        if h is None and len(p) >= 2 and p[0] == "h" and p[1:].isdigit():
            h = int(p[1:])
            continue
        if t is not None and h is not None:
            break

    if t is None or h is None:
        return None
    return str(pred), int(t), int(h)


@torch.no_grad()
def outside_face_switch_penalty(
    *,
    batch,
    bin_logits: torch.Tensor,
    bin_y: torch.Tensor,
    supervised_mask: torch.Tensor,
    penalty_weight: float,
) -> torch.Tensor:
    w = float(penalty_weight)
    if w <= 0.0:
        return bin_logits.new_zeros(())

    if "b" not in getattr(batch, "node_types", []):
        return bin_logits.new_zeros(())
    b = batch["b"]
    if not hasattr(b, "role_str") or not hasattr(b, "node_desc_id_str"):
        return bin_logits.new_zeros(())

    roles = list(getattr(b, "role_str"))
    ids = list(getattr(b, "node_desc_id_str"))
    nb = int(bin_logits.numel())
    if len(roles) != nb or len(ids) != nb:
        return bin_logits.new_zeros(())

    by_pred_t: Dict[str, Dict[int, Dict[int, int]]] = {}
    for i in range(nb):
        if not bool(supervised_mask[i].item()):
            continue
        if str(roles[i]) != "outside_face":
            continue
        parsed = _parse_outside_face_id(str(ids[i]))
        if parsed is None:
            continue
        pred, t, h = parsed
        if h not in (0, 1, 2, 3) or t < 0:
            continue
        by_pred_t.setdefault(pred, {}).setdefault(t, {})[h] = int(i)

    total = bin_logits.new_zeros(())
    count = 0

    for _, tmap in by_pred_t.items():
        ts = sorted(tmap.keys())
        if len(ts) < 2:
            continue
        for j in range(1, len(ts)):
            t0 = int(ts[j - 1])
            t1 = int(ts[j])
            if t1 != t0 + 1:
                continue

            h0 = tmap.get(t0, {})
            h1 = tmap.get(t1, {})
            if any(h not in h0 for h in (0, 1, 2, 3)):
                continue
            if any(h not in h1 for h in (0, 1, 2, 3)):
                continue

            idx0 = torch.tensor([h0[0], h0[1], h0[2], h0[3]], device=bin_logits.device, dtype=torch.long)
            idx1 = torch.tensor([h1[0], h1[1], h1[2], h1[3]], device=bin_logits.device, dtype=torch.long)

            p0 = torch.softmax(bin_logits.index_select(0, idx0), dim=0)
            p1 = torch.softmax(bin_logits.index_select(0, idx1), dim=0)

            y0 = bin_y.index_select(0, idx0)
            y1 = bin_y.index_select(0, idx1)
            gt0 = int(torch.argmax(y0).item())
            gt1 = int(torch.argmax(y1).item())
            gt_switch = 1 if gt0 != gt1 else 0

            switch_prob = 1.0 - torch.dot(p0, p1)
            total = total + (switch_prob if gt_switch == 0 else (1.0 - switch_prob))
            count += 1

    if count <= 0:
        return bin_logits.new_zeros(())
    return (total / float(count)) * float(w)


@torch.no_grad()
def summarize_hetero_batch_graph(batch) -> str:
    if not hasattr(batch, "edge_index_dict"):
        return "no_edge_index_dict"

    eid = batch.edge_index_dict
    nd = batch.num_nodes_dict if hasattr(batch, "num_nodes_dict") else {}

    def ecount(et):
        ei = eid.get(et, None)
        return int(ei.size(1)) if ei is not None else 0

    def indeg_to_b(et):
        ei = eid.get(et, None)
        if ei is None:
            return 0.0, 0.0
        dst = ei[1]
        nb = int(nd.get("b", 0))
        if nb <= 0:
            return 0.0, 0.0
        deg = torch.bincount(dst, minlength=nb).float()
        return float(deg.mean().item()), float(deg.max().item())

    et_op_b = ("op", "to_b", "b")
    et_pred_b = ("pred", "to_b", "b")
    et_ent_b = ("ent", "to_b", "b")
    et_ref = ("pred", "refers_to", "ent")
    et_spat = ("ent", "spatial", "ent")
    et_tnext = ("b", "time_next", "b")

    nb = int(nd.get("b", 0))
    npred = int(nd.get("pred", 0))
    nent = int(nd.get("ent", 0))
    nop = int(nd.get("op", 0))

    e_op_b = ecount(et_op_b)
    e_pred_b = ecount(et_pred_b)
    e_ent_b = ecount(et_ent_b)
    e_ref = ecount(et_ref)
    e_spat = ecount(et_spat)
    e_tnext = ecount(et_tnext)

    m_op_b, x_op_b = indeg_to_b(et_op_b)
    m_pred_b, x_pred_b = indeg_to_b(et_pred_b)
    m_ent_b, x_ent_b = indeg_to_b(et_ent_b)

    sup = "sup=?"
    if "b" in getattr(batch, "node_types", []):
        if hasattr(batch["b"], "bin_supervised_mask"):
            mask = batch["b"].bin_supervised_mask.view(-1).bool()
            sup = f"sup={int(mask.sum().item())}/{int(mask.numel())}"

    return (
        f"nodes(op={nop},pred={npred},ent={nent},b={nb}) "
        f"edges(op->b={e_op_b},pred->b={e_pred_b},ent->b={e_ent_b},ref={e_ref},spat={e_spat},tnext={e_tnext}) "
        f"b_indeg_mean(op={m_op_b:.2f},pred={m_pred_b:.2f},ent={m_ent_b:.2f}) "
        f"b_indeg_max(op={x_op_b:.0f},pred={x_pred_b:.0f},ent={x_ent_b:.0f}) "
        f"{sup}"
    )


@torch.no_grad()
def make_summary_figure(
    *,
    out_path: str,
    history: Dict[str, List[float]],
    model: STLBinaryNet,
    ds: JsonlGzSTLDataset,
    viz_idx: int,
    device: torch.device,
    thr: float,
) -> None:
    _ensure_dir(out_path)

    if viz_idx < 0:
        viz_idx = 0

    ex_id = int(ds.example_id) if hasattr(ds, "example_id") else -1

    one = ds.get(viz_idx).to(device)
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([one])

    model.eval()
    bin_logits = model(batch).detach().cpu().view(-1)

    bin_y = batch["b"].bin_y.detach().cpu().float().view(-1)
    mask = batch["b"].bin_supervised_mask.detach().cpu().bool().view(-1)

    if bin_logits.numel() != bin_y.numel() or bin_logits.numel() != mask.numel():
        raise RuntimeError(
            f"Summary shape mismatch: bin_logits={int(bin_logits.numel())} "
            f"bin_y={int(bin_y.numel())} mask={int(mask.numel())}"
        )

    n_sup = int(mask.sum().item())
    acc = binary_accuracy(bin_logits, bin_y, mask, thr=thr)
    loss = float(binary_bce_unweighted(bin_logits, bin_y, mask).detach().cpu().item())

    idxs = torch.nonzero(mask, as_tuple=False).view(-1)
    if idxs.numel() > 0:
        p = torch.sigmoid(bin_logits[idxs]).numpy()
        y = bin_y[idxs].numpy()
        order = np.argsort(-p)
        p = p[order]
        y = y[order]
    else:
        p = np.zeros((0,), dtype=np.float32)
        y = np.zeros((0,), dtype=np.float32)

    epochs = np.arange(1, len(history.get("loss", [])) + 1)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    if len(epochs) > 0:
        ax0.plot(epochs, history.get("loss", []), marker="o", linewidth=1)
        ax0.plot(epochs, history.get("loss_bin", []), marker="o", linewidth=1)
        ax0.plot(epochs, history.get("acc_bin", []), marker="o", linewidth=1)
        ax0.legend(["total", "bin_bce", "acc_bin"])

    ax0.set_title("Train curves (binary-node supervision)")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("value")

    ax1.plot(np.arange(len(p)), p, marker="o", linewidth=1)
    if len(y) > 0:
        ax1.plot(np.arange(len(y)), y, marker="x", linewidth=0)
        ax1.legend(["pred sigmoid", "GT (0/1)"])

    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(f"Sample {viz_idx} (example_id={ex_id}): supervised={n_sup}, loss={loss:.4f}, acc@{thr}={acc:.3f}")
    ax1.set_xlabel("supervised nodes (sorted by pred prob)")
    ax1.set_ylabel("value")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[summary] wrote: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="results/train.jsonl.gz")
    p.add_argument("--example_id", type=int, default=1, help="train only this example_id (1..)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--layers", type=int, default=MODEL_DEFAULTS.layers)
    p.add_argument("--hidden", type=int, default=MODEL_DEFAULTS.hidden_dim)
    p.add_argument("--dropout", type=float, default=MODEL_DEFAULTS.dropout_p)

    p.add_argument("--viz_idx", type=int, default=-1, help="sample index for binary-node summary fig")
    p.add_argument("--viz_thr", type=float, default=0.5)
    p.add_argument("--summary_out", type=str, default="results/summary.png")

    p.add_argument("--ckpt_out", type=str, default="", help="if set, write a single final checkpoint .pt")
    p.add_argument(
        "--switch_penalty",
        type=float,
        default=0.1,
        help="outside_face temporal switch penalty weight (0 disables)",
    )

    args = p.parse_args()

    ex_id = int(args.example_id)

    ds = JsonlGzSTLDataset(args.data, example_id=ex_id)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = STLBinaryNet(hidden_dim=args.hidden, dropout_p=args.dropout, layers=args.layers).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history: Dict[str, List[float]] = {"loss": [], "loss_bin": [], "acc_bin": []}

    model.train()
    for ep in range(args.epochs):
        sum_loss = 0.0
        sum_bin = 0.0
        sum_acc = 0.0

        n_used_batches = 0
        n_skipped_batches = 0

        for batch in dl:
            batch = batch.to(device)

            if n_used_batches < 1 and ep == 0:
                print("[graph]", summarize_hetero_batch_graph(batch))

            if "b" not in batch.node_types:
                raise RuntimeError("Hetero batch missing 'b' node type.")
            if (not hasattr(batch["b"], "bin_y")) or (not hasattr(batch["b"], "bin_supervised_mask")):
                raise RuntimeError("Batch['b'] missing bin_y/bin_supervised_mask.")

            bin_y = batch["b"].bin_y.view(-1).float()
            mask = batch["b"].bin_supervised_mask.view(-1).bool()

            n_sup = int(mask.sum().item())
            if n_sup <= 0:
                n_skipped_batches += 1
                continue

            bin_logits = model(batch)
            if bin_logits.dim() != 1:
                raise RuntimeError("Model must return 1D logits for b nodes.")
            if bin_logits.shape[0] != bin_y.shape[0]:
                raise RuntimeError(
                    f"bin_logits/bin_y length mismatch: {int(bin_logits.shape[0])} vs {int(bin_y.shape[0])}"
                )
            if bin_logits.shape[0] != mask.shape[0]:
                raise RuntimeError(
                    f"bin_logits/mask length mismatch: {int(bin_logits.shape[0])} vs {int(mask.shape[0])}"
                )

            loss_bin = binary_bce_unweighted(bin_logits, bin_y, mask)
            loss_sw = outside_face_switch_penalty(
                batch=batch,
                bin_logits=bin_logits,
                bin_y=bin_y,
                supervised_mask=mask,
                penalty_weight=float(args.switch_penalty),
            )
            loss = loss_bin + loss_sw

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            acc = binary_accuracy(bin_logits.detach(), bin_y.detach(), mask.detach(), thr=float(args.viz_thr))

            sum_loss += float(loss.item())
            sum_bin += float(loss_bin.item())
            sum_acc += float(acc) if not (isinstance(acc, float) and np.isnan(acc)) else 0.0

            n_used_batches += 1

        if n_used_batches <= 0:
            raise RuntimeError(
                "No usable batches in this epoch (all batches had 0 supervised nodes). "
                "This means your dataset/graph_builder is not producing supervised binary nodes."
            )

        avg_loss = sum_loss / max(1, n_used_batches)
        avg_bin = sum_bin / max(1, n_used_batches)
        avg_acc = sum_acc / max(1, n_used_batches)

        history["loss"].append(avg_loss)
        history["loss_bin"].append(avg_bin)
        history["acc_bin"].append(avg_acc)

        print(
            f"epoch {ep+1:03d} "
            f"loss={avg_loss:.6f} (bin_bce={avg_bin:.6f}) "
            f"acc_bin@{float(args.viz_thr):.2f}={avg_acc:.3f} "
            f"(used_batches={n_used_batches}, skipped_batches={n_skipped_batches})"
        )

    ckpt_out = args.ckpt_out.strip()
    if not ckpt_out:
        ckpt_out = f"results/ckpt_ex{ex_id}.pt"

    _ensure_dir(ckpt_out)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_hparams": {
                "hidden_dim": args.hidden,
                "layers": args.layers,
                "dropout_p": args.dropout,
            },
            "trained_example_id": int(ex_id),
        },
        ckpt_out,
    )
    print(f"[ckpt] wrote: {ckpt_out}")

    model.eval()
    make_summary_figure(
        out_path=args.summary_out,
        history=history,
        model=model,
        ds=ds,
        viz_idx=args.viz_idx,
        device=device,
        thr=float(args.viz_thr),
    )


if __name__ == "__main__":
    main()