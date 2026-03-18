from __future__ import annotations

import gzip
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .cnn_rasterizer import CNNRasterConfig, rasterize_env_to_image


def decompress_gz_to_temp_jsonl(path: str) -> str:
    """
    Decompress .jsonl.gz to a temporary plain .jsonl file and return its path.

    Caller owns the lifecycle of the returned file and should delete it later.
    If `path` is already a plain file, it is returned as-is.
    """
    if not str(path).endswith(".gz"):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return str(path)

    fd, tmp_path = tempfile.mkstemp(prefix="cnn_jsonl_", suffix=".jsonl")
    os.close(fd)

    with gzip.open(path, "rb") as fin, open(tmp_path, "wb") as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)

    return tmp_path


def _build_plain_jsonl_byte_index(path: str) -> List[int]:
    offsets: List[int] = []
    pos = 0
    with open(path, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    return offsets


def _read_jsonl_line_by_offset(path: str, offsets: List[int], idx: int) -> Dict[str, Any]:
    off = int(offsets[idx])
    with open(path, "rb") as f:
        f.seek(off)
        line = f.readline()
    return json.loads(line.decode("utf-8"))


@dataclass(frozen=True)
class CNNRecord:
    img: torch.Tensor
    desc_ids: List[str]
    desc_meta: List[Dict[str, Any]]
    y: torch.Tensor


class CNNJsonlGzSTLDataset(Dataset):
    """
    Dataset over a plain .jsonl file path with fast byte-offset random access.

    Important:
    - This class does NOT own deletion of the input file.
    - If you pass a temp file produced from a .gz, the caller must remove it.
    """

    def __init__(
        self,
        path: str,
        *,
        example_id: Optional[int] = None,
        raster_cfg: Optional[CNNRasterConfig] = None,
    ) -> None:
        super().__init__()
        self.path = str(path)
        self.example_id = int(example_id) if example_id is not None else None
        self.raster_cfg = raster_cfg or CNNRasterConfig()

        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

        self._offsets = _build_plain_jsonl_byte_index(self.path)

        if len(self._offsets) > 0:
            r0 = _read_jsonl_line_by_offset(self.path, self._offsets, 0)
            if "sample" not in r0 or "micp" not in r0:
                raise ValueError("Dataset record must contain keys: 'sample' and 'micp'.")

        if self.example_id is not None:
            self._filtered_indices = self._build_filtered_indices(self.example_id)
        else:
            self._filtered_indices = None

    def _build_filtered_indices(self, example_id: int) -> List[int]:
        keep: List[int] = []
        for raw_idx in range(len(self._offsets)):
            rec = _read_jsonl_line_by_offset(self.path, self._offsets, raw_idx)
            sample = rec.get("sample", {})
            ex = sample.get("example_id", None)
            if ex is None:
                keep.append(raw_idx)
            elif int(ex) == int(example_id):
                keep.append(raw_idx)
        return keep

    def __len__(self) -> int:
        if self._filtered_indices is not None:
            return int(len(self._filtered_indices))
        return int(len(self._offsets))

    def _map_index(self, idx: int) -> int:
        if self._filtered_indices is None:
            return int(idx)
        return int(self._filtered_indices[int(idx)])

    def __getitem__(self, idx: int) -> CNNRecord:
        raw_idx = self._map_index(int(idx))
        rec = _read_jsonl_line_by_offset(self.path, self._offsets, raw_idx)

        sample = rec.get("sample", {})
        micp = rec.get("micp", {})

        descs = micp.get("descs", [])
        strategy = micp.get("strategy", {})

        if not isinstance(descs, list):
            raise ValueError("micp.descs must be a list.")
        if not isinstance(strategy, dict):
            raise ValueError("micp.strategy must be a dict.")

        desc_ids: List[str] = []
        desc_meta: List[Dict[str, Any]] = []
        y_list: List[float] = []

        for d in descs:
            if not isinstance(d, dict):
                continue

            did = str(d.get("id", "")).strip()
            if did == "":
                continue

            role = str(d.get("role", "")).strip()
            node_tag = str(d.get("node_tag", "")).strip()
            meta_raw = d.get("meta", {})
            meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}

            meta_out: Dict[str, Any] = dict(meta)
            meta_out["id"] = did
            meta_out["role"] = role
            meta_out["node_tag"] = node_tag

            if "op_uid" not in meta_out:
                meta_out["op_uid"] = str(meta.get("op_uid", node_tag))
            if "pred" not in meta_out:
                meta_out["pred"] = str(meta.get("pred", meta.get("ent_name", "")))
            if "ent_name" not in meta_out:
                meta_out["ent_name"] = str(meta.get("ent_name", meta_out["pred"]))
            if "pred_uid" not in meta_out:
                meta_out["pred_uid"] = str(meta.get("pred_uid", meta_out["pred"]))

            if "time" not in meta_out:
                if "t" in meta_out:
                    meta_out["time"] = int(meta_out["t"])
                elif "k" in meta_out:
                    meta_out["time"] = int(meta_out["k"])
                else:
                    meta_out["time"] = 0

            if "t" in meta_out:
                meta_out["t"] = int(meta_out["t"])
            if "k" in meta_out:
                meta_out["k"] = int(meta_out["k"])
            if "h" in meta_out and "face" not in meta_out:
                meta_out["face"] = int(meta_out["h"])
            if "face" in meta_out:
                meta_out["face"] = int(meta_out["face"])

            desc_ids.append(did)
            desc_meta.append(meta_out)
            y_list.append(float(int(strategy.get(did, 0))))

        img = rasterize_env_to_image(
            sample=sample,
            cfg=self.raster_cfg,
            device=torch.device("cpu"),
        )

        y = torch.tensor(y_list, dtype=torch.float32)
        return CNNRecord(img=img, desc_ids=desc_ids, desc_meta=desc_meta, y=y)


def cnn_collate_records(batch: List[CNNRecord]) -> Dict[str, Any]:
    if len(batch) <= 0:
        raise ValueError("Empty batch.")

    images = torch.stack([b.img for b in batch], dim=0)

    flat_desc_ids: List[str] = []
    flat_desc_meta: List[Dict[str, Any]] = []
    flat_y_list: List[torch.Tensor] = []
    desc_batch_idx_list: List[torch.Tensor] = []

    for i, b in enumerate(batch):
        m = int(b.y.numel())
        if len(b.desc_ids) != m or len(b.desc_meta) != m:
            raise ValueError(
                f"Descriptor alignment mismatch in batch item {i}: "
                f"len(desc_ids)={len(b.desc_ids)}, len(desc_meta)={len(b.desc_meta)}, y={m}"
            )
        flat_desc_ids.extend(list(b.desc_ids))
        flat_desc_meta.extend(list(b.desc_meta))
        flat_y_list.append(b.y)
        desc_batch_idx_list.append(torch.full((m,), int(i), dtype=torch.long))

    flat_y = torch.cat(flat_y_list, dim=0)
    desc_batch_idx = torch.cat(desc_batch_idx_list, dim=0)

    return {
        "images": images,
        "flat_desc_ids": flat_desc_ids,
        "flat_desc_meta": flat_desc_meta,
        "flat_y": flat_y,
        "desc_batch_idx": desc_batch_idx,
    }