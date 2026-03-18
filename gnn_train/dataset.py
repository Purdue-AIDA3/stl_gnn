# neurosymbolic/gnn_train/dataset.py
from __future__ import annotations

import atexit
import gzip
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Dataset

from .graph_builder import X_DIM, build_hetero_graph_from_record


def _get_strategy_binaries(rec: Dict[str, Any]) -> Dict[str, int]:
    """
    Ground-truth strategic binaries come from rec["micp"]["strategy"].
    Must be a dict[str, int] with values in {0,1} (nonzero -> 1).
    """
    micp = rec.get("micp", {})
    if not isinstance(micp, dict):
        raise ValueError("Record missing dict 'micp'.")

    strat = micp.get("strategy", None)
    if not isinstance(strat, dict):
        raise ValueError("Record missing dict micp['strategy'].")

    out: Dict[str, int] = {}
    for k, v in strat.items():
        if not isinstance(k, str):
            continue
        try:
            iv = int(v)
        except Exception:
            continue
        out[k] = 1 if iv != 0 else 0
    return out


def _is_gz_path(path: str) -> bool:
    return path.endswith(".gz")


def _safe_unlink(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception:
        return


def build_labeled_hetero_graph_from_record(rec: Dict[str, Any]):
    """
    Build one hetero graph and attach binary supervision on b-nodes.

    Attached fields:
      - data.example_id: (1,) long
      - data.T:          already set by graph_builder, checked against record
      - data.tau:        already set by graph_builder, checked against record
      - data["b"].bin_y: (Nb,) float
      - data["b"].bin_supervised_mask: (Nb,) bool
      - data.n_bin_supervised: (1,) long
    """
    sample = rec.get("sample", {})
    if not isinstance(sample, dict):
        raise ValueError("Record must contain dict 'sample'.")

    data = build_hetero_graph_from_record(rec)

    ex = int(sample.get("example_id", 1))
    data.example_id = torch.tensor([ex], dtype=torch.long)

    T_rec = int(sample.get("T", 0))
    if not hasattr(data, "T"):
        raise RuntimeError("Graph missing data.T (graph_builder must set it).")
    T_graph = int(data.T.view(-1)[0].item())
    if T_graph != T_rec:
        raise RuntimeError(f"T mismatch: record={T_rec} graph_builder={T_graph}")

    env = sample.get("env", {}) if isinstance(sample.get("env", {}), dict) else {}
    meta = env.get("meta", {}) if isinstance(env.get("meta", {}), dict) else {}
    tau_rec = int(meta.get("tau", env.get("tau", 0)))
    if not hasattr(data, "tau"):
        raise RuntimeError("Graph missing data.tau (graph_builder must set it).")
    tau_graph = int(data.tau.view(-1)[0].item())
    if tau_graph != tau_rec:
        raise RuntimeError(f"tau mismatch: record={tau_rec} graph_builder={tau_graph}")

    strategy = _get_strategy_binaries(rec)

    if "b" not in data.node_types:
        raise RuntimeError("Hetero graph missing node type 'b'.")
    if not hasattr(data["b"], "node_desc_id_str"):
        raise RuntimeError("Hetero graph missing data['b'].node_desc_id_str.")

    desc_ids = data["b"].node_desc_id_str
    if not isinstance(desc_ids, list) or len(desc_ids) != int(data["b"].x.size(0)):
        raise RuntimeError("data['b'].node_desc_id_str must be list[str] aligned with b nodes.")

    nb = int(data["b"].x.size(0))
    bin_y = torch.zeros((nb,), dtype=torch.float)
    bin_mask = torch.zeros((nb,), dtype=torch.bool)

    for i in range(nb):
        did = desc_ids[i]
        if did in strategy:
            bin_y[i] = float(strategy[did])
            bin_mask[i] = True

    data["b"].bin_y = bin_y
    data["b"].bin_supervised_mask = bin_mask
    data.n_bin_supervised = torch.tensor([int(bin_mask.sum().item())], dtype=torch.long)

    return data


class JsonlGzSTLDataset(Dataset):
    """
    Binary-only hetero dataset.

    Performance notes:
      - Reading random lines from .jsonl.gz by scanning from the start is extremely slow.
      - This dataset supports a "unzip once + byte-offset index" strategy:
          1) If input is .jsonl.gz, it is decompressed once to a temp .jsonl.
          2) A byte-offset index (per line) is built once.
          3) get() does seek+readline+json.loads.
    """

    def __init__(self, path: str, *, example_id: Optional[int] = None):
        super().__init__()
        self.path = path
        self.example_id = int(example_id) if example_id is not None else None
        self.x_dim = int(X_DIM)

        self._jsonl_path: str = ""
        self._offsets: List[int] = []
        self._indices: List[int] = []
        self._n: int = 0

        self._created_temp: bool = False
        self._temp_jsonl_path: Optional[str] = None
        self._lock_path: Optional[str] = None

        self._prepare_jsonl_and_index()

        if len(self._indices) == 0:
            raise ValueError(f"No samples found for example_id={self.example_id} in {self.path}")

        self._n = len(self._indices)

        _ = self._read_record_by_line_no(self._indices[0])

        atexit.register(self.cleanup)

    def _temp_paths(self) -> Tuple[str, str]:
        """
        Returns (temp_jsonl_path, lock_path) used when input is .gz.
        """
        abs_path = os.path.abspath(self.path)
        base = os.path.basename(abs_path)
        safe = base.replace("/", "_")
        suffix = str(abs(hash(abs_path)))[:10]
        tmp_dir = tempfile.gettempdir()
        temp_jsonl = os.path.join(tmp_dir, f"{safe}.{suffix}.tmp.jsonl")
        lock = os.path.join(tmp_dir, f"{safe}.{suffix}.tmp.jsonl.lock")
        return temp_jsonl, lock

    def _acquire_lock(self, lock_path: str, *, max_wait_s: float = 300.0) -> bool:
        """
        Cross-process lock using O_EXCL file creation.
        Returns True if acquired, False if timed out.
        """
        t0 = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.close(fd)
                return True
            except FileExistsError:
                if (time.time() - t0) > float(max_wait_s):
                    return False
                time.sleep(0.05)
            except Exception:
                return False

    def _release_lock(self, lock_path: str) -> None:
        _safe_unlink(lock_path)

    def _ensure_unzipped_jsonl(self) -> str:
        """
        Ensure there is a plain .jsonl available and return its path.
        """
        if not _is_gz_path(self.path):
            return self.path

        temp_jsonl, lock_path = self._temp_paths()
        self._temp_jsonl_path = temp_jsonl
        self._lock_path = lock_path

        if os.path.exists(temp_jsonl) and os.path.getsize(temp_jsonl) > 0:
            return temp_jsonl

        acquired = self._acquire_lock(lock_path)
        if not acquired:
            if os.path.exists(temp_jsonl) and os.path.getsize(temp_jsonl) > 0:
                return temp_jsonl
            raise RuntimeError(f"Failed to acquire unzip lock: {lock_path}")

        try:
            if os.path.exists(temp_jsonl) and os.path.getsize(temp_jsonl) > 0:
                return temp_jsonl

            tmp_write = temp_jsonl + ".writing"
            _safe_unlink(tmp_write)

            with gzip.open(self.path, "rb") as f_in, open(tmp_write, "wb") as f_out:
                while True:
                    chunk = f_in.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)

            os.replace(tmp_write, temp_jsonl)
            self._created_temp = True
            return temp_jsonl
        finally:
            self._release_lock(lock_path)

    def _prepare_jsonl_and_index(self) -> None:
        """
        Builds:
          - self._jsonl_path
          - self._offsets
          - self._indices
        """
        self._jsonl_path = self._ensure_unzipped_jsonl()

        offsets: List[int] = []
        indices: List[int] = []

        with open(self._jsonl_path, "rb") as f:
            line_no = 0
            pos = f.tell()
            line = f.readline()
            while line:
                offsets.append(pos)

                if self.example_id is None:
                    indices.append(line_no)
                else:
                    try:
                        rec = json.loads(line.decode("utf-8"))
                        sample = rec.get("sample", {})
                        ex = int(sample.get("example_id", 1)) if isinstance(sample, dict) else 1
                        if ex == int(self.example_id):
                            indices.append(line_no)
                    except Exception:
                        pass

                line_no += 1
                pos = f.tell()
                line = f.readline()

        if len(offsets) == 0:
            raise ValueError(f"Empty dataset file: {self._jsonl_path}")

        self._offsets = offsets
        self._indices = indices

    def _read_record_by_line_no(self, line_no: int) -> Dict[str, Any]:
        if line_no < 0 or line_no >= len(self._offsets):
            raise IndexError(line_no)

        off = int(self._offsets[line_no])
        with open(self._jsonl_path, "rb") as f:
            f.seek(off)
            line = f.readline()
        if not line:
            raise IndexError(line_no)
        return json.loads(line.decode("utf-8"))

    def len(self) -> int:
        return self._n

    def get(self, idx: int):
        line_no = self._indices[idx]
        rec = self._read_record_by_line_no(line_no)
        return build_labeled_hetero_graph_from_record(rec)

    def cleanup(self) -> None:
        """
        Best-effort cleanup of temp jsonl when the Dataset created it.
        """
        if self._created_temp and self._temp_jsonl_path:
            _safe_unlink(self._temp_jsonl_path)
        if self._lock_path:
            _safe_unlink(self._lock_path)


__all__ = [
    "JsonlGzSTLDataset",
    "X_DIM",
    "build_labeled_hetero_graph_from_record",
]