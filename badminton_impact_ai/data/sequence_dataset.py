"""Sequence dataset utilities for M4 deep scout."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .cohort import MAIN_TASK_TARGETS, filter_eligible_rows, sample_id


@dataclass
class ContextEncoder:
    stage_to_idx: dict[str, int]
    fatigue_to_idx: dict[str, int]
    phase_to_idx: dict[str, int] | None = None

    @classmethod
    def fit(cls, rows: list[dict[str, str]]) -> "ContextEncoder":
        stage_vals = sorted({r.get("stage", "unknown") for r in rows})
        fatigue_vals = sorted({r.get("fatigue_state", "unknown") for r in rows})
        raw_ph = {str(r.get("protocol_phase", "")).strip() for r in rows}
        raw_ph.discard("")
        phase_vals = sorted(raw_ph)
        phase_to_idx = {v: i for i, v in enumerate(phase_vals)} if phase_vals else None
        return cls(
            stage_to_idx={v: i for i, v in enumerate(stage_vals)},
            fatigue_to_idx={v: i for i, v in enumerate(fatigue_vals)},
            phase_to_idx=phase_to_idx,
        )

    def output_dim(self) -> int:
        n = len(self.stage_to_idx) + len(self.fatigue_to_idx)
        if self.phase_to_idx is not None:
            n += len(self.phase_to_idx)
        return n

    def encode(self, stage: str, fatigue: str, protocol_phase: str | None = None) -> np.ndarray:
        if self.phase_to_idx is None:
            x = np.zeros(len(self.stage_to_idx) + len(self.fatigue_to_idx), dtype=np.float32)
        else:
            x = np.zeros(len(self.stage_to_idx) + len(self.fatigue_to_idx) + len(self.phase_to_idx), dtype=np.float32)
        s_idx = self.stage_to_idx.get(stage)
        f_idx = self.fatigue_to_idx.get(fatigue)
        if s_idx is not None:
            x[s_idx] = 1.0
        if f_idx is not None:
            x[len(self.stage_to_idx) + f_idx] = 1.0
        if self.phase_to_idx is not None:
            ph = str(protocol_phase if protocol_phase is not None else "0").strip()
            if ph not in self.phase_to_idx:
                ph = next(iter(self.phase_to_idx))
            p_idx = self.phase_to_idx.get(ph)
            if p_idx is not None:
                x[len(self.stage_to_idx) + len(self.fatigue_to_idx) + p_idx] = 1.0
        return x


def load_stat_features(features_npz: Path, feature_meta_csv: Path) -> tuple[dict[str, np.ndarray], int]:
    with np.load(features_npz, allow_pickle=True) as d:
        X = np.asarray(d["X"], dtype=np.float32)
        npz_paths = [str(x) for x in d["npz_path"]]
        sample_ids = [str(x) for x in d["sample_id"]] if "sample_id" in d else npz_paths
    valid_meta = {}
    with feature_meta_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("feature_valid", "")).lower() in {"true", "1", "yes", "y"}:
                valid_meta[r.get("sample_id") or r["npz_path"]] = True
    out = {}
    for i, (sid, path) in enumerate(zip(sample_ids, npz_paths)):
        if sid in valid_meta or path in valid_meta:
            out[sid] = X[i]
            out[path] = X[i]
    feat_dim = int(X.shape[1]) if X.ndim == 2 else 0
    return out, feat_dim


def load_context_rows(context_csv: Path, fold_subject: str, role: str) -> list[dict[str, str]]:
    with context_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("test_subject") == fold_subject and r.get("role") == role]


class SequenceContactDataset(Dataset):
    """Dataset returning sequence, context, optional stat features, and targets."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        context_encoder: ContextEncoder,
        stat_map: dict[str, np.ndarray] | None = None,
        require_targets: tuple[str, ...] = MAIN_TASK_TARGETS,
    ) -> None:
        self.context_encoder = context_encoder
        self.stat_map = stat_map or {}
        self.require_targets = require_targets
        self.rows = filter_eligible_rows(rows, require_targets=require_targets)
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_csv(
        cls,
        context_csv: Path,
        fold_subject: str,
        role: str,
        context_encoder: ContextEncoder,
        stat_map: dict[str, np.ndarray] | None = None,
    ) -> "SequenceContactDataset":
        rows = load_context_rows(context_csv, fold_subject=fold_subject, role=role)
        return cls(rows=rows, context_encoder=context_encoder, stat_map=stat_map)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_seq(self, npz_path: str) -> np.ndarray:
        if npz_path in self._cache:
            return self._cache[npz_path]
        with np.load(npz_path, allow_pickle=True) as d:
            if "keypoints_norm" not in d:
                raise KeyError("missing keypoints_norm")
            arr = np.asarray(d["keypoints_norm"], dtype=np.float32)
            if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] < 2:
                raise ValueError("keypoints_norm expected shape (T,17,>=2)")
            arr = arr[:, :, :2]
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            seq = arr.reshape(arr.shape[0], -1).astype(np.float32)
            self._cache[npz_path] = seq
            return seq

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        npz_path = r["npz_path"]
        seq = self._load_seq(npz_path)
        if self.context_encoder.phase_to_idx is not None:
            ph = str(r.get("protocol_phase", "0")).strip()
            context = self.context_encoder.encode(r.get("stage", "unknown"), r.get("fatigue_state", "unknown"), ph)
        else:
            context = self.context_encoder.encode(r.get("stage", "unknown"), r.get("fatigue_state", "unknown"))
        stat = self.stat_map.get(sample_id(r), self.stat_map.get(npz_path))
        if stat is None:
            stat = np.zeros(0, dtype=np.float32)
        y = {target: np.float32(float(r[target])) for target in self.require_targets}
        return {
            "x_seq": seq,
            "x_context": context.astype(np.float32),
            "x_stat": stat.astype(np.float32),
            "targets": y,
            "metadata": {
                "npz_path": npz_path,
                "subject_id": r.get("subject_id", "unknown"),
                "sample_id": sample_id(r),
                "unique_impact_key_candidate": r.get("unique_impact_key_candidate", "unknown"),
                "trial_id": r.get("trial_id", "unknown"),
                "camera_id": r.get("camera_id", "unknown"),
                "stage": r.get("stage", "unknown"),
                "fatigue_state": r.get("fatigue_state", "unknown"),
            },
        }


def collate_sequence_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch")
    lens = [item["x_seq"].shape[0] for item in batch]
    tmax = max(lens)
    bsz = len(batch)
    dseq = batch[0]["x_seq"].shape[1]
    dctx = batch[0]["x_context"].shape[0]
    dstat = batch[0]["x_stat"].shape[0]
    x_seq = np.zeros((bsz, tmax, dseq), dtype=np.float32)
    mask = np.zeros((bsz, tmax), dtype=np.float32)
    x_ctx = np.zeros((bsz, dctx), dtype=np.float32)
    x_stat = np.zeros((bsz, dstat), dtype=np.float32)
    tgt = {k: np.zeros((bsz,), dtype=np.float32) for k in batch[0]["targets"].keys()}
    meta = []
    for i, item in enumerate(batch):
        t = item["x_seq"].shape[0]
        x_seq[i, :t] = item["x_seq"]
        mask[i, :t] = 1.0
        x_ctx[i] = item["x_context"]
        if dstat > 0:
            x_stat[i] = item["x_stat"]
        for k, v in item["targets"].items():
            tgt[k][i] = np.float32(v)
        meta.append(item["metadata"])
    return {
        "x_seq": torch.from_numpy(x_seq),
        "seq_mask": torch.from_numpy(mask),
        "x_context": torch.from_numpy(x_ctx),
        "x_stat": torch.from_numpy(x_stat),
        "targets": {k: torch.from_numpy(v) for k, v in tgt.items()},
        "metadata": meta,
    }
