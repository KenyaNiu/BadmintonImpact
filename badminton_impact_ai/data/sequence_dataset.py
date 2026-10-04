"""Pose-sequence dataset, protocol-context encoding and batching for the neural models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .cohort import MAIN_TASK_TARGETS, filter_eligible_rows, sample_id

TRUE_VALUES = {"true", "1", "yes", "y"}


@dataclass
class ContextEncoder:
    """One-hot encoding of the protocol context (movement stage, fatigue and an optional phase).

    The vocabulary is fitted on training rows only, so unseen categories at test time encode to zeros.
    """

    stage_to_idx: dict[str, int]
    fatigue_to_idx: dict[str, int]
    phase_to_idx: dict[str, int] | None = None

    @classmethod
    def fit(cls, rows: list[dict[str, str]]) -> ContextEncoder:
        stages = sorted({row.get("stage", "unknown") for row in rows})
        fatigue = sorted({row.get("fatigue_state", "unknown") for row in rows})
        phases = sorted({str(row.get("protocol_phase", "")).strip() for row in rows} - {""})
        return cls(
            stage_to_idx={value: i for i, value in enumerate(stages)},
            fatigue_to_idx={value: i for i, value in enumerate(fatigue)},
            phase_to_idx={value: i for i, value in enumerate(phases)} if phases else None,
        )

    def output_dim(self) -> int:
        phase_dim = len(self.phase_to_idx) if self.phase_to_idx is not None else 0
        return len(self.stage_to_idx) + len(self.fatigue_to_idx) + phase_dim

    def encode(self, stage: str, fatigue: str, protocol_phase: str | None = None) -> np.ndarray:
        vector = np.zeros(self.output_dim(), dtype=np.float32)
        if (stage_index := self.stage_to_idx.get(stage)) is not None:
            vector[stage_index] = 1.0
        if (fatigue_index := self.fatigue_to_idx.get(fatigue)) is not None:
            vector[len(self.stage_to_idx) + fatigue_index] = 1.0
        if self.phase_to_idx is not None:
            phase = str(protocol_phase if protocol_phase is not None else "0").strip()
            if phase not in self.phase_to_idx:
                phase = next(iter(self.phase_to_idx))
            vector[len(self.stage_to_idx) + len(self.fatigue_to_idx) + self.phase_to_idx[phase]] = 1.0
        return vector


def load_stat_features(features_npz: Path, feature_meta_csv: Path) -> dict[str, np.ndarray]:
    """Window-statistics vectors of the valid samples, keyed by both sample id and NPZ path."""
    with np.load(features_npz, allow_pickle=True) as archive:
        matrix = np.asarray(archive["X"], dtype=np.float32)
        paths = [str(path) for path in archive["npz_path"]]
        sample_ids = [str(sid) for sid in archive["sample_id"]] if "sample_id" in archive else paths
    with feature_meta_csv.open("r", encoding="utf-8", newline="") as stream:
        valid = {
            row.get("sample_id") or row["npz_path"]
            for row in csv.DictReader(stream)
            if str(row.get("feature_valid", "")).lower() in TRUE_VALUES
        }
    features: dict[str, np.ndarray] = {}
    for vector, sid, path in zip(matrix, sample_ids, paths, strict=True):
        if sid in valid or path in valid:
            features[sid] = features[path] = vector
    return features


class SequenceContactDataset(Dataset):
    """Pose sequence, context vector, window statistics and targets of each eligible landing view."""

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

    def __len__(self) -> int:
        return len(self.rows)

    def _load_sequence(self, npz_path: str) -> np.ndarray:
        """Flattened COCO-17 (x, y) coordinates, shape ``(T, 34)``."""
        if npz_path not in self._cache:
            with np.load(npz_path, allow_pickle=True) as archive:
                if "keypoints_norm" not in archive:
                    raise KeyError("missing keypoints_norm")
                keypoints = np.asarray(archive["keypoints_norm"], dtype=np.float32)
            if keypoints.ndim != 3 or keypoints.shape[1] != 17 or keypoints.shape[2] < 2:
                raise ValueError("keypoints_norm expected shape (T,17,>=2)")
            keypoints = np.nan_to_num(keypoints[:, :, :2], nan=0.0, posinf=0.0, neginf=0.0)
            self._cache[npz_path] = keypoints.reshape(keypoints.shape[0], -1).astype(np.float32)
        return self._cache[npz_path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        npz_path = row["npz_path"]
        context = self.context_encoder.encode(
            row.get("stage", "unknown"), row.get("fatigue_state", "unknown"), row.get("protocol_phase", "0")
        )
        stat = self.stat_map.get(sample_id(row), self.stat_map.get(npz_path))
        if stat is None:
            stat = np.zeros(0, dtype=np.float32)
        return {
            "x_seq": self._load_sequence(npz_path),
            "x_context": context.astype(np.float32),
            "x_stat": stat.astype(np.float32),
            "targets": {target: np.float32(float(row[target])) for target in self.require_targets},
            "metadata": {
                "npz_path": npz_path,
                "subject_id": row.get("subject_id", "unknown"),
                "sample_id": sample_id(row),
                "unique_impact_key_candidate": row.get("unique_impact_key_candidate", "unknown"),
                "trial_id": row.get("trial_id", "unknown"),
                "camera_id": row.get("camera_id", "unknown"),
                "stage": row.get("stage", "unknown"),
                "fatigue_state": row.get("fatigue_state", "unknown"),
            },
        }


def collate_sequence_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Zero-pad variable-length sequences and return a validity mask ``seq_mask`` of shape ``(B, T)``."""
    if not batch:
        raise ValueError("empty batch")
    max_len = max(item["x_seq"].shape[0] for item in batch)
    x_seq = np.zeros((len(batch), max_len, batch[0]["x_seq"].shape[1]), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.float32)
    for i, item in enumerate(batch):
        length = item["x_seq"].shape[0]
        x_seq[i, :length] = item["x_seq"]
        mask[i, :length] = 1.0
    targets = {
        name: np.asarray([item["targets"][name] for item in batch], dtype=np.float32) for name in batch[0]["targets"]
    }
    return {
        "x_seq": torch.from_numpy(x_seq),
        "seq_mask": torch.from_numpy(mask),
        "x_context": torch.from_numpy(np.stack([item["x_context"] for item in batch])),
        "x_stat": torch.from_numpy(np.stack([item["x_stat"] for item in batch])),
        "targets": {name: torch.from_numpy(values) for name, values in targets.items()},
        "metadata": [item["metadata"] for item in batch],
    }
