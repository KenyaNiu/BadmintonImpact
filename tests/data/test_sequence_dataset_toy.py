"""Toy dataset tests for sequence dataset loader."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from badminton_impact_ai.data.sequence_dataset import ContextEncoder, SequenceContactDataset, collate_sequence_batch


def test_toy_sequence_dataset(tmp_path: Path) -> None:
    toy_npz = tmp_path / "toy_segment.npz"
    np.savez(toy_npz, keypoints_norm=np.zeros((8, 17, 2), dtype=np.float32))
    csv_path = tmp_path / "toy_context.csv"
    row = {
        "npz_path": str(toy_npz),
        "subject_id": "sub_toy",
        "unique_impact_key_candidate": "sub_toy__trial__impact_001",
        "stage": "stage1",
        "fatigue_state": "fresh",
        "context_high_impact_q75": "1",
        "peak_fz_context_centered": "0.2",
        "impulse_fz_full_window": "0.3",
        "log1p_loading_rate_proxy": "0.4",
    }
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    enc = ContextEncoder.fit([row])
    ds = SequenceContactDataset(rows=[row], context_encoder=enc, stat_map={str(toy_npz): np.zeros(16, dtype="float32")})
    sample = ds[0]
    assert sample["x_seq"].shape[1] == 34
    batch = collate_sequence_batch([sample])
    assert batch["x_seq"].shape[0] == 1


def test_auxiliary_targets_do_not_filter_main_cohort() -> None:
    row = {
        "npz_path": "unused.npz",
        "subject_id": "sub_toy",
        "context_high_impact_q75": "1",
        "peak_fz_context_centered": "0.2",
        "impulse_fz_full_window": "nan",
        "log1p_loading_rate_proxy": "",
    }
    ds = SequenceContactDataset(rows=[row], context_encoder=ContextEncoder.fit([row]))
    assert len(ds) == 1
    assert ds.require_targets == ("context_high_impact_q75", "peak_fz_context_centered")
