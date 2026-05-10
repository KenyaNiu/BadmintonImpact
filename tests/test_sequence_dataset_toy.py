"""Toy dataset tests for sequence dataset loader."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.sequence_dataset import ContextEncoder, SequenceContactDataset, collate_sequence_batch


def test_toy_sequence_dataset(tmp_path: Path) -> None:
    toy_npz = tmp_path / "toy_segment.npz"
    t, j = 8, 17
    keypoints = np.random.randn(t, j, 2).astype(np.float32)
    np.savez(toy_npz, keypoints_norm=keypoints)

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
    ds = SequenceContactDataset(
        rows=[row], context_encoder=enc, stat_map={str(toy_npz): np.zeros(16, dtype="float32")}
    )
    sample = ds[0]
    assert sample["x_seq"].shape[1] == 34
    batch = collate_sequence_batch([sample])
    assert batch["x_seq"].shape[0] == 1
