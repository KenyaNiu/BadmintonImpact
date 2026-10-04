"""Unit tests for contact-state label extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from badminton_impact_ai.data.label_extraction import compute_contact_state_labels_from_fz
from badminton_impact_ai.data.preparation import build_pose_features, extract_base_labels


def test_compute_contact_state_labels_from_fz_basic_curve() -> None:
    fz = np.array([0.0, 0.05, 0.2, 0.9, 1.6, 1.2, 0.6, 0.2, 0.0], dtype=float)
    labels = compute_contact_state_labels_from_fz(fz=fz, fps=120.0)

    assert labels["peak_fz"] == 1.6
    assert labels["peak_index"] == 4
    assert labels["impulse_fz_full_window"] > 0.0
    assert labels["loading_rate_proxy"] > 0.0
    assert labels["valid_label"] is True


def test_raw_npz_preparation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    path = data_root / "sub_001" / "stage1_01_cam1_impact_001.npz"
    path.parent.mkdir(parents=True)
    grf = np.zeros((9, 3), dtype=np.float32)
    grf[:, 2] = [0.0, 0.05, 0.2, 0.9, 1.6, 1.2, 0.6, 0.2, 0.0]
    keypoints = np.ones((9, 17, 3), dtype=np.float32)
    np.savez(path, grf_normalized=grf, keypoints_norm=keypoints)

    rows, audit = extract_base_labels(data_root)
    features, names, metadata, feature_audit = build_pose_features(rows, data_root)

    assert audit["valid_rows"] == 1
    assert features.shape == (1, 16)
    assert len(names) == 16
    assert metadata[0]["feature_valid"] is True
    assert feature_audit["feature_rows"] == 1
