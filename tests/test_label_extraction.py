"""Unit tests for contact-state label extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.label_extraction import compute_contact_state_labels_from_fz


def test_compute_contact_state_labels_from_fz_basic_curve() -> None:
    fz = np.array([0.0, 0.05, 0.2, 0.9, 1.6, 1.2, 0.6, 0.2, 0.0], dtype=float)
    labels = compute_contact_state_labels_from_fz(fz=fz, fps=120.0)

    assert labels["peak_fz"] == 1.6
    assert labels["peak_index"] == 4
    assert labels["impulse_fz_full_window"] > 0.0
    assert labels["loading_rate_proxy"] > 0.0
    assert labels["valid_label"] is True
