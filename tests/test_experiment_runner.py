"""Small HGB-only integration check for the canonical run directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from badminton_impact_ai.experiment.analysis import analyze_run
from badminton_impact_ai.experiment.artifacts import make_paper_artifacts
from badminton_impact_ai.experiment.runner import run_experiment


def test_runner_writes_complete_provenance_bundle(tmp_path: Path) -> None:
    rows = []
    for index in range(16):
        subject = "s1" if index < 4 else ("s2" if index < 10 else "s3")
        npz_path = tmp_path / f"sample_{index}.npz"
        np.savez(npz_path, keypoints_norm=np.full((5 + index % 3, 17, 2), index / 10, dtype=np.float32))
        rows.append(
            {
                "npz_path": str(npz_path),
                "rel_path": f"sample_{index}.npz",
                "subject_id": subject,
                "test_subject": "s1",
                "role": "test" if subject == "s1" else "train",
                "trial_id": f"trial_{subject}",
                "camera_id": "cam1",
                "stage": "stage",
                "fatigue_state": "fresh",
                "unique_impact_key_candidate": f"impact_{index}",
                "context_high_impact_q75": str(index % 2),
                "peak_fz_context_centered": str(float(index % 5)),
            }
        )
    labels = tmp_path / "labels.csv"
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    features = tmp_path / "features.npz"
    np.savez(
        features,
        X=np.arange(16 * 3, dtype=np.float32).reshape(16, 3),
        sample_id=np.asarray([row["rel_path"] for row in rows]),
        npz_path=np.asarray([row["npz_path"] for row in rows]),
    )
    feature_meta = tmp_path / "feature_meta.csv"
    with feature_meta.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "npz_path", "feature_valid"])
        writer.writeheader()
        writer.writerows(
            {"sample_id": row["rel_path"], "npz_path": row["npz_path"], "feature_valid": "true"} for row in rows
        )
    config = {
        "data": {
            "labels": str(labels),
            "features": str(features),
            "feature_meta": str(feature_meta),
            "expected_test_views": 4,
            "expected_test_impacts": 4,
        },
        "training": {
            "seed": 42,
            "split_seed": 42,
            "val_ratio": 0.2,
            "batch_size": 4,
            "epochs": 1,
            "patience": 1,
            "learning_rate": 0.001,
            "hidden_dim": 16,
            "aux_loss_weight": 0.5,
            "device": "cpu",
        },
        "folds": ["s1"],
        "models": [{"name": "hgb_pose_context"}, {"name": "cn_hildnet"}],
        "evaluation": {"top_fraction": 0.5, "bootstrap_draws": 100, "statistical_seed": 42},
    }
    preflight_dir = tmp_path / "preflight"
    run_experiment(config, preflight_dir, check_only=True)
    assert json.loads((preflight_dir / "status.json").read_text())["state"] == "preflight_complete"
    assert not (preflight_dir / "predictions").exists()
    run_dir = tmp_path / "run"
    run_experiment(config, run_dir)
    assert json.loads((run_dir / "status.json").read_text())["state"] == "complete"
    assert (run_dir / "cohort.csv").exists()
    assert (run_dir / "splits.csv").exists()
    assert len(list((run_dir / "predictions").glob("*.csv"))) == 2
    assert analyze_run(run_dir).exists()
    assert make_paper_artifacts(run_dir, tmp_path / "paper_artifacts").exists()
