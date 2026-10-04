"""Canonical raw-NPZ preparation for labels and flat pose descriptors."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from badminton_impact_ai.io import read_csv, write_csv, write_json

from .context_labels import build_context_normalized_loso
from .label_extraction import compute_contact_state_labels_from_fz
from .metadata_parse import parse_npz_metadata


def extract_base_labels(data_root: Path, fps: float = 120.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract one view-level supervision row from every readable Tier-1 NPZ."""
    data_root = data_root.resolve()
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    paths = sorted(data_root.rglob("*.npz"))
    for path in paths:
        try:
            with np.load(path, allow_pickle=True) as archive:
                source_key = next((key for key in ("grf_normalized", "grf_at_video_fps") if key in archive), None)
                if source_key is None:
                    skipped["missing_grf"] += 1
                    continue
                grf = np.asarray(archive[source_key])
                if grf.ndim != 2 or grf.shape[1] < 3:
                    skipped["invalid_grf_shape"] += 1
                    continue
                metadata = {
                    key: archive[key].item() if np.asarray(archive[key]).ndim == 0 else archive[key]
                    for key in ("subject", "camera", "trial", "stage", "impact_id")
                    if key in archive
                }
            rel_path = path.relative_to(data_root).as_posix()
            parsed = parse_npz_metadata(rel_path, metadata)
            labels = compute_contact_state_labels_from_fz(grf[:, 2], fps=fps)
            rows.append(
                {
                    "npz_path": str(path),
                    "rel_path": rel_path,
                    "source_grf_key": source_key,
                    **{
                        key: parsed[key]
                        for key in (
                            "subject_id",
                            "camera_id",
                            "trial_id",
                            "stage",
                            "fatigue_state",
                            "unique_impact_key_candidate",
                        )
                    },
                    **labels,
                }
            )
        except Exception as exc:  # corrupted external input must not abort the full scan
            skipped[f"read_error:{type(exc).__name__}"] += 1

    return rows, {
        "data_root": str(data_root),
        "npz_found": len(paths),
        "rows_written": len(rows),
        "valid_rows": sum(bool(row["valid_label"]) for row in rows),
        "skipped": dict(skipped),
    }


def _pose_descriptors(keypoints: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if keypoints.ndim != 3 or keypoints.shape[1] != 17 or keypoints.shape[2] < 2:
        raise ValueError("keypoints_norm must have shape (T, 17, >=2)")
    coordinates = np.nan_to_num(keypoints[..., :2], nan=0.0, posinf=0.0, neginf=0.0).reshape(keypoints.shape[0], -1)
    flat = coordinates.reshape(-1)
    velocity = np.diff(coordinates, axis=0).reshape(-1) if len(coordinates) > 1 else np.zeros(1)
    confidence = (
        np.nan_to_num(keypoints[..., 2], nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        if keypoints.shape[2] >= 3
        else np.ones(keypoints.shape[:2], dtype=float).reshape(-1)
    )
    values = [
        np.mean(flat),
        np.std(flat),
        np.min(flat),
        np.max(flat),
        np.percentile(flat, 25),
        np.percentile(flat, 50),
        np.percentile(flat, 75),
        np.mean(coordinates[0]),
        np.mean(coordinates[-1]),
        np.mean(coordinates[-1] - coordinates[0]),
        np.mean(velocity),
        np.std(velocity),
        np.max(np.abs(velocity)),
        np.mean(confidence),
        np.min(confidence),
        np.std(confidence),
    ]
    names = [
        "mean",
        "std",
        "min",
        "max",
        "p25",
        "p50",
        "p75",
        "first_frame_mean",
        "last_frame_mean",
        "delta_last_first_mean",
        "velocity_mean",
        "velocity_std",
        "velocity_max_abs",
        "confidence_mean",
        "confidence_min",
        "confidence_std",
    ]
    return np.asarray(values, dtype=np.float32), names


def build_pose_features(
    rows: list[dict[str, Any]], data_root: Path
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], dict[str, Any]]:
    """Build exactly one 16-D descriptor row per valid camera view."""
    unique = {
        str(row.get("rel_path") or row["npz_path"]): row
        for row in rows
        if str(row["valid_label"]).strip().lower() in {"1", "true", "yes", "y"}
    }
    features: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    invalid: Counter[str] = Counter()
    feature_names: list[str] = []

    for sample_id, row in unique.items():
        path = data_root / row["rel_path"] if row.get("rel_path") else Path(row["npz_path"])
        reason = ""
        try:
            with np.load(path, allow_pickle=True) as archive:
                if "keypoints_norm" not in archive:
                    raise KeyError("keypoints_norm")
                values, names = _pose_descriptors(np.asarray(archive["keypoints_norm"], dtype=float))
            feature_names = feature_names or names
            if names != feature_names:
                raise ValueError("feature_schema_mismatch")
            features.append(values)
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            invalid[type(exc).__name__] += 1
        metadata.append(
            {
                "sample_id": sample_id,
                "rel_path": row.get("rel_path", ""),
                "npz_path": str(path),
                "subject_id": row.get("subject_id", "unknown"),
                "camera_id": row.get("camera_id", "unknown"),
                "stage": row.get("stage", "unknown"),
                "fatigue_state": row.get("fatigue_state", "unknown"),
                "unique_impact_key_candidate": row.get("unique_impact_key_candidate", "unknown"),
                "feature_valid": not reason,
                "feature_invalid_reason": reason,
            }
        )

    matrix = np.vstack(features).astype(np.float32) if features else np.zeros((0, 16), dtype=np.float32)
    valid_metadata = [row for row in metadata if row["feature_valid"]]
    return (
        matrix,
        feature_names,
        metadata,
        {
            "eligible_views": len(unique),
            "feature_rows": len(valid_metadata),
            "feature_dim": int(matrix.shape[1]),
            "invalid": dict(invalid),
        },
    )


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, fieldnames=list(rows[0]))


def prepare_dataset(config_path: Path, overwrite: bool = False) -> Path:
    """Run the canonical, training-free preparation pipeline."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"]).resolve()
    out_dir = Path(config.get("out_dir", "outputs/prepared"))
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace canonical files")
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.get("base_labels"):
        base_rows = read_csv(Path(config["base_labels"]))
        extraction_audit = {"source": "existing_csv", "rows_written": len(base_rows)}
    else:
        base_rows, extraction_audit = extract_base_labels(data_root, fps=float(config.get("fps", 120.0)))
    if not base_rows:
        raise RuntimeError("no label rows were extracted")
    _write_table(out_dir / "base_labels.csv", base_rows)

    labeled, threshold_audit = build_context_normalized_loso(
        base_rows,
        percentile=float(config.get("percentile", 75.0)),
        minimum_events=int(config.get("minimum_events", 50)),
    )
    for row in labeled:
        row["npz_path"] = str(data_root / row["rel_path"]) if row.get("rel_path") else row["npz_path"]
    _write_table(out_dir / "context_labels_event_q75.csv", labeled)

    matrix, feature_names, metadata, feature_audit = build_pose_features(base_rows, data_root)
    valid_metadata = [row for row in metadata if row["feature_valid"]]
    np.savez(
        out_dir / "pose_features_unique.npz",
        X=matrix,
        sample_id=np.asarray([row["sample_id"] for row in valid_metadata], dtype=str),
        npz_path=np.asarray([row["npz_path"] for row in valid_metadata], dtype=str),
        feature_names=np.asarray(feature_names, dtype=str),
    )
    _write_table(out_dir / "pose_features_unique_meta.csv", metadata)
    audit = {"label_extraction": extraction_audit, "thresholds": threshold_audit, "features": feature_audit}
    write_json(out_dir / "preparation_audit.json", audit)
    return out_dir
