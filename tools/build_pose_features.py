#!/usr/bin/env python3
"""Build flat statistical pose features from NPZ pose/keypoint inputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

CANDIDATE_TOKENS = ("keypoints", "keypoint", "pose", "kpts", "joints", "coco", "input", "feature", "features")
EXCLUDE_TOKENS = ("grf", "force", "fz", "target", "label", "moment", "cop")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build flat pose features from labels + NPZ.")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--out-meta", type=Path, required=True)
    p.add_argument("--out-report", type=Path, required=True)
    return p.parse_args()


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _is_candidate(k: str) -> bool:
    low = k.lower()
    return any(t in low for t in CANDIDATE_TOKENS) and not any(t in low for t in EXCLUDE_TOKENS)


def _score_key(k: str, arr: np.ndarray) -> tuple[int, int, int]:
    low = k.lower()
    name_score = 0
    if "keypoints" in low:
        name_score = 3
    elif "keypoint" in low or "pose" in low:
        name_score = 2
    elif "feature" in low or "input" in low:
        name_score = 1
    shape_score = 0
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        shape_score = 3
    elif arr.ndim == 2:
        shape_score = 2
    elif arr.ndim == 1:
        shape_score = 1
    return (shape_score, name_score, int(np.prod(arr.shape) > 0))


def _to_seq2d(arr: np.ndarray) -> np.ndarray | None:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 3:
        t = a.shape[0]
        return a.reshape(t, -1)
    if a.ndim == 2:
        return a
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return None


def _extract_confidence(data: np.lib.npyio.NpzFile, source_arr: np.ndarray) -> np.ndarray | None:
    if source_arr.ndim == 3 and source_arr.shape[-1] >= 3:
        return np.asarray(source_arr[..., 2], dtype=float)
    for k in data.keys():
        low = k.lower()
        if any(tok in low for tok in ("score", "confidence", "conf")) and not any(tok in low for tok in EXCLUDE_TOKENS):
            try:
                c = np.asarray(data[k], dtype=float)
                return c
            except Exception:
                continue
    return None


def _stats_features(seq2d: np.ndarray, conf: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    x = np.nan_to_num(seq2d, nan=0.0, posinf=0.0, neginf=0.0)
    flat = x.reshape(-1)
    # Global scalar statistics to keep fixed dimensionality.
    feats = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p25": float(np.percentile(flat, 25)),
        "p50": float(np.percentile(flat, 50)),
        "p75": float(np.percentile(flat, 75)),
        "first_frame_mean": float(np.mean(x[0])) if x.shape[0] > 0 else np.nan,
        "last_frame_mean": float(np.mean(x[-1])) if x.shape[0] > 0 else np.nan,
        "delta_last_first_mean": float(np.mean(x[-1] - x[0])) if x.shape[0] > 0 else np.nan,
    }
    if x.shape[0] >= 2:
        v = np.diff(x, axis=0).reshape(-1)
        feats.update(
            {
                "velocity_mean": float(np.mean(v)),
                "velocity_std": float(np.std(v)),
                "velocity_max_abs": float(np.max(np.abs(v))),
            }
        )
    else:
        feats.update({"velocity_mean": np.nan, "velocity_std": np.nan, "velocity_max_abs": np.nan})
    if conf is not None:
        c = np.nan_to_num(np.asarray(conf, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        feats.update(
            {
                "confidence_mean": float(np.mean(c)),
                "confidence_min": float(np.min(c)),
                "confidence_std": float(np.std(c)),
            }
        )
    names = list(feats.keys())
    values = np.asarray([feats[n] for n in names], dtype=np.float32)
    return values, names


def main() -> int:
    args = parse_args()
    rows = []
    with args.labels.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if _as_bool(r.get("valid_label", "false")):
                rows.append(r)

    X_rows: list[np.ndarray] = []
    out_paths: list[str] = []
    meta_rows: list[dict[str, Any]] = []
    key_counter = Counter()
    invalid_counter = Counter()
    feature_names: list[str] | None = None
    matched = 0

    for r in rows:
        npz_path = r["npz_path"]
        meta = {
            "npz_path": npz_path,
            "subject_id": r.get("subject_id", "unknown"),
            "camera_id": r.get("camera_id", "unknown"),
            "stage": r.get("stage", "unknown"),
            "fatigue_state": r.get("fatigue_state", "unknown"),
            "unique_impact_key_candidate": r.get("unique_impact_key_candidate", "unknown"),
            "feature_source_key": "unknown",
            "feature_valid": False,
            "feature_invalid_reason": "",
        }
        p = Path(npz_path)
        if not p.exists():
            meta["feature_invalid_reason"] = "npz_missing"
            invalid_counter["npz_missing"] += 1
            meta_rows.append(meta)
            continue
        try:
            with np.load(p, allow_pickle=True) as data:
                candidates = []
                for k in data.keys():
                    if not _is_candidate(k):
                        continue
                    arr = np.asarray(data[k])
                    if arr.size == 0:
                        continue
                    seq2d = _to_seq2d(arr)
                    if seq2d is None or seq2d.shape[0] == 0:
                        continue
                    candidates.append((k, arr, seq2d, _score_key(k, arr)))
                if not candidates:
                    meta["feature_invalid_reason"] = "no_pose_candidate_key"
                    invalid_counter["no_pose_candidate_key"] += 1
                    meta_rows.append(meta)
                    continue
                candidates.sort(key=lambda x: x[3], reverse=True)
                key, arr_raw, seq2d, _ = candidates[0]
                conf = _extract_confidence(data, np.asarray(arr_raw))
                values, names = _stats_features(seq2d, conf)
                if feature_names is None:
                    feature_names = names
                if names != feature_names:
                    meta["feature_invalid_reason"] = "feature_name_mismatch"
                    invalid_counter["feature_name_mismatch"] += 1
                    meta_rows.append(meta)
                    continue
                X_rows.append(values)
                out_paths.append(npz_path)
                key_counter[key] += 1
                matched += 1
                meta["feature_source_key"] = key
                meta["feature_valid"] = True
                meta_rows.append(meta)
        except Exception as exc:  # pragma: no cover
            reason = f"npz_read_error:{type(exc).__name__}"
            meta["feature_invalid_reason"] = reason
            invalid_counter[reason] += 1
            meta_rows.append(meta)

    X = np.vstack(X_rows).astype(np.float32) if X_rows else np.zeros((0, 0), dtype=np.float32)
    feature_names = feature_names or []
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        X=X,
        npz_path=np.asarray(out_paths, dtype=str),
        feature_names=np.asarray(feature_names, dtype=str),
    )

    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    with args.out_meta.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "npz_path",
                "subject_id",
                "camera_id",
                "stage",
                "fatigue_state",
                "unique_impact_key_candidate",
                "feature_source_key",
                "feature_valid",
                "feature_invalid_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    report = {
        "labels_valid_rows_input": len(rows),
        "feature_rows_built": int(X.shape[0]),
        "feature_dim": int(X.shape[1]) if X.ndim == 2 else 0,
        "matched_rows": matched,
        "feature_source_key_counts": dict(key_counter),
        "feature_invalid_reason_counts": dict(invalid_counter),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M3 Flat Pose Feature Report",
        "",
        f"- labels_valid_rows_input: {report['labels_valid_rows_input']}",
        f"- feature_rows_built: {report['feature_rows_built']}",
        f"- feature_dim: {report['feature_dim']}",
        f"- matched_rows: {report['matched_rows']}",
        "",
        "## Feature source key counts",
    ]
    for k, v in key_counter.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## Invalid reasons"])
    for k, v in invalid_counter.most_common():
        lines.append(f"- `{k}`: {v}")
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Embed machine-readable json tail for convenience.
    args.out_report.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Built features: rows={X.shape[0]} dim={X.shape[1] if X.ndim==2 else 0}")
    print(f"Wrote: {args.out_npz}")
    print(f"Wrote: {args.out_meta}")
    print(f"Wrote: {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
