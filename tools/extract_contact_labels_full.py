#!/usr/bin/env python3
"""Full contact-state label extraction for model-ready table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.label_extraction import compute_contact_state_labels_from_fz
from data.metadata_parse import parse_npz_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract contact-state labels from all NPZ under data root.")
    parser.add_argument("--data-root", type=Path, default=Path("data/BadmintonGRF-data"))
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-audit", type=Path, required=True)
    return parser.parse_args()


def _extract_fz(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray | None, str | None, str | None]:
    if "grf_normalized" in data:
        arr = np.asarray(data["grf_normalized"])
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, 2], "grf_normalized", None
        return None, "grf_normalized", "grf_normalized_not_2d_or_missing_col2"
    if "grf_at_video_fps" in data:
        arr = np.asarray(data["grf_at_video_fps"])
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, 2], "grf_at_video_fps", None
        return None, "grf_at_video_fps", "grf_at_video_fps_not_2d_or_missing_col2"
    return None, None, "missing_grf_normalized_and_grf_at_video_fps"


def _to_py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max")}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: None for k in ("min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max")}
    return {
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    npz_paths = sorted(data_root.rglob("*.npz"))

    rows: list[dict[str, Any]] = []
    skip_reasons = Counter()
    invalid_reasons = Counter()
    source_grf_key_counts = Counter()
    subject_cov = Counter()
    camera_cov = Counter()
    stage_cov = Counter()
    fatigue_cov = Counter()
    unique_impact_keys = set()
    valid_count = 0
    invalid_count = 0

    for npz_path in npz_paths:
        rel_path = npz_path.relative_to(data_root).as_posix()
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                fz, source_key, skip_reason = _extract_fz(data)
                if source_key is not None:
                    source_grf_key_counts[source_key] += 1
                if fz is None:
                    skip_reasons[skip_reason or "unknown_skip_reason"] += 1
                    continue

                metadata_guess = {}
                for key in ("subject", "camera", "trial", "stage", "impact_id"):
                    if key in data:
                        metadata_guess[key] = _to_py(data[key])
                meta = parse_npz_metadata(npz_path=rel_path, npz_metadata=metadata_guess)

                labels = compute_contact_state_labels_from_fz(fz=fz, fps=args.fps)
                if labels["valid_label"]:
                    valid_count += 1
                else:
                    invalid_count += 1
                    invalid_reasons[str(labels["invalid_reason"])] += 1

                subject_cov[str(meta["subject_id"])] += 1
                camera_cov[str(meta["camera_id"])] += 1
                stage_cov[str(meta["stage"])] += 1
                fatigue_cov[str(meta["fatigue_state"])] += 1
                unique_impact_keys.add(str(meta["unique_impact_key_candidate"]))

                row = {
                    "npz_path": str(npz_path),
                    "rel_path": rel_path,
                    "source_grf_key": source_key,
                    "subject_id": meta["subject_id"],
                    "camera_id": meta["camera_id"],
                    "trial_id": meta["trial_id"],
                    "stage": meta["stage"],
                    "fatigue_state": meta["fatigue_state"],
                    "unique_impact_key_candidate": meta["unique_impact_key_candidate"],
                    "num_frames": labels["num_frames"],
                    "peak_fz": labels["peak_fz"],
                    "peak_fz_valid": labels["peak_fz_valid"],
                    "peak_index": labels["peak_index"],
                    "peak_time_sec_from_window_start": labels["peak_time_sec_from_window_start"],
                    "peak_time_sec_from_window_center": labels["peak_time_sec_from_window_center"],
                    "impulse_fz_full_window": labels["impulse_fz_full_window"],
                    "onset_index_proxy": labels["onset_index_proxy"],
                    "onset_time_sec_proxy": labels["onset_time_sec_proxy"],
                    "rise_frames_proxy": labels["rise_frames_proxy"],
                    "time_to_peak_from_onset_sec_proxy": labels["time_to_peak_from_onset_sec_proxy"],
                    "loading_rate_proxy": labels["loading_rate_proxy"],
                    "loading_rate_proxy_valid": labels["loading_rate_proxy_valid"],
                    "log1p_loading_rate_proxy": labels["log1p_loading_rate_proxy"],
                    "valid_label": labels["valid_label"],
                    "invalid_reason": labels["invalid_reason"],
                }
                rows.append(row)
        except Exception as exc:  # pragma: no cover
            skip_reasons[f"npz_read_error:{type(exc).__name__}"] += 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "npz_path",
        "rel_path",
        "source_grf_key",
        "subject_id",
        "camera_id",
        "trial_id",
        "stage",
        "fatigue_state",
        "unique_impact_key_candidate",
        "num_frames",
        "peak_fz",
        "peak_fz_valid",
        "peak_index",
        "peak_time_sec_from_window_start",
        "peak_time_sec_from_window_center",
        "impulse_fz_full_window",
        "onset_index_proxy",
        "onset_time_sec_proxy",
        "rise_frames_proxy",
        "time_to_peak_from_onset_sec_proxy",
        "loading_rate_proxy",
        "loading_rate_proxy_valid",
        "log1p_loading_rate_proxy",
        "valid_label",
        "invalid_reason",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [r for r in rows if bool(r["valid_label"])]
    peak_vals = [float(r["peak_fz"]) for r in valid_rows]
    impulse_vals = [float(r["impulse_fz_full_window"]) for r in valid_rows]
    ttp_vals = [float(r["time_to_peak_from_onset_sec_proxy"]) for r in valid_rows]
    loading_vals = [float(r["loading_rate_proxy"]) for r in valid_rows]
    log_loading_vals = [float(r["log1p_loading_rate_proxy"]) for r in valid_rows]

    audit = {
        "data_root": str(data_root),
        "total_npz": len(npz_paths),
        "rows_written": len(rows),
        "skipped_counts": dict(skip_reasons),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "source_grf_key_counts": dict(source_grf_key_counts),
        "subject_id_coverage": dict(subject_cov),
        "camera_id_coverage": dict(camera_cov),
        "stage_coverage": dict(stage_cov),
        "fatigue_state_coverage": dict(fatigue_cov),
        "unique_impact_key_candidate_unique_count": len(unique_impact_keys),
        "quantiles": {
            "peak_fz": _quantiles(peak_vals),
            "impulse_fz_full_window": _quantiles(impulse_vals),
            "time_to_peak_from_onset_sec_proxy": _quantiles(ttp_vals),
            "loading_rate_proxy": _quantiles(loading_vals),
            "log1p_loading_rate_proxy": _quantiles(log_loading_vals),
        },
        "top_invalid_reasons": invalid_reasons.most_common(10),
        "top_skip_reasons": skip_reasons.most_common(10),
    }

    args.out_audit.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Data root: {data_root}")
    print(f"total_npz={len(npz_paths)} rows_written={len(rows)} valid={valid_count} invalid={invalid_count}")
    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote audit: {args.out_audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
