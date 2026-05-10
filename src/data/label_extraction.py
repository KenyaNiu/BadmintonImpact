"""Contact-state label extraction utilities (training-free)."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_contact_state_labels_from_fz(fz: np.ndarray, fps: float = 120.0) -> dict[str, Any]:
    """Compute contact-state labels from BW-normalized vertical GRF.

    Notes:
    - `loading_rate_proxy` is a proxy metric only.
    - This function does not produce high-impact labels (fold-threshold required).
    """
    result: dict[str, Any] = {
        "num_frames": 0,
        "fps": float(fps),
        "peak_fz": np.nan,
        "peak_index": -1,
        "peak_time_sec_from_window_start": np.nan,
        "peak_time_sec_from_window_center": np.nan,
        "impulse_fz_full_window": np.nan,
        "onset_index_proxy": -1,
        "onset_time_sec_proxy": np.nan,
        "onset_threshold": np.nan,
        "time_to_peak_from_onset_sec_proxy": np.nan,
        "rise_frames_proxy": np.nan,
        "loading_rate_proxy": np.nan,
        "loading_rate_proxy_valid": False,
        "log1p_loading_rate_proxy": np.nan,
        "peak_fz_valid": False,
        "valid_label": False,
        "invalid_reason": "",
    }

    if fz is None:
        result["invalid_reason"] = "fz_is_none"
        return result
    if fps <= 0:
        result["invalid_reason"] = "invalid_fps"
        return result

    arr = np.asarray(fz, dtype=float).reshape(-1)
    t = arr.size
    result["num_frames"] = int(t)
    if t == 0:
        result["invalid_reason"] = "empty_fz"
        return result
    if np.all(np.isnan(arr)):
        result["invalid_reason"] = "all_nan_fz"
        return result

    dt = 1.0 / float(fps)
    safe_arr = np.nan_to_num(arr, nan=-np.inf)
    peak_index = int(np.argmax(safe_arr))
    if peak_index < 0 or peak_index >= t:
        result["invalid_reason"] = "invalid_peak_index"
        return result
    peak_fz = float(arr[peak_index])
    if np.isnan(peak_fz):
        result["invalid_reason"] = "nan_peak_value"
        return result

    positive_arr = np.maximum(np.nan_to_num(arr, nan=0.0), 0.0)
    impulse = float(np.trapezoid(positive_arr, dx=dt))
    onset_threshold = float(max(0.10, 0.05 * peak_fz))
    pre_peak = arr[: peak_index + 1]
    candidate_idx = np.where(np.nan_to_num(pre_peak, nan=-np.inf) >= onset_threshold)[0]
    if candidate_idx.size == 0:
        result["invalid_reason"] = "onset_not_found"
        result["peak_fz"] = peak_fz
        result["peak_index"] = peak_index
        result["onset_threshold"] = onset_threshold
        result["impulse_fz_full_window"] = impulse
        return result
    onset_index = int(candidate_idx[0])
    onset_val = float(arr[onset_index])
    if np.isnan(onset_val):
        result["invalid_reason"] = "nan_onset_value"
        return result

    time_to_peak = (peak_index - onset_index) * dt
    if time_to_peak <= 0:
        result["invalid_reason"] = "non_positive_time_to_peak"
        result["peak_fz"] = peak_fz
        result["peak_index"] = peak_index
        result["onset_index_proxy"] = onset_index
        result["onset_threshold"] = onset_threshold
        result["impulse_fz_full_window"] = impulse
        result["peak_fz_valid"] = bool(np.isfinite(peak_fz) and peak_fz > 0.0)
        return result

    loading_rate_proxy = float((peak_fz - onset_val) / time_to_peak)
    rise_frames_proxy = int(peak_index - onset_index)
    loading_rate_proxy_valid = bool(np.isfinite(loading_rate_proxy) and rise_frames_proxy >= 3)
    log1p_loading_rate_proxy = (
        float(np.log1p(max(loading_rate_proxy, 0.0))) if loading_rate_proxy_valid else float(np.nan)
    )
    peak_fz_valid = bool(np.isfinite(peak_fz) and peak_fz > 0.0)

    result.update(
        {
            "peak_fz": peak_fz,
            "peak_index": peak_index,
            "peak_time_sec_from_window_start": float(peak_index * dt),
            "peak_time_sec_from_window_center": float((peak_index - (t - 1) / 2.0) * dt),
            "impulse_fz_full_window": impulse,
            "onset_index_proxy": onset_index,
            "onset_time_sec_proxy": float(onset_index * dt),
            "onset_threshold": onset_threshold,
            "time_to_peak_from_onset_sec_proxy": float(time_to_peak),
            "rise_frames_proxy": rise_frames_proxy,
            "loading_rate_proxy": loading_rate_proxy,
            "loading_rate_proxy_valid": loading_rate_proxy_valid,
            "log1p_loading_rate_proxy": log1p_loading_rate_proxy,
            "peak_fz_valid": peak_fz_valid,
            "valid_label": True,
            "invalid_reason": "",
        }
    )
    return result
