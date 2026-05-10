"""Regression metric helpers."""

from __future__ import annotations

import numpy as np


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "Pearson_r": float("nan")}
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": _rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
        "Pearson_r": _pearson(y_true, y_pred),
    }
