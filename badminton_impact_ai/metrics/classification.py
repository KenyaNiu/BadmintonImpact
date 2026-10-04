"""Binary-classification metrics, F1 threshold selection and view -> physical-impact aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        upper_ok = y_prob <= hi if i == n_bins - 1 else y_prob < hi  # the last bin is closed on the right
        mask = (y_prob >= lo) & upper_ok
        if np.sum(mask) == 0:
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / y_true.size) * abs(acc - conf)
    return float(ece)


F1_THRESHOLD_GRID = np.linspace(0.05, 0.95, 37)


def select_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Operating threshold with the best F1 on a fixed 37-point grid (used on validation data only)."""
    from sklearn.metrics import f1_score

    return float(max(F1_THRESHOLD_GRID, key=lambda t: f1_score(y_true, y_prob >= t, zero_division=0)))


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "AUROC": float("nan"),
        "AUPRC": float("nan"),
        "F1": float("nan"),
        "balanced_accuracy": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "Brier": float("nan"),
        "ECE": float("nan"),
        "positive_rate_test": float("nan"),
        "selected_threshold_from_val": float(threshold),
    }
    if y_true.size == 0:
        return out
    out["positive_rate_test"] = float(np.mean(y_true))
    if len(np.unique(y_true)) >= 2:
        out["AUROC"] = float(roc_auc_score(y_true, y_prob))
        out["AUPRC"] = float(average_precision_score(y_true, y_prob))
    out["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    out["Brier"] = float(brier_score_loss(y_true, y_prob))
    out["ECE"] = compute_ece(y_true, y_prob, n_bins=10)
    return out


def aggregate_unique_impact(
    rows: list[dict[str, Any]],
    group_col: str = "unique_impact_key_candidate",
    pred_keys: tuple[str, ...] = ("y_true", "y_prob"),
) -> dict[str, np.ndarray]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = str(r[group_col])
        for k in pred_keys:
            grouped[g][k].append(float(r[k]))
    out = {k: [] for k in pred_keys}
    for g in grouped:
        for k in pred_keys:
            if k.startswith("y_true") and not np.allclose(grouped[g][k], grouped[g][k][0]):
                raise ValueError(f"inconsistent {k} values within unique impact {g}")
            out[k].append(float(np.mean(grouped[g][k])))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}
