"""Classification metrics and calibration helpers."""

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
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if np.sum(mask) == 0:
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / y_true.size) * abs(acc - conf)
    return float(ece)


def compute_ece_equal_frequency(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-mass (quantile) bins on predicted probabilities.

    Sorts by ``y_prob`` and partitions into ``n_bins`` groups with (approximately) equal counts,
    then applies the standard ECE weighting |acc - conf| per bin. Reduces sensitivity to empty or
    dominant equal-width intervals when the score distribution is skewed.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = y_true.size
    if n == 0:
        return float("nan")
    order = np.argsort(y_prob)
    y_s = y_true[order]
    p_s = y_prob[order]
    ece = 0.0
    for i in range(n_bins):
        lo = i * n // n_bins
        hi = (i + 1) * n // n_bins if i < n_bins - 1 else n
        if hi <= lo:
            continue
        acc = float(np.mean(y_s[lo:hi]))
        conf = float(np.mean(p_s[lo:hi]))
        ece += ((hi - lo) / n) * abs(acc - conf)
    return float(ece)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

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
    prob_key: str = "y_prob",
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
            out[k].append(float(np.mean(grouped[g][k])))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}
