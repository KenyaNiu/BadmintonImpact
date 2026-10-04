"""Validation-only probability calibration for canonical prediction rows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .classification import aggregate_unique_impact, compute_binary_metrics, select_f1_threshold


def _probability_logistic(labels: np.ndarray, scores: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000).fit(scores.reshape(-1, 1), labels.astype(int))
    return lambda values: model.predict_proba(values.reshape(-1, 1))[:, 1]


def _isotonic(labels: np.ndarray, scores: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
    return lambda values: np.asarray(model.predict(values), dtype=float)


def _temperature(labels: np.ndarray, logits: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    from scipy.optimize import minimize_scalar

    def loss(temperature: float) -> float:
        probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
        probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
        return float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)))

    temperature = float(minimize_scalar(loss, bounds=(0.05, 100.0), method="bounded").x)
    return lambda values: 1.0 / (1.0 + np.exp(-np.asarray(values) / temperature))


def calibrate_fold_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit each mapping on validation rows and evaluate on test rows."""
    val = [row for row in rows if row["split"] == "val"]
    test = [row for row in rows if row["split"] == "test"]
    if not val or not test:
        raise ValueError("calibration requires validation and test predictions")
    val_y = np.asarray([float(row["y_true_cls"]) for row in val])
    test_y = np.asarray([float(row["y_true_cls"]) for row in test])
    val_prob = np.asarray([float(row["y_prob"]) for row in val])
    test_prob = np.asarray([float(row["y_prob"]) for row in test])
    val_logit = np.asarray([float(row["cls_logit"]) for row in val])
    test_logit = np.asarray([float(row["cls_logit"]) for row in test])
    mappings = {
        "none": (lambda values: values, val_prob, test_prob),
        "probability_logistic": (_probability_logistic(val_y, val_prob), val_prob, test_prob),
        "temperature": (_temperature(val_y, val_logit), val_logit, test_logit),
        "isotonic": (_isotonic(val_y, val_prob), val_prob, test_prob),
    }
    output = []
    for method, (mapping, val_input, test_input) in mappings.items():
        calibrated_val = np.asarray(mapping(val_input), dtype=float)
        calibrated_test = np.asarray(mapping(test_input), dtype=float)
        threshold = select_f1_threshold(val_y, calibrated_val)
        view = compute_binary_metrics(test_y, calibrated_test, threshold)
        unique = aggregate_unique_impact(
            [
                {
                    "unique_impact_key_candidate": row["unique_impact_key_candidate"],
                    "y_true_cls": label,
                    "y_prob": score,
                }
                for row, label, score in zip(test, test_y, calibrated_test)
            ],
            pred_keys=("y_true_cls", "y_prob"),
        )
        for resolution, metrics in (
            ("view", view),
            ("unique_impact", compute_binary_metrics(unique["y_true_cls"], unique["y_prob"], threshold)),
        ):
            output.append({"calibration_method": method, "resolution": resolution, **metrics})
    return output
