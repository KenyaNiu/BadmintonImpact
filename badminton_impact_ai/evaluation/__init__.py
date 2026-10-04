"""Evaluation: pure metric and statistics functions (no file I/O, no training)."""

from .calibration import calibrate_fold_predictions
from .classification import aggregate_unique_impact, compute_binary_metrics, compute_ece, select_f1_threshold
from .paired import paired_fold_summary
from .ranking import trial_top_fraction_metrics
from .regression import compute_regression_metrics

__all__ = [
    "aggregate_unique_impact",
    "calibrate_fold_predictions",
    "compute_binary_metrics",
    "compute_ece",
    "compute_regression_metrics",
    "paired_fold_summary",
    "select_f1_threshold",
    "trial_top_fraction_metrics",
]
