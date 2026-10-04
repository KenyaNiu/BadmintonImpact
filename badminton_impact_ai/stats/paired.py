"""Fold-level paired summaries used by the corrected evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def paired_fold_summary(
    values_a: dict[str, float],
    values_b: dict[str, float],
    bootstrap_draws: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    folds = sorted(set(values_a) & set(values_b))
    if not folds:
        raise ValueError("paired comparison has no overlapping folds")
    differences = np.asarray([values_a[fold] - values_b[fold] for fold in folds], dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            float(np.mean(differences[rng.integers(0, len(differences), len(differences))]))
            for _ in range(bootstrap_draws)
        ]
    )
    method = "exact" if not np.any(differences == 0) else "approx"
    if np.all(differences == 0):
        statistic, p_value, method = 0.0, 1.0, "all_zero"
    else:
        result = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method=method)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "folds": folds,
        "n_folds": len(folds),
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
        "bootstrap_ci_95": [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])],
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(seed),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_two_sided": p_value,
        "wilcoxon_method": method,
        "zero_differences": int(np.sum(differences == 0)),
        "per_fold": [
            {"fold": fold, "value_a": values_a[fold], "value_b": values_b[fold], "difference": float(difference)}
            for fold, difference in zip(folds, differences)
        ],
    }
