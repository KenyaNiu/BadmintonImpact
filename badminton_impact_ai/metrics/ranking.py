"""Coach-facing review-budget metrics at physical-impact level."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def trial_top_fraction_metrics(
    rows: list[dict[str, Any]],
    fraction: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Aggregate camera views, then score the top fraction within each held-out trial."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    impacts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        impacts[(str(row["fold"]), str(row["unique_impact_key_candidate"]))].append(row)

    trials: dict[tuple[str, str, str], list[tuple[str, float, int]]] = defaultdict(list)
    for (fold, impact), views in impacts.items():
        labels = np.asarray([float(view["y_true_cls"]) for view in views])
        if not np.allclose(labels, labels[0]):
            raise ValueError(f"inconsistent labels across views for {impact}")
        subject = str(views[0]["subject_id"])
        trial = str(views[0]["trial_id"])
        if any(str(view["subject_id"]) != subject or str(view["trial_id"]) != trial for view in views):
            raise ValueError(f"inconsistent trial metadata across views for {impact}")
        score = float(np.mean([float(view["y_prob"]) for view in views]))
        trials[(fold, subject, trial)].append((impact, score, int(labels[0])))

    per_trial: list[dict[str, Any]] = []
    for (fold, subject, trial), candidates in sorted(trials.items()):
        ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
        k = max(1, math.ceil(len(ranked) * fraction))
        selected = ranked[:k]
        positives = sum(item[2] for item in ranked)
        hits = sum(item[2] for item in selected)
        dcg = sum(item[2] / math.log2(rank + 2) for rank, item in enumerate(selected))
        ideal_hits = min(positives, k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        per_trial.append(
            {
                "fold": fold,
                "subject_id": subject,
                "trial_id": trial,
                "n_impacts": len(ranked),
                "k": k,
                "n_positive": positives,
                "hits": hits,
                "precision_at_fraction": hits / k,
                "recall_at_fraction": hits / positives if positives else float("nan"),
                "ndcg_at_fraction": dcg / idcg if idcg else float("nan"),
            }
        )

    def finite_mean(key: str) -> float:
        values = np.asarray([float(row[key]) for row in per_trial], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")

    return per_trial, {
        "fraction": float(fraction),
        "trial_count": len(per_trial),
        "trials_without_positives": sum(int(row["n_positive"] == 0) for row in per_trial),
        "macro_precision_at_fraction": finite_mean("precision_at_fraction"),
        "macro_recall_at_fraction": finite_mean("recall_at_fraction"),
        "macro_ndcg_at_fraction": finite_mean("ndcg_at_fraction"),
    }
