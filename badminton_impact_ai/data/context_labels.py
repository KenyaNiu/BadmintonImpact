"""Fold-safe, event-weighted context-normalized supervision."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stats(values: list[float], percentile: float) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot compute context statistics from an empty event set")
    percentile = float(np.clip(percentile, 0.0, 100.0))
    std = float(np.std(array))
    return {
        "threshold": float(np.percentile(array, percentile)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": std if std > 1e-8 else 1.0,
        "count": int(array.size),
        "percentile_used": percentile,
    }


def _unique_events(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return one validated representative view per physical impact."""
    events: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("unique_impact_key_candidate", "")
        if not key:
            raise ValueError("missing unique_impact_key_candidate")
        previous = events.get(key)
        if previous is None:
            events[key] = row
            continue
        identity = ("subject_id", "stage", "fatigue_state", "trial_id")
        if any(previous.get(field) != row.get(field) for field in identity):
            raise ValueError(f"inconsistent metadata across views for impact {key}")
        if not np.isclose(float(previous["peak_fz"]), float(row["peak_fz"]), rtol=0.0, atol=1e-8):
            raise ValueError(f"inconsistent peak_fz across views for impact {key}")
    return list(events.values())


def build_context_normalized_loso(
    rows: list[dict[str, str]],
    percentile: float = 75.0,
    minimum_events: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create LOSO labels with every training impact weighted exactly once."""
    data = [
        row
        for row in rows
        if _truthy(row.get("valid_label"))
        and _truthy(row.get("peak_fz_valid"))
        and row.get("subject_id", "unknown") != "unknown"
    ]
    subjects = sorted({row["subject_id"] for row in data})
    output: list[dict[str, Any]] = []
    fold_summary: dict[str, Any] = {}
    warnings: list[str] = []

    for fold_index, test_subject in enumerate(subjects):
        fold_id = f"loso_{fold_index:03d}_{test_subject}"
        train_views = [row for row in data if row["subject_id"] != test_subject]
        test_views = [row for row in data if row["subject_id"] == test_subject]
        train_events = _unique_events(train_views)

        global_stats = _stats([float(row["peak_fz"]) for row in train_events], percentile)
        context_values: dict[str, list[float]] = defaultdict(list)
        stage_values: dict[str, list[float]] = defaultdict(list)
        for row in train_events:
            context = f"{row.get('stage', 'unknown')}|{row.get('fatigue_state', 'unknown')}"
            stage = row.get("stage", "unknown")
            context_values[context].append(float(row["peak_fz"]))
            stage_values[stage].append(float(row["peak_fz"]))
        context_stats = {key: _stats(values, percentile) for key, values in context_values.items()}
        stage_stats = {key: _stats(values, percentile) for key, values in stage_values.items()}

        context_usage: dict[str, dict[str, Any]] = {}
        fallback_views = 0
        for role, subset in (("train", train_views), ("test", test_views)):
            for row in subset:
                context = f"{row.get('stage', 'unknown')}|{row.get('fatigue_state', 'unknown')}"
                stage = row.get("stage", "unknown")
                if context in context_stats and int(context_stats[context]["count"]) >= minimum_events:
                    stat, source = context_stats[context], "context"
                elif stage in stage_stats and int(stage_stats[stage]["count"]) >= minimum_events:
                    stat, source = stage_stats[stage], "stage"
                else:
                    stat, source = global_stats, "global"
                fallback_views += source != "context"
                peak = float(row["peak_fz"])
                threshold = float(stat["threshold"])
                labeled = dict(row)
                labeled.update(
                    {
                        "fold_id": fold_id,
                        "test_subject": test_subject,
                        "role": role,
                        "context_key_v1": context,
                        "context_quantile": float(percentile) / 100.0,
                        "context_threshold": threshold,
                        "context_high_impact": int(peak >= threshold),
                        "context_threshold_q75": threshold,
                        "context_high_impact_q75": int(peak >= threshold),
                        "threshold_source": source,
                        "peak_fz_context_centered": peak - float(stat["median"]),
                        "peak_fz_context_z": (peak - float(stat["mean"])) / float(stat["std"]),
                    }
                )
                output.append(labeled)
                usage = context_usage.setdefault(context, {"used_source": source, "view_count": 0})
                usage["view_count"] += 1

        fold_rows = [row for row in output if row["fold_id"] == fold_id]
        train_labeled_events = _unique_events([row for row in fold_rows if row["role"] == "train"])
        test_labeled_events = _unique_events([row for row in fold_rows if row["role"] == "test"])
        fold_summary[fold_id] = {
            "test_subject": test_subject,
            "quantile_unit": "unique_impact",
            "global_train_stats": global_stats,
            "context_stats_train": context_stats,
            "stage_stats_train": stage_stats,
            "context_usage": context_usage,
            "train_positive_rate_unique": float(np.mean([row["context_high_impact"] for row in train_labeled_events])),
            "test_positive_rate_unique": float(np.mean([row["context_high_impact"] for row in test_labeled_events])),
            "train_unique_impacts": len(train_labeled_events),
            "test_unique_impacts": len(test_labeled_events),
            "fallback_view_count": int(fallback_views),
        }
        if any(value["used_source"] != "context" for value in context_usage.values()):
            warnings.append(f"{fold_id}: some contexts used fallback")

    return output, {
        "fold_count": len(subjects),
        "percentile": float(percentile),
        "minimum_events": int(minimum_events),
        "quantile_unit": "unique_impact",
        "folds": fold_summary,
        "warnings": warnings,
    }
