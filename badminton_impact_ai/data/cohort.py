"""Canonical sample identity and eligibility rules."""

from __future__ import annotations

from typing import Any

import numpy as np

MAIN_TASK_TARGETS = ("context_high_impact_q75", "peak_fz_context_centered")


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("rel_path") or row.get("npz_path") or "")


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def exclusion_reason(
    row: dict[str, Any],
    require_targets: tuple[str, ...] = MAIN_TASK_TARGETS,
    valid_samples: set[str] | None = None,
) -> str:
    if row.get("subject_id", "unknown") == "unknown":
        return "unknown_subject"
    if valid_samples is not None and sample_id(row) not in valid_samples and row.get("npz_path") not in valid_samples:
        return "invalid_or_missing_features"
    missing = [target for target in require_targets if not _finite(row.get(target))]
    return f"invalid_target:{','.join(missing)}" if missing else ""


def filter_eligible_rows(
    rows: list[dict[str, str]],
    require_targets: tuple[str, ...] = MAIN_TASK_TARGETS,
    valid_paths: set[str] | None = None,
) -> list[dict[str, str]]:
    return [row for row in rows if not exclusion_reason(row, require_targets, valid_paths)]
