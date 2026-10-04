"""Checks that context supervision is defined over physical impacts."""

from __future__ import annotations

from badminton_impact_ai.data.context_labels import build_context_normalized_loso


def _views(subject: str, impact: str, peak: float, count: int) -> list[dict[str, str]]:
    return [
        {
            "subject_id": subject,
            "unique_impact_key_candidate": f"{subject}_{impact}",
            "trial_id": "trial",
            "stage": "stage",
            "fatigue_state": "fresh",
            "peak_fz": str(peak),
            "valid_label": "true",
            "peak_fz_valid": "true",
            "camera_id": f"cam{i}",
        }
        for i in range(count)
    ]


def test_quantile_weights_each_impact_once() -> None:
    rows = []
    rows += _views("s1", "a", 1.0, 8)
    rows += _views("s1", "b", 9.0, 6)
    rows += _views("s2", "a", 2.0, 8)
    rows += _views("s2", "b", 10.0, 6)
    labeled, summary = build_context_normalized_loso(rows, percentile=50.0, minimum_events=1)
    fold = next(value for value in summary["folds"].values() if value["test_subject"] == "s2")
    assert fold["context_stats_train"]["stage|fresh"]["threshold"] == 5.0
    assert fold["context_stats_train"]["stage|fresh"]["count"] == 2
    assert {
        row["context_high_impact"] for row in labeled if row["test_subject"] == "s2" and row["peak_fz"] == "2.0"
    } == {0}
