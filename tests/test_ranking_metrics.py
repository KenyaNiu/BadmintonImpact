"""Checks for event aggregation and fixed review-budget metrics."""

from badminton_impact_ai.metrics import trial_top_fraction_metrics


def test_top_fraction_aggregates_views_before_ranking() -> None:
    rows = []
    for impact, label, scores in (
        ("a", 1, [0.9, 0.7]),
        ("b", 0, [0.8, 0.6]),
        ("c", 0, [0.2, 0.1]),
        ("d", 0, [0.3, 0.2]),
    ):
        for score in scores:
            rows.append(
                {
                    "fold": "s1",
                    "subject_id": "s1",
                    "trial_id": "trial1",
                    "unique_impact_key_candidate": impact,
                    "y_true_cls": label,
                    "y_prob": score,
                }
            )
    per_trial, summary = trial_top_fraction_metrics(rows, fraction=0.25)
    assert per_trial[0]["k"] == 1
    assert per_trial[0]["hits"] == 1
    assert summary["macro_precision_at_fraction"] == 1.0
    assert summary["macro_recall_at_fraction"] == 1.0
