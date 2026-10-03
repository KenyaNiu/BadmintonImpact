"""Checks explicit Wilcoxon method selection."""

from badminton_impact_ai.stats import paired_fold_summary


def test_exact_wilcoxon_without_zero_differences() -> None:
    result = paired_fold_summary({"a": 0.8, "b": 0.7, "c": 0.9}, {"a": 0.7, "b": 0.6, "c": 0.8}, 100, 1)
    assert result["wilcoxon_method"] == "exact"
    assert result["zero_differences"] == 0


def test_approximate_wilcoxon_with_zero_difference() -> None:
    result = paired_fold_summary({"a": 0.8, "b": 0.7, "c": 0.9}, {"a": 0.8, "b": 0.6, "c": 0.8}, 100, 1)
    assert result["wilcoxon_method"] == "approx"
    assert result["zero_differences"] == 1


def test_all_zero_differences_are_reported_without_error() -> None:
    result = paired_fold_summary({"a": 0.8, "b": 0.7}, {"a": 0.8, "b": 0.7}, 100, 1)
    assert result["wilcoxon_method"] == "all_zero"
    assert result["wilcoxon_p_two_sided"] == 1.0
