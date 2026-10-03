"""Stable experiment seeds must not depend on execution order."""

from badminton_impact_ai.experiment import derived_seed


def test_derived_seed_is_stable_and_scoped() -> None:
    assert derived_seed(42, "sub_001", "cn") == derived_seed(42, "sub_001", "cn")
    assert derived_seed(42, "sub_001", "cn") != derived_seed(42, "sub_002", "cn")
