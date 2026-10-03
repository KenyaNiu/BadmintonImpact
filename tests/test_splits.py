"""Checks for deterministic group-safe inner splits."""

from badminton_impact_ai.data.splits import grouped_train_val_split


def test_grouped_split_is_stable_and_disjoint() -> None:
    rows = [{"unique_impact_key_candidate": f"impact_{i // 2}", "view": str(i)} for i in range(20)]
    train_a, val_a = grouped_train_val_split(rows, val_ratio=0.2, seed=7)
    train_b, val_b = grouped_train_val_split(list(reversed(rows)), val_ratio=0.2, seed=7)
    assert {row["view"] for row in train_a} == {row["view"] for row in train_b}
    assert {row["view"] for row in val_a} == {row["view"] for row in val_b}
    assert not ({row["unique_impact_key_candidate"] for row in train_a} & {row["unique_impact_key_candidate"] for row in val_a})
