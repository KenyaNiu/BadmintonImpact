"""Deterministic group-safe split utilities."""

from __future__ import annotations

import random
from collections import defaultdict


def grouped_train_val_split(
    rows: list[dict[str, str]],
    val_ratio: float = 0.15,
    seed: int = 42,
    group_key: str = "unique_impact_key_candidate",
    stratify_key: str = "context_high_impact_q75",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between zero and one")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    rng = random.Random(seed)
    can_stratify = all(stratify_key in item for items in groups.values() for item in items)
    if can_stratify:
        strata: dict[str, list[str]] = defaultdict(list)
        for key, items in groups.items():
            labels = {str(item[stratify_key]) for item in items}
            if len(labels) != 1:
                raise ValueError(f"inconsistent {stratify_key} within group {key}")
            strata[next(iter(labels))].append(key)
        val_keys = set()
        for keys in strata.values():
            keys.sort()
            rng.shuffle(keys)
            n_val = max(1, int(len(keys) * val_ratio))
            if len(keys) > 1:
                n_val = min(n_val, len(keys) - 1)
            val_keys.update(keys[:n_val])
    else:
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_ratio))
        val_keys = set(keys[:n_val])
    train, val = [], []
    for key, items in groups.items():
        (val if key in val_keys else train).extend(items)
    if {row[group_key] for row in train} & {row[group_key] for row in val}:
        raise AssertionError("group leakage between train and validation")
    return train, val
