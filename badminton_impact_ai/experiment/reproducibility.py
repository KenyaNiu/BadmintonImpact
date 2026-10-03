"""Stable per-run randomness independent of execution order."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def derived_seed(base_seed: int, *parts: str) -> int:
    payload = ":".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
