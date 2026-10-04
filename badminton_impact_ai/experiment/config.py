"""Experiment configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from badminton_impact_ai.models import DEEP_MODELS, HGB_MODELS

REQUIRED_SECTIONS = ("data", "training", "models", "evaluation")
REQUIRED_DATA_KEYS = ("labels", "features", "feature_meta", "expected_test_views", "expected_test_impacts")


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"missing config section: {section}")
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValueError("models must be a non-empty list")
    allowed = DEEP_MODELS | HGB_MODELS
    for model in config["models"]:
        if not isinstance(model, dict) or model.get("name") not in allowed:
            raise ValueError(f"invalid model entry: {model}")
        if model["name"] in HGB_MODELS and model.get("task_mode", "cls_peak") != "cls_peak":
            raise ValueError("HGB models support task_mode=cls_peak only")
    missing = [key for key in REQUIRED_DATA_KEYS if key not in config["data"]]
    if missing:
        raise ValueError(f"missing data settings: {missing}")
    return config
