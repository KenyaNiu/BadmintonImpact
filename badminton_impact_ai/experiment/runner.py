"""Config-driven, provenance-preserving leave-one-subject-out experiment runner.

A run directory records the resolved config, input hashes, environment, cohort manifest, splits, per-fold
predictions/metrics/checkpoints and a ``status.json`` that makes ``--resume`` safe.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from badminton_impact_ai.data.cohort import exclusion_reason, filter_eligible_rows, sample_id
from badminton_impact_ai.data.sequence_dataset import load_stat_features
from badminton_impact_ai.data.splits import grouped_train_val_split
from badminton_impact_ai.evaluation import trial_top_fraction_metrics
from badminton_impact_ai.io import atomic_write_text, read_csv, sha256_file, write_csv, write_json
from badminton_impact_ai.models import DEEP_MODELS, HGB_MODELS

from .reproducibility import derived_seed
from .training import TrainSettings, prepare_deep_fold, train_deep_fold, train_hgb_fold

Rows = list[dict[str, str]]
Partition = tuple[Rows, Rows, Rows]  # (inner train, validation, held-out test)


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class _RunStatus:
    """``status.json``: run state plus the set of completed (fold, model, task) units."""

    def __init__(self, path: Path, config_sha256: str, completed: set[str]) -> None:
        self.path = path
        self.completed = completed
        self.data: dict[str, Any] = {"state": "running", "config_sha256": config_sha256, "completed": sorted(completed)}
        self._save()

    def _save(self) -> None:
        write_json(self.path, self.data)

    def set_state(self, state: str, **extra: Any) -> None:
        self.data.update(state=state, completed=sorted(self.completed), **extra)
        self._save()

    def mark_done(self, task_key: str) -> None:
        self.completed.add(task_key)
        self.data["completed"] = sorted(self.completed)
        self._save()


def _init_run_dir(config: dict[str, Any], run_dir: Path, resume: bool) -> _RunStatus:
    if run_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists: {run_dir}; use --resume only for the same config")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(config, sort_keys=False)
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    stored = run_dir / "config.yaml"
    if stored.exists() and hashlib.sha256(stored.read_bytes()).hexdigest() != config_hash:
        raise ValueError("resume config differs from the run's resolved config")
    atomic_write_text(stored, config_text)
    status_path = run_dir / "status.json"
    completed = (
        set(json.loads(status_path.read_text(encoding="utf-8")).get("completed", []))
        if resume and status_path.exists()
        else set()
    )
    return _RunStatus(status_path, config_hash, completed)


def _cohort_manifest(raw_rows: Rows, valid_samples: set[str]) -> list[dict[str, Any]]:
    manifest = []
    for row in raw_rows:
        reason = exclusion_reason(row, valid_samples=valid_samples)
        manifest.append(
            {
                "fold": row.get("test_subject", ""),
                "outer_role": row.get("role", ""),
                "sample_id": sample_id(row),
                "subject_id": row.get("subject_id", ""),
                "trial_id": row.get("trial_id", ""),
                "camera_id": row.get("camera_id", ""),
                "unique_impact_key_candidate": row.get("unique_impact_key_candidate", ""),
                "eligible": not reason,
                "exclusion_reason": reason,
                "y_true_cls": row.get("context_high_impact_q75", ""),
                "y_true_peak": row.get("peak_fz_context_centered", ""),
            }
        )
    return manifest


def _build_partitions(
    selected: Rows, folds: list[str], training: dict[str, Any]
) -> tuple[dict[str, Partition], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split every outer fold into train / validation / test; return partitions, split table and audit rows."""
    split_seed = int(training.get("split_seed", training["seed"]))
    partitions: dict[str, Partition] = {}
    split_rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for fold in folds:
        fold_rows = [row for row in selected if row["test_subject"] == fold]
        outer_train = [row for row in fold_rows if row["role"] == "train"]
        outer_test = [row for row in fold_rows if row["role"] == "test"]
        inner_train, val = grouped_train_val_split(outer_train, val_ratio=float(training["val_ratio"]), seed=split_seed)
        partitions[fold] = (inner_train, val, outer_test)
        for role, rows in (("train", inner_train), ("val", val), ("test", outer_test)):
            classes = {int(float(row["context_high_impact_q75"])) for row in rows}
            if len(classes) < 2:
                raise AssertionError(f"{fold}/{role} contains fewer than two classes")
            audit.append(
                {
                    "fold": fold,
                    "split": role,
                    "view_count": len(rows),
                    "unique_impact_count": len({row["unique_impact_key_candidate"] for row in rows}),
                    "class_count": len(classes),
                }
            )
            split_rows.extend(
                {
                    "fold": fold,
                    "split": role,
                    "sample_id": sample_id(row),
                    "unique_impact_key_candidate": row["unique_impact_key_candidate"],
                }
                for row in rows
            )
    return partitions, split_rows, audit


def _save_checkpoint(run_dir: Path, stem: str, model_name: str, checkpoint: dict[str, Any]) -> None:
    path = run_dir / "checkpoints" / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name in HGB_MODELS:
        with path.with_suffix(".pkl").open("wb") as stream:
            pickle.dump(checkpoint, stream)
    else:
        torch.save(checkpoint, path.with_suffix(".pt"))


def _write_top_fraction_reports(run_dir: Path, config: dict[str, Any]) -> None:
    """Review-budget metrics (top fraction of every held-out trial) for each configured model."""
    for entry in config["models"]:
        model_name, task_mode = str(entry["name"]), str(entry.get("task_mode", "cls_peak"))
        predictions = []
        for path in sorted((run_dir / "predictions").glob(f"*__{model_name}__{task_mode}.csv")):
            predictions.extend(row for row in read_csv(path) if row["split"] == "test")
        per_trial, summary = trial_top_fraction_metrics(predictions, float(config["evaluation"]["top_fraction"]))
        write_csv(run_dir / "analysis" / f"{model_name}__{task_mode}__top_fraction.csv", per_trial)
        write_json(run_dir / "analysis" / f"{model_name}__{task_mode}__top_fraction.json", summary)


def run_experiment(config: dict[str, Any], run_dir: Path, resume: bool = False, check_only: bool = False) -> None:
    """Validate the cohort and splits and (unless ``check_only``) train and evaluate every configured model."""
    run_dir = run_dir.resolve()
    status = _init_run_dir(config, run_dir, resume)
    try:
        _run(config, run_dir, status, check_only)
    except Exception as error:
        status.set_state("failed", error=f"{type(error).__name__}: {error}")
        raise


def _run(config: dict[str, Any], run_dir: Path, status: _RunStatus, check_only: bool) -> None:
    data, training = config["data"], config["training"]
    paths = {key: Path(data[key]) for key in ("labels", "features", "feature_meta")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing experiment inputs: {missing}")
    write_json(
        run_dir / "input_hashes.json", {name: {"path": str(p), "sha256": sha256_file(p)} for name, p in paths.items()}
    )
    write_json(run_dir / "environment.json", _environment())

    # ---- cohort: one eligibility rule for every model
    stat_map = load_stat_features(paths["features"], paths["feature_meta"])
    raw_rows = read_csv(paths["labels"])
    eligible = filter_eligible_rows(raw_rows, valid_paths=set(stat_map))
    configured_folds = config.get("folds")
    folds = list(configured_folds) if configured_folds else sorted({row["test_subject"] for row in eligible})
    selected = [row for row in eligible if row["test_subject"] in folds]
    test_rows = [row for row in selected if row["role"] == "test"]
    test_views = len(test_rows)
    test_impacts = len({row["unique_impact_key_candidate"] for row in test_rows})
    if configured_folds is None:  # the frozen expectations only apply to the full ten-fold run
        if test_views != int(data["expected_test_views"]):
            raise AssertionError(f"test view count {test_views} != expected {data['expected_test_views']}")
        if test_impacts != int(data["expected_test_impacts"]):
            raise AssertionError(f"test impact count {test_impacts} != expected {data['expected_test_impacts']}")
    write_csv(run_dir / "cohort.csv", _cohort_manifest(raw_rows, set(stat_map)))

    # ---- splits and preflight
    partitions, split_rows, split_audit = _build_partitions(selected, folds, training)
    write_csv(run_dir / "splits.csv", split_rows)
    write_json(
        run_dir / "preflight.json",
        {
            "folds": folds,
            "eligible_test_views": test_views,
            "eligible_test_impacts": test_impacts,
            "split_audit": split_audit,
            "passed": True,
        },
    )
    if check_only:
        status.set_state("preflight_complete")
        return

    # ---- training
    device = _resolve_device(str(training.get("device", "auto")))
    base_seed = int(training["seed"])
    has_deep_models = any(entry["name"] in DEEP_MODELS for entry in config["models"])
    for fold in folds:
        inner_train, val, outer_test = partitions[fold]
        deep_data = (
            prepare_deep_fold(inner_train, val, outer_test, stat_map, batch_size=int(training["batch_size"]))
            if has_deep_models
            else None
        )
        expected_test_keys = {sample_id(row) for row in outer_test}
        for entry in config["models"]:
            model_name, task_mode = str(entry["name"]), str(entry.get("task_mode", "cls_peak"))
            stem = f"{fold}__{model_name}__{task_mode}"
            task_key = f"{fold}/{model_name}/{task_mode}"
            prediction_path, metric_path = run_dir / "predictions" / f"{stem}.csv", run_dir / "metrics" / f"{stem}.csv"
            if task_key in status.completed and prediction_path.exists() and metric_path.exists():
                continue
            model_seed = derived_seed(base_seed, fold, model_name, task_mode)
            if model_name in HGB_MODELS:
                result = train_hgb_fold(inner_train, val, outer_test, stat_map, model_name, fold, model_seed)
            else:
                settings = TrainSettings.from_config(training, task_mode)
                result = train_deep_fold(deep_data, model_name, fold, model_seed, settings, device)
            for row in result["predictions"]:
                row["run_id"], row["model_seed"] = run_dir.name, model_seed
            if {row["sample_id"] for row in result["predictions"] if row["split"] == "test"} != expected_test_keys:
                raise AssertionError(f"test cohort mismatch for {task_key}")
            write_csv(prediction_path, result["predictions"])
            write_csv(metric_path, result["metrics"])
            write_csv(run_dir / "logs" / f"{stem}.csv", result["logs"])
            _save_checkpoint(run_dir, stem, model_name, result["checkpoint"])
            status.mark_done(task_key)

    _write_top_fraction_reports(run_dir, config)
    status.set_state("complete")
