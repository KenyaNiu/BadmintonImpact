"""One config-driven, provenance-preserving LOSO experiment runner."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
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
from badminton_impact_ai.metrics import trial_top_fraction_metrics

from .models import DEEP_MODELS, HGB_MODELS
from .reproducibility import derived_seed
from .training import prepare_deep_fold, train_deep_fold, train_hgb_fold


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
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


def run_experiment(config: dict[str, Any], run_dir: Path, resume: bool = False, check_only: bool = False) -> None:
    run_dir = run_dir.resolve()
    status_path = run_dir / "status.json"
    if run_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists: {run_dir}; use --resume only for the same config")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(config, sort_keys=False)
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    stored_config = run_dir / "config.yaml"
    if stored_config.exists() and hashlib.sha256(stored_config.read_bytes()).hexdigest() != config_hash:
        raise ValueError("resume config differs from the run's resolved config")
    _atomic_text(stored_config, config_text)

    completed: set[str] = set()
    if resume and status_path.exists():
        completed = set(json.loads(status_path.read_text(encoding="utf-8")).get("completed", []))
    status: dict[str, Any] = {"state": "running", "config_sha256": config_hash, "completed": sorted(completed)}
    _write_json(status_path, status)

    try:
        data = config["data"]
        paths = {key: Path(data[key]) for key in ("labels", "features", "feature_meta")}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing experiment inputs: {missing}")
        _write_json(run_dir / "input_hashes.json", {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()})
        _write_json(run_dir / "environment.json", _environment())

        stat_map, _ = load_stat_features(paths["features"], paths["feature_meta"])
        with paths["labels"].open("r", encoding="utf-8", newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
        eligible = filter_eligible_rows(raw_rows, valid_paths=set(stat_map))
        configured_folds = config.get("folds")
        folds = list(configured_folds) if configured_folds else sorted({row["test_subject"] for row in eligible})
        selected = [row for row in eligible if row["test_subject"] in folds]
        test_rows = [row for row in selected if row["role"] == "test"]
        test_views = len(test_rows)
        test_impacts = len({row["unique_impact_key_candidate"] for row in test_rows})
        full_folds = configured_folds is None
        if full_folds and test_views != int(data["expected_test_views"]):
            raise AssertionError(f"test view count {test_views} != expected {data['expected_test_views']}")
        if full_folds and test_impacts != int(data["expected_test_impacts"]):
            raise AssertionError(f"test impact count {test_impacts} != expected {data['expected_test_impacts']}")

        manifest = []
        for row in raw_rows:
            reason = exclusion_reason(row, valid_samples=set(stat_map))
            manifest.append(
                {
                    "fold": row.get("test_subject", ""),
                    "outer_role": row.get("role", ""),
                    "sample_id": sample_id(row),
                    "subject_id": row.get("subject_id", ""),
                    "trial_id": row.get("trial_id", ""),
                    "camera_id": row.get("camera_id", ""),
                    "unique_impact_key_candidate": row.get("unique_impact_key_candidate", ""),
                    "eligible": not bool(reason),
                    "exclusion_reason": reason,
                    "y_true_cls": row.get("context_high_impact_q75", ""),
                    "y_true_peak": row.get("peak_fz_context_centered", ""),
                }
            )
        _write_csv(run_dir / "cohort.csv", manifest)

        base_seed = int(config["training"]["seed"])
        split_seed = int(config["training"].get("split_seed", base_seed))
        split_rows: list[dict[str, Any]] = []
        partitions: dict[str, tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]] = {}
        split_audit = []
        for fold in folds:
            fold_rows = [row for row in selected if row["test_subject"] == fold]
            outer_train = [row for row in fold_rows if row["role"] == "train"]
            outer_test = [row for row in fold_rows if row["role"] == "test"]
            inner_train, val = grouped_train_val_split(
                outer_train,
                val_ratio=float(config["training"]["val_ratio"]),
                seed=split_seed,
            )
            partitions[fold] = (inner_train, val, outer_test)
            for role, rows in (("train", inner_train), ("val", val), ("test", outer_test)):
                labels = {int(float(row["context_high_impact_q75"])) for row in rows}
                split_audit.append(
                    {
                        "fold": fold,
                        "split": role,
                        "view_count": len(rows),
                        "unique_impact_count": len({row["unique_impact_key_candidate"] for row in rows}),
                        "class_count": len(labels),
                    }
                )
                if role in {"train", "val", "test"} and len(labels) < 2:
                    raise AssertionError(f"{fold}/{role} contains fewer than two classes")
                split_rows.extend(
                    {
                        "fold": fold,
                        "split": role,
                        "sample_id": sample_id(row),
                        "unique_impact_key_candidate": row["unique_impact_key_candidate"],
                    }
                    for row in rows
                )
        _write_csv(run_dir / "splits.csv", split_rows)
        _write_json(
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
            status["state"] = "preflight_complete"
            _write_json(status_path, status)
            return

        device_name = str(config["training"].get("device", "auto"))
        device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))

        for fold in folds:
            inner_train, val, outer_test = partitions[fold]

            deep_models = [entry for entry in config["models"] if entry["name"] in DEEP_MODELS]
            deep_data = None
            if deep_models:
                deep_data = prepare_deep_fold(
                    inner_train,
                    val,
                    outer_test,
                    stat_map,
                    batch_size=int(config["training"]["batch_size"]),
                )
            expected_test_keys = {sample_id(row) for row in outer_test}

            for model_entry in config["models"]:
                model_name = str(model_entry["name"])
                task_mode = str(model_entry.get("task_mode", "cls_peak"))
                task_key = f"{fold}/{model_name}/{task_mode}"
                prediction_path = run_dir / "predictions" / f"{fold}__{model_name}__{task_mode}.csv"
                metric_path = run_dir / "metrics" / f"{fold}__{model_name}__{task_mode}.csv"
                if task_key in completed and prediction_path.exists() and metric_path.exists():
                    continue
                model_seed = derived_seed(base_seed, fold, model_name, task_mode)
                if model_name in HGB_MODELS:
                    result = train_hgb_fold(inner_train, val, outer_test, stat_map, model_name, fold, model_seed)
                else:
                    settings = {**config["training"], "task_mode": task_mode}
                    result = train_deep_fold(deep_data, model_name, fold, model_seed, settings, device)  # type: ignore[arg-type]
                for row in result["predictions"]:
                    row["run_id"] = run_dir.name
                    row["model_seed"] = model_seed
                actual_test_keys = {row["sample_id"] for row in result["predictions"] if row["split"] == "test"}
                if actual_test_keys != expected_test_keys:
                    raise AssertionError(f"test cohort mismatch for {task_key}")
                _write_csv(prediction_path, result["predictions"])
                _write_csv(metric_path, result["metrics"])
                _write_csv(run_dir / "logs" / f"{fold}__{model_name}__{task_mode}.csv", result["logs"])
                checkpoint_path = run_dir / "checkpoints" / f"{fold}__{model_name}__{task_mode}"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                if model_name in HGB_MODELS:
                    with checkpoint_path.with_suffix(".pkl").open("wb") as stream:
                        pickle.dump(result["checkpoint"], stream)
                else:
                    torch.save(result["checkpoint"], checkpoint_path.with_suffix(".pt"))
                completed.add(task_key)
                status["completed"] = sorted(completed)
                _write_json(status_path, status)

        for model_entry in config["models"]:
            model_name = str(model_entry["name"])
            task_mode = str(model_entry.get("task_mode", "cls_peak"))
            predictions = []
            for path in sorted((run_dir / "predictions").glob(f"*__{model_name}__{task_mode}.csv")):
                with path.open("r", encoding="utf-8", newline="") as stream:
                    predictions.extend(row for row in csv.DictReader(stream) if row["split"] == "test")
            per_trial, summary = trial_top_fraction_metrics(predictions, float(config["evaluation"]["top_fraction"]))
            _write_csv(run_dir / "analysis" / f"{model_name}__{task_mode}__top_fraction.csv", per_trial)
            _write_json(run_dir / "analysis" / f"{model_name}__{task_mode}__top_fraction.json", summary)

        status["state"] = "complete"
        status["completed"] = sorted(completed)
        _write_json(status_path, status)
    except Exception as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        _write_json(status_path, status)
        raise
