"""Aggregate a completed run into fold summaries, paired comparisons, calibration and subgroup tables."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from badminton_impact_ai.evaluation import (
    aggregate_unique_impact,
    calibrate_fold_predictions,
    compute_binary_metrics,
    paired_fold_summary,
    select_f1_threshold,
)
from badminton_impact_ai.io import read_csv, write_csv, write_json

CANDIDATE, REFERENCE = "cn_hildnet", "hgb_pose_context"  # the paired comparison reported in the paper
REVIEW_METRICS = ("precision_at_fraction", "recall_at_fraction", "ndcg_at_fraction")
SUBGROUPS = (
    ("rally", lambda row: row.get("stage") == "rally"),
    ("non_rally", lambda row: row.get("stage") != "rally"),
    ("fresh", lambda row: row.get("fatigue_state") == "fresh"),
    ("fatigued", lambda row: row.get("fatigue_state") == "fatigued"),
)

ModelFold = tuple[str, str, str]  # (model, task_mode, fold)


def _auroc_summaries(metric_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Fold-wise AUROC mean and SD for every (model, task mode, resolution, task)."""
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        if row.get("AUROC") not in {None, "", "nan", "NaN"}:
            groups[(row["model"], row["task_mode"], row["resolution"], row["task"])].append(float(row["AUROC"]))
    return [
        {
            "model": model,
            "task_mode": task_mode,
            "resolution": resolution,
            "task": task,
            "fold_count": len(values),
            "AUROC_mean": float(np.mean(values)),
            "AUROC_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for (model, task_mode, resolution, task), values in sorted(groups.items())
    ]


def _paired(maps: dict[str, dict[str, float]], evaluation: dict[str, Any]) -> dict[str, Any] | None:
    if not (maps[CANDIDATE] and maps[REFERENCE]):
        return None
    return paired_fold_summary(
        maps[CANDIDATE],
        maps[REFERENCE],
        bootstrap_draws=int(evaluation["bootstrap_draws"]),
        seed=int(evaluation["statistical_seed"]),
    )


def _paired_auroc(metric_rows: list[dict[str, str]], evaluation: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for resolution in ("unique_impact", "view"):
        maps = {
            model: {
                row["fold"]: float(row["AUROC"])
                for row in metric_rows
                if row.get("model") == model
                and row.get("task_mode") == "cls_peak"
                and row.get("resolution") == resolution
                and row.get("task") == "classification"
            }
            for model in (CANDIDATE, REFERENCE)
        }
        if (summary := _paired(maps, evaluation)) is not None:
            comparisons[f"cn_minus_hgb_{resolution}"] = summary
    return comparisons


def _top_fraction_by_fold(run_dir: Path) -> list[dict[str, Any]]:
    """Trial-macro review-budget metrics averaged within each fold."""
    table = []
    for path in sorted((run_dir / "analysis").glob("*__top_fraction.csv")):
        model, task_mode, _ = path.stem.split("__", 2)
        rows = read_csv(path)
        for fold in sorted({row["fold"] for row in rows}):
            fold_rows = [row for row in rows if row["fold"] == fold]
            entry: dict[str, Any] = {"model": model, "task_mode": task_mode, "fold": fold}
            for metric in REVIEW_METRICS:
                values = np.asarray([float(row[metric]) for row in fold_rows], dtype=float)
                values = values[np.isfinite(values)]
                entry[metric] = float(np.mean(values)) if values.size else float("nan")
            table.append(entry)
    return table


def _paired_review(table: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for metric in REVIEW_METRICS:
        maps = {
            model: {
                row["fold"]: float(row[metric])
                for row in table
                if row["model"] == model and row["task_mode"] == "cls_peak" and np.isfinite(float(row[metric]))
            }
            for model in (CANDIDATE, REFERENCE)
        }
        if (summary := _paired(maps, evaluation)) is not None:
            comparisons[f"cn_minus_hgb_top20_{metric}"] = summary
    return comparisons


def _load_predictions(run_dir: Path) -> tuple[dict[ModelFold, list[dict[str, str]]], dict[str, set[tuple[str, str]]]]:
    """Prediction rows grouped by (model, task mode, fold) and the test sample set of every model."""
    by_model_fold: dict[ModelFold, list[dict[str, str]]] = defaultdict(list)
    test_sets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for path in sorted((run_dir / "predictions").glob("*.csv")):
        for row in read_csv(path):
            by_model_fold[(row["model"], row["task_mode"], row["fold"])].append(row)
            if row["split"] == "test":
                test_sets[f"{row['model']}/{row['task_mode']}"].add((row["fold"], row["sample_id"]))
    return by_model_fold, test_sets


def _calibration_table(by_model_fold: dict[ModelFold, list[dict[str, str]]]) -> list[dict[str, Any]]:
    table = []
    for (model, task_mode, fold), rows in by_model_fold.items():
        if model in {CANDIDATE, REFERENCE} and task_mode == "cls_peak":
            table.extend(
                {"model": model, "task_mode": task_mode, "fold": fold, **row}
                for row in calibrate_fold_predictions(rows)
            )
    return table


def _subgroup_table(by_model_fold: dict[ModelFold, list[dict[str, str]]]) -> list[dict[str, Any]]:
    """Held-out metrics per subgroup (rally/non-rally, fresh/fatigued) at the validation-selected threshold."""
    table = []
    for (model, task_mode, fold), rows in by_model_fold.items():
        validation = [row for row in rows if row["split"] == "val"]
        threshold = select_f1_threshold(
            np.asarray([float(row["y_true_cls"]) for row in validation]),
            np.asarray([float(row["y_prob"]) for row in validation]),
        )
        test = [row for row in rows if row["split"] == "test"]
        for subgroup, belongs in SUBGROUPS:
            subset = [row for row in test if belongs(row)]
            key = {"model": model, "task_mode": task_mode, "fold": fold, "subgroup": subgroup}
            if not subset:
                table.append({**key, "resolution": "view", "excluded_reason": "no_samples"})
                continue
            labels = np.asarray([float(row["y_true_cls"]) for row in subset])
            scores = np.asarray([float(row["y_prob"]) for row in subset])
            unique = aggregate_unique_impact(subset, pred_keys=("y_true_cls", "y_prob"))
            for resolution, metrics in (
                ("view", compute_binary_metrics(labels, scores, threshold)),
                ("unique_impact", compute_binary_metrics(unique["y_true_cls"], unique["y_prob"], threshold)),
            ):
                table.append(
                    {
                        **key,
                        "resolution": resolution,
                        "view_count": len(subset),
                        "unique_impact_count": len(unique["y_true_cls"]),
                        "excluded_reason": "single_class" if not np.isfinite(metrics["AUROC"]) else "",
                        **metrics,
                    }
                )
    return table


def analyze_run(run_dir: Path) -> Path:
    """Analyse a completed run (no retraining) and return the path of ``analysis/summary.json``."""
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError("analysis requires a completed run")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    analysis_dir = run_dir / "analysis"

    metric_rows = [row for path in sorted((run_dir / "metrics").glob("*.csv")) for row in read_csv(path)]
    review_table = _top_fraction_by_fold(run_dir)
    comparisons = {**_paired_auroc(metric_rows, evaluation), **_paired_review(review_table, evaluation)}

    by_model_fold, test_sets = _load_predictions(run_dir)
    if not (test_sets and len({frozenset(values) for values in test_sets.values()}) == 1):
        raise AssertionError("paper-facing models do not share one test cohort")

    write_csv(analysis_dir / "top_fraction_by_fold.csv", review_table)
    write_csv(analysis_dir / "calibration.csv", _calibration_table(by_model_fold))
    write_csv(analysis_dir / "subgroups.csv", _subgroup_table(by_model_fold))

    summary_path = analysis_dir / "summary.json"
    write_json(
        summary_path,
        {
            "run_dir": str(run_dir.resolve()),
            "cohort_match": True,
            "test_cohort_counts": {key: len(value) for key, value in test_sets.items()},
            "summaries": _auroc_summaries(metric_rows),
            "paired_comparisons": comparisons,
        },
    )
    return summary_path
