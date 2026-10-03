"""Aggregate one completed canonical run without retraining."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from badminton_impact_ai.metrics import aggregate_unique_impact, calibrate_fold_predictions, compute_binary_metrics
from badminton_impact_ai.stats import paired_fold_summary


def _read_csvs(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def analyze_run(run_dir: Path) -> Path:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError("analysis requires a completed run")
    config = __import__("yaml").safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metric_rows = _read_csvs(sorted((run_dir / "metrics").glob("*.csv")))
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        if row.get("AUROC") not in {None, "", "nan", "NaN"}:
            groups[(row["model"], row["task_mode"], row["resolution"], row["task"])].append(float(row["AUROC"]))
    summaries = [
        {
            "model": key[0],
            "task_mode": key[1],
            "resolution": key[2],
            "task": key[3],
            "fold_count": len(values),
            "AUROC_mean": float(np.mean(values)),
            "AUROC_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for key, values in sorted(groups.items())
    ]

    comparisons: dict[str, Any] = {}
    for resolution in ("unique_impact", "view"):
        maps = {}
        for model in ("cn_hildnet", "hgb_pose_context"):
            maps[model] = {
                row["fold"]: float(row["AUROC"])
                for row in metric_rows
                if row.get("model") == model
                and row.get("task_mode") == "cls_peak"
                and row.get("resolution") == resolution
                and row.get("task") == "classification"
            }
        if maps["cn_hildnet"] and maps["hgb_pose_context"]:
            comparisons[f"cn_minus_hgb_{resolution}"] = paired_fold_summary(
                maps["cn_hildnet"],
                maps["hgb_pose_context"],
                bootstrap_draws=int(config["evaluation"]["bootstrap_draws"]),
                seed=int(config["evaluation"]["statistical_seed"]),
            )

    top_fraction_rows = []
    for path in sorted((run_dir / "analysis").glob("*__top_fraction.csv")):
        model, task_mode, _ = path.stem.split("__", 2)
        rows = _read_csvs([path])
        for fold in sorted({row["fold"] for row in rows}):
            fold_rows = [row for row in rows if row["fold"] == fold]
            summary_row: dict[str, Any] = {"model": model, "task_mode": task_mode, "fold": fold}
            for metric in ("precision_at_fraction", "recall_at_fraction", "ndcg_at_fraction"):
                values = np.asarray([float(row[metric]) for row in fold_rows], dtype=float)
                values = values[np.isfinite(values)]
                summary_row[metric] = float(np.mean(values)) if values.size else float("nan")
            top_fraction_rows.append(summary_row)
    for metric in ("precision_at_fraction", "recall_at_fraction", "ndcg_at_fraction"):
        maps = {
            model: {
                row["fold"]: float(row[metric])
                for row in top_fraction_rows
                if row["model"] == model and row["task_mode"] == "cls_peak" and np.isfinite(float(row[metric]))
            }
            for model in ("cn_hildnet", "hgb_pose_context")
        }
        if maps["cn_hildnet"] and maps["hgb_pose_context"]:
            comparisons[f"cn_minus_hgb_top20_{metric}"] = paired_fold_summary(
                maps["cn_hildnet"],
                maps["hgb_pose_context"],
                bootstrap_draws=int(config["evaluation"]["bootstrap_draws"]),
                seed=int(config["evaluation"]["statistical_seed"]),
            )

    prediction_sets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    predictions_by_model_fold: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path in sorted((run_dir / "predictions").glob("*.csv")):
        for row in _read_csvs([path]):
            predictions_by_model_fold[(row["model"], row["task_mode"], row["fold"])].append(row)
            if row["split"] == "test":
                prediction_sets[f"{row['model']}/{row['task_mode']}"].add((row["fold"], row["sample_id"]))
    cohort_match = bool(prediction_sets) and len({frozenset(values) for values in prediction_sets.values()}) == 1
    if not cohort_match:
        raise AssertionError("paper-facing models do not share one test cohort")

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_dir / "top_fraction_by_fold.csv", top_fraction_rows)
    calibration_rows = []
    for (model, task_mode, fold), rows in predictions_by_model_fold.items():
        if model not in {"cn_hildnet", "hgb_pose_context"} or task_mode != "cls_peak":
            continue
        for row in calibrate_fold_predictions(rows):
            calibration_rows.append({"model": model, "task_mode": task_mode, "fold": fold, **row})
    _write_csv(analysis_dir / "calibration.csv", calibration_rows)

    subgroup_rows = []
    for (model, task_mode, fold), rows in predictions_by_model_fold.items():
        from sklearn.metrics import f1_score

        validation = [row for row in rows if row["split"] == "val"]
        val_labels = np.asarray([float(row["y_true_cls"]) for row in validation])
        val_scores = np.asarray([float(row["y_prob"]) for row in validation])
        threshold = float(
            max(np.linspace(0.05, 0.95, 37), key=lambda value: f1_score(val_labels, val_scores >= value, zero_division=0))
        )
        test = [row for row in rows if row["split"] == "test"]
        for subgroup, subset in (
            ("rally", [row for row in test if row.get("stage") == "rally"]),
            ("non_rally", [row for row in test if row.get("stage") != "rally"]),
            ("fresh", [row for row in test if row.get("fatigue_state") == "fresh"]),
            ("fatigued", [row for row in test if row.get("fatigue_state") == "fatigued"]),
        ):
            if not subset:
                subgroup_rows.append({"model": model, "task_mode": task_mode, "fold": fold, "subgroup": subgroup, "resolution": "view", "excluded_reason": "no_samples"})
                continue
            labels = np.asarray([float(row["y_true_cls"]) for row in subset])
            scores = np.asarray([float(row["y_prob"]) for row in subset])
            view = compute_binary_metrics(labels, scores, threshold)
            unique = aggregate_unique_impact(subset, pred_keys=("y_true_cls", "y_prob"))
            for resolution, count, unique_count, metrics in (
                ("view", len(subset), len(unique["y_true_cls"]), view),
                ("unique_impact", len(subset), len(unique["y_true_cls"]), compute_binary_metrics(unique["y_true_cls"], unique["y_prob"], threshold)),
            ):
                reason = "single_class" if not np.isfinite(metrics["AUROC"]) else ""
                subgroup_rows.append(
                    {
                        "model": model,
                        "task_mode": task_mode,
                        "fold": fold,
                        "subgroup": subgroup,
                        "resolution": resolution,
                        "view_count": count,
                        "unique_impact_count": unique_count,
                        "excluded_reason": reason,
                        **metrics,
                    }
                )
    _write_csv(analysis_dir / "subgroups.csv", subgroup_rows)

    output = {
        "run_dir": str(run_dir.resolve()),
        "cohort_match": cohort_match,
        "test_cohort_counts": {key: len(value) for key, value in prediction_sets.items()},
        "summaries": summaries,
        "paired_comparisons": comparisons,
    }
    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path
