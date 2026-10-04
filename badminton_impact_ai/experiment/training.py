"""Per-fold training and prediction for the neural models and the gradient-boosting baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from badminton_impact_ai.data.cohort import sample_id
from badminton_impact_ai.data.sequence_dataset import ContextEncoder, SequenceContactDataset, collate_sequence_batch
from badminton_impact_ai.metrics import (
    aggregate_unique_impact,
    compute_binary_metrics,
    compute_regression_metrics,
    select_f1_threshold,
)
from badminton_impact_ai.models import build_deep_model, uses_context, uses_stats

from .reproducibility import seed_everything

LABEL, PEAK = "context_high_impact_q75", "peak_fz_context_centered"


@dataclass(frozen=True)
class TrainSettings:
    """Optimisation settings of one neural model (``task_mode``: ``cls_peak`` or ``cls_only``)."""

    batch_size: int
    learning_rate: float
    epochs: int
    patience: int
    hidden_dim: int
    aux_loss_weight: float
    task_mode: str

    @classmethod
    def from_config(cls, training: dict[str, Any], task_mode: str) -> TrainSettings:
        return cls(
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
            hidden_dim=int(training.get("hidden_dim", 128)),
            aux_loss_weight=float(training.get("aux_loss_weight", 0.0)),
            task_mode=task_mode,
        )


@dataclass
class DeepFoldData:
    """Datasets and label statistics of one outer fold, shared by all neural models."""

    train: SequenceContactDataset
    val_loader: DataLoader
    test_loader: DataLoader
    context_encoder: ContextEncoder
    peak_mean: float
    peak_std: float
    positive_weight: float


def prepare_deep_fold(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    stat_map: dict[str, np.ndarray],
    batch_size: int,
) -> DeepFoldData:
    context = ContextEncoder.fit(train_rows)
    train, val, test = (SequenceContactDataset(rows, context, stat_map) for rows in (train_rows, val_rows, test_rows))
    if not train or not val or not test:
        raise ValueError("empty train, validation, or test dataset after eligibility filtering")
    labels = np.asarray([float(row[LABEL]) for row in train.rows])
    positive = max(float(labels.sum()), 1.0)
    negative = max(float(len(labels) - labels.sum()), 1.0)
    peaks = np.asarray([float(row[PEAK]) for row in train.rows])
    peak_std = float(np.std(peaks))

    def loader(dataset: SequenceContactDataset) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequence_batch)

    return DeepFoldData(
        train=train,
        val_loader=loader(val),
        test_loader=loader(test),
        context_encoder=context,
        peak_mean=float(np.mean(peaks)),
        peak_std=peak_std if peak_std > 1e-8 else 1.0,
        positive_weight=negative / positive,
    )


def _select_threshold(predictions: list[dict[str, Any]]) -> float:
    labels = np.asarray([row["y_true_cls"] for row in predictions])
    scores = np.asarray([row["y_prob"] for row in predictions])
    return select_f1_threshold(labels, scores)


def _metric_rows(
    predictions: list[dict[str, Any]], threshold: float, fold: str, model: str, task_mode: str
) -> list[dict[str, Any]]:
    """Test-set metrics at view and unique-impact resolution (plus peak regression for ``cls_peak``)."""
    base = {"fold": fold, "model": model, "task_mode": task_mode}

    def row(resolution: str, task: str, metrics: dict[str, float]) -> dict[str, Any]:
        return {**base, "resolution": resolution, "task": task, **metrics}

    labels = np.asarray([r["y_true_cls"] for r in predictions])
    scores = np.asarray([r["y_prob"] for r in predictions])
    unique = aggregate_unique_impact(predictions, pred_keys=("y_true_cls", "y_prob"))
    rows = [
        row("view", "classification", compute_binary_metrics(labels, scores, threshold)),
        row(
            "unique_impact",
            "classification",
            compute_binary_metrics(unique["y_true_cls"], unique["y_prob"], threshold),
        ),
    ]
    if task_mode == "cls_peak":
        true_peak = np.asarray([r["y_true_peak"] for r in predictions])
        pred_peak = np.asarray([r["y_pred_peak"] for r in predictions])
        unique_peak = aggregate_unique_impact(predictions, pred_keys=("y_true_peak", "y_pred_peak"))
        rows += [
            row("view", "regression_peak", compute_regression_metrics(true_peak, pred_peak)),
            row(
                "unique_impact",
                "regression_peak",
                compute_regression_metrics(unique_peak["y_true_peak"], unique_peak["y_pred_peak"]),
            ),
        ]
    return rows


@dataclass(frozen=True)
class _Run:
    """Identity of one (fold, model) training run, attached to every prediction row."""

    fold: str
    model: str
    task_mode: str
    peak_mean: float
    peak_std: float
    device: torch.device


def _forward(model: nn.Module, batch: dict[str, Any], model_name: str, device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        batch["x_seq"].to(device),
        batch["seq_mask"].to(device),
        batch["x_context"].to(device) if uses_context(model_name) else None,
        batch["x_stat"].to(device) if uses_stats(model_name) else None,
    )


@torch.no_grad()
def _predict_deep(loader: DataLoader, model: nn.Module, run: _Run, split: str) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        output = _forward(model, batch, run.model, run.device)
        logits = output["cls_logits"].cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-logits))
        peaks = output["peak_pred"].cpu().numpy() * run.peak_std + run.peak_mean
        labels = batch["targets"][LABEL].numpy()
        true_peaks = batch["targets"][PEAK].numpy()
        for index, metadata in enumerate(batch["metadata"]):
            rows.append(
                {
                    "fold": run.fold,
                    "model": run.model,
                    "backend": "torch",
                    "task_mode": run.task_mode,
                    "split": split,
                    **metadata,
                    "y_true_cls": float(labels[index]),
                    "cls_logit": float(logits[index]),
                    "y_prob": float(scores[index]),
                    "y_true_peak": float(true_peaks[index]),
                    "y_pred_peak": float(peaks[index]),
                }
            )
    return rows


def train_deep_fold(
    fold_data: DeepFoldData,
    model_name: str,
    fold: str,
    model_seed: int,
    settings: TrainSettings,
    device: torch.device,
) -> dict[str, Any]:
    """Train one neural model with early stopping on validation AUROC and evaluate it on the held-out subject."""
    seed_everything(model_seed)
    run = _Run(fold, model_name, settings.task_mode, fold_data.peak_mean, fold_data.peak_std, device)
    train_loader = DataLoader(
        fold_data.train,
        batch_size=settings.batch_size,
        shuffle=True,
        collate_fn=collate_sequence_batch,
        generator=torch.Generator().manual_seed(model_seed),
    )
    model = build_deep_model(
        model_name,
        stat_dim=fold_data.train[0]["x_stat"].shape[0],
        context_dim=fold_data.context_encoder.output_dim(),
        hidden_dim=settings.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(fold_data.positive_weight, device=device))
    huber = nn.SmoothL1Loss()

    best_score, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    logs: list[dict[str, Any]] = []
    for epoch in range(settings.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            output = _forward(model, batch, model_name, device)
            loss = bce(output["cls_logits"], batch["targets"][LABEL].to(device))
            if settings.task_mode == "cls_peak":
                peaks = batch["targets"][PEAK].to(device)
                normalized = (peaks - fold_data.peak_mean) / fold_data.peak_std
                loss = loss + settings.aux_loss_weight * huber(output["peak_pred"], normalized)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_rows = _predict_deep(fold_data.val_loader, model, run, "val")
        val_auroc = float(
            compute_binary_metrics(
                np.asarray([r["y_true_cls"] for r in val_rows]), np.asarray([r["y_prob"] for r in val_rows]), 0.5
            )["AUROC"]
        )
        logs.append(
            {
                "fold": fold,
                "model": model_name,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_AUROC": val_auroc,
            }
        )
        if np.isfinite(val_auroc) and val_auroc > best_score:
            best_score, best_epoch, stale = val_auroc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break

    if best_state is None:
        raise RuntimeError(f"no finite validation checkpoint for {fold}/{model_name}")
    model.load_state_dict(best_state)
    val_predictions = _predict_deep(fold_data.val_loader, model, run, "val")
    test_predictions = _predict_deep(fold_data.test_loader, model, run, "test")
    threshold = _select_threshold(val_predictions)
    encoder = fold_data.context_encoder
    return {
        "predictions": val_predictions + test_predictions,
        "metrics": _metric_rows(test_predictions, threshold, fold, model_name, settings.task_mode),
        "logs": logs,
        "checkpoint": {
            "model_state_dict": best_state,
            "model": model_name,
            "task_mode": settings.task_mode,
            "model_seed": model_seed,
            "best_epoch": best_epoch,
            "peak_mean": fold_data.peak_mean,
            "peak_std": fold_data.peak_std,
            "context_encoder": {
                "stage_to_idx": encoder.stage_to_idx,
                "fatigue_to_idx": encoder.fatigue_to_idx,
                "phase_to_idx": encoder.phase_to_idx,
            },
        },
    }


def _hgb_design(
    rows: list[dict[str, str]], model_name: str, context: ContextEncoder, stat_map: dict[str, np.ndarray]
) -> np.ndarray:
    """Feature matrix of a gradient-boosting baseline: context only, pose statistics only, or both."""
    if model_name not in {"hgb_context", "hgb_pose", "hgb_pose_context"}:
        raise ValueError(f"unknown HGB model: {model_name}")
    features = []
    for row in rows:
        stat = stat_map.get(sample_id(row), stat_map.get(row["npz_path"]))
        if stat is None:
            raise KeyError(f"missing statistical features for {sample_id(row)}")
        ctx = context.encode(row.get("stage", "unknown"), row.get("fatigue_state", "unknown"))
        if model_name == "hgb_context":
            features.append(ctx)
        elif model_name == "hgb_pose":
            features.append(stat)
        else:
            features.append(np.concatenate([stat, ctx]))
    return np.asarray(features, dtype=float)


def _logit(probability: float) -> float:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return float(np.log(clipped / (1 - clipped)))


def train_hgb_fold(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    stat_map: dict[str, np.ndarray],
    model_name: str,
    fold: str,
    model_seed: int,
) -> dict[str, Any]:
    """Fit a HistGradientBoosting classifier (ranking score) and regressor (peak) on the training subjects."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    context = ContextEncoder.fit(train_rows)
    design = {
        name: _hgb_design(rows, model_name, context, stat_map)
        for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows))
    }
    train_y = np.asarray([int(float(row[LABEL])) for row in train_rows])
    train_peak = np.asarray([float(row[PEAK]) for row in train_rows])
    classifier = HistGradientBoostingClassifier(max_depth=6, random_state=model_seed).fit(design["train"], train_y)
    regressor = HistGradientBoostingRegressor(max_depth=6, random_state=model_seed).fit(design["train"], train_peak)

    def predict(rows: list[dict[str, str]], split: str) -> list[dict[str, Any]]:
        scores = classifier.predict_proba(design[split])[:, 1]
        peaks = regressor.predict(design[split])
        return [
            {
                "fold": fold,
                "model": model_name,
                "backend": "sklearn",
                "task_mode": "cls_peak",
                "split": split,
                "sample_id": sample_id(row),
                "npz_path": row["npz_path"],
                "subject_id": row["subject_id"],
                "trial_id": row.get("trial_id", "unknown"),
                "camera_id": row.get("camera_id", "unknown"),
                "unique_impact_key_candidate": row["unique_impact_key_candidate"],
                "stage": row.get("stage", "unknown"),
                "fatigue_state": row.get("fatigue_state", "unknown"),
                "y_true_cls": float(row[LABEL]),
                "cls_logit": _logit(score),
                "y_prob": float(score),
                "y_true_peak": float(row[PEAK]),
                "y_pred_peak": float(peak),
            }
            for row, score, peak in zip(rows, scores, peaks, strict=True)
        ]

    val_predictions, test_predictions = predict(val_rows, "val"), predict(test_rows, "test")
    threshold = _select_threshold(val_predictions)
    return {
        "predictions": val_predictions + test_predictions,
        "metrics": _metric_rows(test_predictions, threshold, fold, model_name, "cls_peak"),
        "logs": [],
        "checkpoint": {
            "classifier": classifier,
            "regressor": regressor,
            "context_encoder": context,
            "model_seed": model_seed,
        },
    }
