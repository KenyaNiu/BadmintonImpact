"""Shared fold training and prediction for neural and HGB models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from badminton_impact_ai.data.cohort import sample_id
from badminton_impact_ai.data.sequence_dataset import ContextEncoder, SequenceContactDataset, collate_sequence_batch
from badminton_impact_ai.metrics import aggregate_unique_impact, compute_binary_metrics, compute_regression_metrics

from .models import build_deep_model, uses_context, uses_stats
from .reproducibility import seed_everything


@dataclass
class DeepFoldData:
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
    train = SequenceContactDataset(train_rows, context, stat_map)
    val = SequenceContactDataset(val_rows, context, stat_map)
    test = SequenceContactDataset(test_rows, context, stat_map)
    if not train or not val or not test:
        raise ValueError("empty train, validation, or test dataset after eligibility filtering")
    labels = np.asarray([float(row["context_high_impact_q75"]) for row in train.rows])
    positive = max(float(labels.sum()), 1.0)
    negative = max(float(len(labels) - labels.sum()), 1.0)
    peaks = np.asarray([float(row["peak_fz_context_centered"]) for row in train.rows])
    peak_std = float(np.std(peaks))
    return DeepFoldData(
        train=train,
        val_loader=DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_sequence_batch),
        test_loader=DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_sequence_batch),
        context_encoder=context,
        peak_mean=float(np.mean(peaks)),
        peak_std=peak_std if peak_std > 1e-8 else 1.0,
        positive_weight=negative / positive,
    )


def _select_threshold(rows: list[dict[str, Any]]) -> float:
    from sklearn.metrics import f1_score

    labels = np.asarray([row["y_true_cls"] for row in rows])
    scores = np.asarray([row["y_prob"] for row in rows])
    candidates = np.linspace(0.05, 0.95, 37)
    return float(max(candidates, key=lambda threshold: f1_score(labels, scores >= threshold, zero_division=0)))


def _metric_rows(
    predictions: list[dict[str, Any]],
    threshold: float,
    fold: str,
    model: str,
    task_mode: str,
) -> list[dict[str, Any]]:
    labels = np.asarray([row["y_true_cls"] for row in predictions])
    scores = np.asarray([row["y_prob"] for row in predictions])
    output = [
        {
            "fold": fold,
            "model": model,
            "task_mode": task_mode,
            "resolution": "view",
            "task": "classification",
            **compute_binary_metrics(labels, scores, threshold),
        }
    ]
    unique = aggregate_unique_impact(predictions, pred_keys=("y_true_cls", "y_prob"))
    output.append(
        {
            "fold": fold,
            "model": model,
            "task_mode": task_mode,
            "resolution": "unique_impact",
            "task": "classification",
            **compute_binary_metrics(unique["y_true_cls"], unique["y_prob"], threshold),
        }
    )
    if task_mode == "cls_peak":
        true_peak = np.asarray([row["y_true_peak"] for row in predictions])
        pred_peak = np.asarray([row["y_pred_peak"] for row in predictions])
        output.append(
            {
                "fold": fold,
                "model": model,
                "task_mode": task_mode,
                "resolution": "view",
                "task": "regression_peak",
                **compute_regression_metrics(true_peak, pred_peak),
            }
        )
        unique_peak = aggregate_unique_impact(predictions, pred_keys=("y_true_peak", "y_pred_peak"))
        output.append(
            {
                "fold": fold,
                "model": model,
                "task_mode": task_mode,
                "resolution": "unique_impact",
                "task": "regression_peak",
                **compute_regression_metrics(unique_peak["y_true_peak"], unique_peak["y_pred_peak"]),
            }
        )
    return output


@torch.no_grad()
def _predict_deep(
    loader: DataLoader,
    model: nn.Module,
    model_name: str,
    fold: str,
    split: str,
    task_mode: str,
    peak_mean: float,
    peak_std: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        output = model(
            batch["x_seq"].to(device),
            batch["seq_mask"].to(device),
            batch["x_context"].to(device) if uses_context(model_name) else None,
            batch["x_stat"].to(device) if uses_stats(model_name) else None,
        )
        logits = output["cls_logits"].cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-logits))
        peaks = output["peak_pred"].cpu().numpy() * peak_std + peak_mean
        labels = batch["targets"]["context_high_impact_q75"].numpy()
        true_peaks = batch["targets"]["peak_fz_context_centered"].numpy()
        for index, metadata in enumerate(batch["metadata"]):
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "backend": "torch",
                    "task_mode": task_mode,
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
    settings: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(model_seed)
    task_mode = str(settings.get("task_mode", "cls_peak"))
    generator = torch.Generator().manual_seed(model_seed)
    train_loader = DataLoader(
        fold_data.train,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        collate_fn=collate_sequence_batch,
        generator=generator,
    )
    model = build_deep_model(
        model_name,
        stat_dim=fold_data.train[0]["x_stat"].shape[0],
        context_dim=fold_data.context_encoder.output_dim(),
        hidden_dim=int(settings.get("hidden_dim", 128)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(fold_data.positive_weight, device=device))
    huber = nn.SmoothL1Loss()
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    logs: list[dict[str, Any]] = []

    for epoch in range(int(settings["epochs"])):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(
                batch["x_seq"].to(device),
                batch["seq_mask"].to(device),
                batch["x_context"].to(device) if uses_context(model_name) else None,
                batch["x_stat"].to(device) if uses_stats(model_name) else None,
            )
            labels = batch["targets"]["context_high_impact_q75"].to(device)
            loss = bce(output["cls_logits"], labels)
            if task_mode == "cls_peak":
                peaks = batch["targets"]["peak_fz_context_centered"].to(device)
                normalized = (peaks - fold_data.peak_mean) / fold_data.peak_std
                loss = loss + float(settings["aux_loss_weight"]) * huber(output["peak_pred"], normalized)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_predictions = _predict_deep(
            fold_data.val_loader,
            model,
            model_name,
            fold,
            "val",
            task_mode,
            fold_data.peak_mean,
            fold_data.peak_std,
            device,
        )
        val_metrics = compute_binary_metrics(
            np.asarray([row["y_true_cls"] for row in val_predictions]),
            np.asarray([row["y_prob"] for row in val_predictions]),
            0.5,
        )
        score = float(val_metrics["AUROC"])
        logs.append(
            {
                "fold": fold,
                "model": model_name,
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_AUROC": score,
            }
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(settings["patience"]):
                break

    if best_state is None:
        raise RuntimeError(f"no finite validation checkpoint for {fold}/{model_name}")
    model.load_state_dict(best_state)
    val_predictions = _predict_deep(
        fold_data.val_loader, model, model_name, fold, "val", task_mode, fold_data.peak_mean, fold_data.peak_std, device
    )
    test_predictions = _predict_deep(
        fold_data.test_loader, model, model_name, fold, "test", task_mode, fold_data.peak_mean, fold_data.peak_std, device
    )
    threshold = _select_threshold(val_predictions)
    return {
        "predictions": val_predictions + test_predictions,
        "metrics": _metric_rows(test_predictions, threshold, fold, model_name, task_mode),
        "logs": logs,
        "checkpoint": {
            "model_state_dict": best_state,
            "model": model_name,
            "task_mode": task_mode,
            "model_seed": model_seed,
            "best_epoch": best_epoch,
            "peak_mean": fold_data.peak_mean,
            "peak_std": fold_data.peak_std,
            "context_encoder": {
                "stage_to_idx": fold_data.context_encoder.stage_to_idx,
                "fatigue_to_idx": fold_data.context_encoder.fatigue_to_idx,
                "phase_to_idx": fold_data.context_encoder.phase_to_idx,
            },
        },
    }


def _hgb_design(
    rows: list[dict[str, str]],
    model_name: str,
    context: ContextEncoder,
    stat_map: dict[str, np.ndarray],
) -> np.ndarray:
    arrays = []
    for row in rows:
        stat = stat_map.get(sample_id(row), stat_map.get(row["npz_path"]))
        if stat is None:
            raise KeyError(f"missing statistical features for {sample_id(row)}")
        ctx = context.encode(row.get("stage", "unknown"), row.get("fatigue_state", "unknown"))
        if model_name == "hgb_context":
            arrays.append(ctx)
        elif model_name == "hgb_pose":
            arrays.append(stat)
        elif model_name == "hgb_pose_context":
            arrays.append(np.concatenate([stat, ctx]))
        else:
            raise ValueError(f"unknown HGB model: {model_name}")
    return np.asarray(arrays, dtype=float)


def train_hgb_fold(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    stat_map: dict[str, np.ndarray],
    model_name: str,
    fold: str,
    model_seed: int,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    context = ContextEncoder.fit(train_rows)
    train_x = _hgb_design(train_rows, model_name, context, stat_map)
    val_x = _hgb_design(val_rows, model_name, context, stat_map)
    test_x = _hgb_design(test_rows, model_name, context, stat_map)
    train_y = np.asarray([int(float(row["context_high_impact_q75"])) for row in train_rows])
    train_peak = np.asarray([float(row["peak_fz_context_centered"]) for row in train_rows])
    classifier = HistGradientBoostingClassifier(max_depth=6, random_state=model_seed).fit(train_x, train_y)
    regressor = HistGradientBoostingRegressor(max_depth=6, random_state=model_seed).fit(train_x, train_peak)

    def predict(rows: list[dict[str, str]], x: np.ndarray, split: str) -> list[dict[str, Any]]:
        scores = classifier.predict_proba(x)[:, 1]
        peaks = regressor.predict(x)
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
                "y_true_cls": float(row["context_high_impact_q75"]),
                "cls_logit": float(np.log(np.clip(score, 1e-7, 1 - 1e-7) / (1 - np.clip(score, 1e-7, 1 - 1e-7)))),
                "y_prob": float(score),
                "y_true_peak": float(row["peak_fz_context_centered"]),
                "y_pred_peak": float(peak),
            }
            for row, score, peak in zip(rows, scores, peaks)
        ]

    val_predictions = predict(val_rows, val_x, "val")
    test_predictions = predict(test_rows, test_x, "test")
    threshold = _select_threshold(val_predictions)
    return {
        "predictions": val_predictions + test_predictions,
        "metrics": _metric_rows(test_predictions, threshold, fold, model_name, "cls_peak"),
        "logs": [],
        "checkpoint": {"classifier": classifier, "regressor": regressor, "context_encoder": context, "model_seed": model_seed},
    }
