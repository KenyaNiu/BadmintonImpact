#!/usr/bin/env python3
"""Train full-LOSO deep models for M5/M6 with resume/skip support.

When invoked with ``--early-stop-metric brier``, checkpoint selection and early stopping use
validation Brier score (minimize) instead of validation AUROC (maximize). All other settings match the
default training recipe. This variant is used for the Brier-early-stop ablation reported in Table X.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.sequence_dataset import (
    ContextEncoder,
    SequenceContactDataset,
    collate_sequence_batch,
    load_context_rows,
    load_stat_features,
)
from metrics import aggregate_unique_impact, compute_binary_metrics, compute_regression_metrics
from models import DeepTCNBaseline, ICSIHybridContext, ICSIHybridContextStageHeads, ICSISeq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--context-labels", type=Path, required=True)
    p.add_argument("--all-labels", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--feature-meta", type=Path, required=True)
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["tcn", "icsi_seq", "icsi_hybrid_context", "icsi_hybrid_context_v2", "icsi_hybrid_stage_heads"],
    )
    p.add_argument(
        "--rally-bce-weight",
        type=float,
        default=1.0,
        help="Per-sample multiplier on rally rows for ICSI BCE+huber (1.0 = disabled). Train/val only; no test leakage.",
    )
    p.add_argument("--task-mode", type=str, default="cls_peak", choices=["cls_peak", "cls_only"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.5,
        dest="aux_loss_weight",
        help="Weight λ on the auxiliary SmoothL1 peak term (paper L = BCE + λ·SmoothL1).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--early-stop-metric",
        choices=["auroc", "brier"],
        default="auroc",
        help="Validation metric for early stopping and best checkpoint (AUROC: higher is better; Brier: lower is better).",
    )
    p.add_argument("--folds", nargs="*")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resume-skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_split(rows: list[dict[str, str]], val_ratio: float = 0.15, seed: int = 42) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    g = defaultdict(list)
    for r in rows:
        g[r["unique_impact_key_candidate"]].append(r)
    keys = list(g.keys())
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio))
    vset = set(keys[:n_val])
    tr, va = [], []
    for k, items in g.items():
        if k in vset:
            va.extend(items)
        else:
            tr.extend(items)
    return tr, va


def _build_model(name: str, stat_dim: int, context_dim: int) -> nn.Module:
    if name == "tcn":
        return DeepTCNBaseline(seq_dim=34, hidden_dim=128)
    if name == "icsi_seq":
        return ICSISeq(seq_dim=34, hidden_dim=128)
    if name in ("icsi_hybrid_context", "icsi_hybrid_context_v2"):
        return ICSIHybridContext(seq_dim=34, stat_dim=stat_dim, context_dim=context_dim, hidden_dim=128)
    if name == "icsi_hybrid_stage_heads":
        return ICSIHybridContextStageHeads(seq_dim=34, stat_dim=stat_dim, context_dim=context_dim, hidden_dim=128)
    raise ValueError(name)


def _stage_rally_mask(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [1.0 if m.get("stage") == "rally" else 0.0 for m in batch["metadata"]],
        device=device,
        dtype=torch.float32,
    )


def _cls_logits_for_pred(model_name: str, out: dict[str, torch.Tensor], batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    if model_name == "icsi_hybrid_stage_heads":
        sr = _stage_rally_mask(batch, device)
        return torch.where(sr > 0.5, out["cls_logits_rally"], out["cls_logits_non"])
    return out["cls_logits"]


def _icsi_weighted_cls_peak_loss(
    logits: torch.Tensor,
    tgt_cls: torch.Tensor,
    peak_pred: torch.Tensor,
    peak_norm: torch.Tensor,
    pos_weight_scalar: float,
    rally_w: float,
    sm: torch.Tensor,
    aux_weight: float,
) -> torch.Tensor:
    """BCE with class imbalance + optional rally row multiplier; SmoothL1 on peak with same multiplier."""
    pw = torch.where(
        tgt_cls > 0.5,
        torch.full_like(tgt_cls, float(pos_weight_scalar)),
        torch.ones_like(tgt_cls),
    )
    el = F.binary_cross_entropy_with_logits(logits, tgt_cls, reduction="none")
    w_stage = 1.0 + (float(rally_w) - 1.0) * sm
    w = pw * w_stage
    hub_e = F.smooth_l1_loss(peak_pred, peak_norm, reduction="none")
    return (el * w).mean() + float(aux_weight) * (hub_e * w).mean()


def _stage_heads_cls_peak_loss(
    out: dict[str, torch.Tensor],
    tgt_cls: torch.Tensor,
    peak_norm: torch.Tensor,
    sm: torch.Tensor,
    pos_weight_scalar: float,
    aux_weight: float,
) -> torch.Tensor:
    lr, lnr = out["cls_logits_rally"], out["cls_logits_non"]
    pw = torch.where(
        tgt_cls > 0.5,
        torch.full_like(tgt_cls, float(pos_weight_scalar)),
        torch.ones_like(tgt_cls),
    )
    el_r = F.binary_cross_entropy_with_logits(lr, tgt_cls, reduction="none")
    el_nr = F.binary_cross_entropy_with_logits(lnr, tgt_cls, reduction="none")
    loss_cls = (el_r * pw * sm + el_nr * pw * (1.0 - sm)).mean()
    hub_e = F.smooth_l1_loss(out["peak_pred"], peak_norm, reduction="none")
    return loss_cls + float(aux_weight) * hub_e.mean()


def _stage_heads_cls_only_loss(out: dict[str, torch.Tensor], tgt_cls: torch.Tensor, sm: torch.Tensor, pos_weight_scalar: float) -> torch.Tensor:
    lr, lnr = out["cls_logits_rally"], out["cls_logits_non"]
    pw = torch.where(
        tgt_cls > 0.5,
        torch.full_like(tgt_cls, float(pos_weight_scalar)),
        torch.ones_like(tgt_cls),
    )
    el_r = F.binary_cross_entropy_with_logits(lr, tgt_cls, reduction="none")
    el_nr = F.binary_cross_entropy_with_logits(lnr, tgt_cls, reduction="none")
    return (el_r * pw * sm + el_nr * pw * (1.0 - sm)).mean()


def _icsi_weighted_cls_only_loss(
    logits: torch.Tensor,
    tgt_cls: torch.Tensor,
    pos_weight_scalar: float,
    rally_w: float,
    sm: torch.Tensor,
) -> torch.Tensor:
    pw = torch.where(
        tgt_cls > 0.5,
        torch.full_like(tgt_cls, float(pos_weight_scalar)),
        torch.ones_like(tgt_cls),
    )
    el = F.binary_cross_entropy_with_logits(logits, tgt_cls, reduction="none")
    w_stage = 1.0 + (float(rally_w) - 1.0) * sm
    return (el * pw * w_stage).mean()


@torch.no_grad()
def _predict_loader(
    loader: DataLoader,
    model: nn.Module,
    peak_stats: tuple[float, float],
    device: torch.device,
    split: str,
    fold: str,
    model_name: str,
    task_mode: str,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    mean, std = peak_stats
    for batch in loader:
        out = model(
            x_seq=batch["x_seq"].to(device),
            seq_mask=batch["seq_mask"].to(device),
            x_context=batch["x_context"].to(device),
            x_stat=batch["x_stat"].to(device),
        )
        cls_logits = _cls_logits_for_pred(model_name, out, batch, device)
        cls_prob = torch.sigmoid(cls_logits).cpu().numpy()
        peak_pred = out["peak_pred"].cpu().numpy() * std + mean
        tgt = {k: v.cpu().numpy() for k, v in batch["targets"].items()}
        for i, m in enumerate(batch["metadata"]):
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "task_mode": task_mode,
                    "split": split,
                    "npz_path": m["npz_path"],
                    "unique_impact_key_candidate": m["unique_impact_key_candidate"],
                    "subject_id": m["subject_id"],
                    "stage": m["stage"],
                    "fatigue_state": m["fatigue_state"],
                    "y_true_cls": float(tgt["context_high_impact_q75"][i]),
                    "y_prob": float(cls_prob[i]),
                    "y_true_peak": float(tgt["peak_fz_context_centered"][i]),
                    "y_pred_peak": float(peak_pred[i]),
                }
            )
    return rows


def _select_threshold_from_val(val_rows: list[dict[str, Any]]) -> float:
    from sklearn.metrics import f1_score

    y = np.asarray([r["y_true_cls"] for r in val_rows], dtype=float)
    p = np.asarray([r["y_prob"] for r in val_rows], dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (p >= t).astype(int)
        f1 = float(f1_score(y, pred, zero_division=0))
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    out_dir = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stat_map, stat_dim = load_stat_features(args.features, args.feature_meta)
    with args.context_labels.open("r", encoding="utf-8", newline="") as f:
        all_rows = [r for r in csv.DictReader(f) if r.get("npz_path") in stat_map and r.get("subject_id", "unknown") != "unknown"]
    all_folds = sorted({r["test_subject"] for r in all_rows})
    folds = args.folds if args.folds else all_folds

    if args.task_mode == "cls_only":
        view_path = out_dir / "ablation_metrics_view.csv"
        uniq_path = out_dir / "ablation_metrics_unique.csv"
        pred_path = out_dir / "ablation_predictions.csv"
        log_path = out_dir / "ablation_training_log.csv"
        summary_path = out_dir / "ablation_summary.json"
    else:
        view_path = out_dir / "deep_metrics_view.csv"
        uniq_path = out_dir / "deep_metrics_unique.csv"
        pred_path = out_dir / "deep_predictions.csv"
        log_path = out_dir / "deep_training_log.csv"
        summary_path = out_dir / "deep_summary.json"
    view_rows = [] if args.overwrite else _load_existing(view_path)
    uniq_rows = [] if args.overwrite else _load_existing(uniq_path)
    pred_rows = [] if args.overwrite else _load_existing(pred_path)
    log_rows = [] if args.overwrite else _load_existing(log_path)

    for fold in folds:
        frows = [r for r in all_rows if r["test_subject"] == fold]
        train_full = [r for r in frows if r["role"] == "train"]
        test_rows = [r for r in frows if r["role"] == "test"]
        if not train_full or not test_rows:
            continue
        train_rows, val_rows = _group_split(train_full, val_ratio=0.15, seed=args.seed)
        ctx_encoder = ContextEncoder.fit(train_rows)
        tr_ds = SequenceContactDataset(rows=train_rows, context_encoder=ctx_encoder, stat_map=stat_map)
        va_ds = SequenceContactDataset(rows=val_rows, context_encoder=ctx_encoder, stat_map=stat_map)
        te_ds = SequenceContactDataset(rows=test_rows, context_encoder=ctx_encoder, stat_map=stat_map)
        if len(tr_ds) == 0 or len(va_ds) == 0 or len(te_ds) == 0:
            continue
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequence_batch)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequence_batch)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequence_batch)

        cls_vals = np.asarray([float(r["context_high_impact_q75"]) for r in tr_ds.rows], dtype=float)
        pos = max(float(np.sum(cls_vals)), 1.0)
        neg = max(float(len(cls_vals) - np.sum(cls_vals)), 1.0)
        pos_weight = neg / pos
        peak_train = np.asarray([float(r["peak_fz_context_centered"]) for r in tr_ds.rows], dtype=float)
        peak_mean = float(np.mean(peak_train))
        peak_std = float(np.std(peak_train) if np.std(peak_train) > 1e-8 else 1.0)

        for model_name in args.models:
            existing = any(
                r.get("fold") == fold and r.get("model") == model_name and r.get("task_mode") == args.task_mode and r.get("task") == "classification"
                for r in view_rows
            )
            if args.resume_skip_existing and existing and not args.overwrite:
                continue

            model = _build_model(model_name, stat_dim=stat_dim, context_dim=ctx_encoder.output_dim()).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
            bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
            huber = nn.SmoothL1Loss()
            best_metric = float("inf") if args.early_stop_metric == "brier" else -1e9
            best_state = None
            patience_count = 0
            best_epoch = 0

            for epoch in range(args.epochs):
                model.train()
                tr_losses = []
                for batch in tr_loader:
                    optim.zero_grad()
                    ctx = batch["x_context"].to(device)
                    stat = batch["x_stat"].to(device)
                    if model_name == "tcn":
                        ctx = None
                        stat = None
                    elif model_name == "icsi_seq":
                        ctx = None
                        stat = None
                    out = model(x_seq=batch["x_seq"].to(device), seq_mask=batch["seq_mask"].to(device), x_context=ctx, x_stat=stat)
                    tgt_cls = batch["targets"]["context_high_impact_q75"].to(device)
                    if args.task_mode == "cls_peak":
                        tgt_peak = batch["targets"]["peak_fz_context_centered"].to(device)
                        peak_norm = (tgt_peak - peak_mean) / peak_std
                        if model_name == "icsi_hybrid_stage_heads":
                            sm = _stage_rally_mask(batch, device)
                            loss = _stage_heads_cls_peak_loss(
                                out, tgt_cls, peak_norm, sm, float(pos_weight), float(args.aux_loss_weight)
                            )
                        elif model_name in ("icsi_hybrid_context", "icsi_hybrid_context_v2") and args.rally_bce_weight > 1.0:
                            sm = _stage_rally_mask(batch, device)
                            loss = _icsi_weighted_cls_peak_loss(
                                out["cls_logits"],
                                tgt_cls,
                                out["peak_pred"],
                                peak_norm,
                                float(pos_weight),
                                float(args.rally_bce_weight),
                                sm,
                                float(args.aux_loss_weight),
                            )
                        else:
                            loss = bce(out["cls_logits"], tgt_cls) + float(args.aux_loss_weight) * huber(
                                out["peak_pred"], peak_norm
                            )
                    else:
                        if model_name == "icsi_hybrid_stage_heads":
                            sm = _stage_rally_mask(batch, device)
                            loss = _stage_heads_cls_only_loss(out, tgt_cls, sm, float(pos_weight))
                        elif model_name in ("icsi_hybrid_context", "icsi_hybrid_context_v2") and args.rally_bce_weight > 1.0:
                            sm = _stage_rally_mask(batch, device)
                            loss = _icsi_weighted_cls_only_loss(out["cls_logits"], tgt_cls, float(pos_weight), float(args.rally_bce_weight), sm)
                        else:
                            loss = bce(out["cls_logits"], tgt_cls)
                    loss.backward()
                    optim.step()
                    tr_losses.append(float(loss.item()))

                # val
                model.eval()
                val_losses = []
                vrows = []
                for batch in va_loader:
                    ctx = batch["x_context"].to(device)
                    stat = batch["x_stat"].to(device)
                    if model_name == "tcn":
                        ctx = None
                        stat = None
                    elif model_name == "icsi_seq":
                        ctx = None
                        stat = None
                    out = model(x_seq=batch["x_seq"].to(device), seq_mask=batch["seq_mask"].to(device), x_context=ctx, x_stat=stat)
                    tgt_cls = batch["targets"]["context_high_impact_q75"].to(device)
                    if args.task_mode == "cls_peak":
                        tgt_peak = batch["targets"]["peak_fz_context_centered"].to(device)
                        peak_norm = (tgt_peak - peak_mean) / peak_std
                        if model_name == "icsi_hybrid_stage_heads":
                            sm = _stage_rally_mask(batch, device)
                            loss = _stage_heads_cls_peak_loss(
                                out, tgt_cls, peak_norm, sm, float(pos_weight), float(args.aux_loss_weight)
                            )
                        elif model_name in ("icsi_hybrid_context", "icsi_hybrid_context_v2") and args.rally_bce_weight > 1.0:
                            sm = _stage_rally_mask(batch, device)
                            loss = _icsi_weighted_cls_peak_loss(
                                out["cls_logits"],
                                tgt_cls,
                                out["peak_pred"],
                                peak_norm,
                                float(pos_weight),
                                float(args.rally_bce_weight),
                                sm,
                                float(args.aux_loss_weight),
                            )
                        else:
                            loss = bce(out["cls_logits"], tgt_cls) + float(args.aux_loss_weight) * huber(
                                out["peak_pred"], peak_norm
                            )
                    else:
                        if model_name == "icsi_hybrid_stage_heads":
                            sm = _stage_rally_mask(batch, device)
                            loss = _stage_heads_cls_only_loss(out, tgt_cls, sm, float(pos_weight))
                        elif model_name in ("icsi_hybrid_context", "icsi_hybrid_context_v2") and args.rally_bce_weight > 1.0:
                            sm = _stage_rally_mask(batch, device)
                            loss = _icsi_weighted_cls_only_loss(out["cls_logits"], tgt_cls, float(pos_weight), float(args.rally_bce_weight), sm)
                        else:
                            loss = bce(out["cls_logits"], tgt_cls)
                    val_losses.append(float(loss.item()))
                    cls_logits = _cls_logits_for_pred(model_name, out, batch, device)
                    p = torch.sigmoid(cls_logits).detach().cpu().numpy()
                    y = batch["targets"]["context_high_impact_q75"].cpu().numpy()
                    for yi, pi in zip(y, p):
                        vrows.append({"y_true": float(yi), "y_prob": float(pi)})
                yv = np.asarray([r["y_true"] for r in vrows], dtype=float)
                pv = np.asarray([r["y_prob"] for r in vrows], dtype=float)
                val_metrics = compute_binary_metrics(yv, pv, threshold=0.5)
                val_auc = val_metrics["AUROC"]
                val_brier = val_metrics["Brier"]
                if args.early_stop_metric == "brier":
                    monitor = float(val_brier) if val_brier == val_brier else float(np.mean(val_losses))
                    improved = monitor < best_metric
                else:
                    monitor = float(val_auc) if val_auc == val_auc else -float(np.mean(val_losses))
                    improved = monitor > best_metric
                log_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "task_mode": args.task_mode,
                        "epoch": epoch,
                        "train_loss": float(np.mean(tr_losses)) if tr_losses else float("nan"),
                        "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
                        "val_AUROC": val_auc,
                    }
                )
                if improved:
                    best_metric = monitor
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= args.patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            ckpt = ckpt_dir / f"{fold}__{model_name}__{args.task_mode}.pt"
            torch.save({"model_state_dict": model.state_dict(), "peak_mean": peak_mean, "peak_std": peak_std, "best_epoch": best_epoch}, ckpt)

            val_pred_rows = _predict_loader(
                va_loader, model=model, peak_stats=(peak_mean, peak_std), device=device, split="val", fold=fold, model_name=model_name, task_mode=args.task_mode
            )
            threshold = _select_threshold_from_val(val_pred_rows)
            test_pred_rows = _predict_loader(
                te_loader, model=model, peak_stats=(peak_mean, peak_std), device=device, split="test", fold=fold, model_name=model_name, task_mode=args.task_mode
            )
            pred_rows.extend(val_pred_rows)
            pred_rows.extend(test_pred_rows)

            # view metrics
            yt = np.asarray([r["y_true_cls"] for r in test_pred_rows], dtype=float)
            yp = np.asarray([r["y_prob"] for r in test_pred_rows], dtype=float)
            cm = compute_binary_metrics(yt, yp, threshold=threshold)
            view_rows.append({"fold": fold, "model": model_name, "task_mode": args.task_mode, "task": "classification", **cm})
            if args.task_mode == "cls_peak":
                ypr = np.asarray([r["y_pred_peak"] for r in test_pred_rows], dtype=float)
                yr = np.asarray([r["y_true_peak"] for r in test_pred_rows], dtype=float)
                rm = compute_regression_metrics(yr, ypr)
                view_rows.append({"fold": fold, "model": model_name, "task_mode": args.task_mode, "task": "regression:peak_fz_context_centered", **rm})

            # unique impact
            cls_rows = [{"unique_impact_key_candidate": r["unique_impact_key_candidate"], "y_true": r["y_true_cls"], "y_prob": r["y_prob"]} for r in test_pred_rows]
            agg = aggregate_unique_impact(cls_rows, pred_keys=("y_true", "y_prob"))
            ucm = compute_binary_metrics(agg["y_true"], agg["y_prob"], threshold=threshold)
            uniq_rows.append({"fold": fold, "model": model_name, "task_mode": args.task_mode, "task": "classification", **ucm})
            if args.task_mode == "cls_peak":
                reg_rows = [{"unique_impact_key_candidate": r["unique_impact_key_candidate"], "y_true": r["y_true_peak"], "y_pred": r["y_pred_peak"]} for r in test_pred_rows]
                agg_r = aggregate_unique_impact(reg_rows, pred_keys=("y_true", "y_pred"))
                urm = compute_regression_metrics(agg_r["y_true"], agg_r["y_pred"])
                uniq_rows.append({"fold": fold, "model": model_name, "task_mode": args.task_mode, "task": "regression:peak_fz_context_centered", **urm})

            # drop previous rows for same fold/model when overwrite not global
            if not args.overwrite:
                def _dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                    latest = {}
                    for rr in rows:
                        key = (rr.get("fold"), rr.get("model"), rr.get("task_mode"), rr.get("task"))
                        latest[key] = rr
                    return list(latest.values())
                view_rows = _dedup(view_rows)
                uniq_rows = _dedup(uniq_rows)

        _write_csv(view_path, view_rows)
        _write_csv(uniq_path, uniq_rows)
        _write_csv(pred_path, pred_rows)
        _write_csv(log_path, log_rows)

    # summary
    summary = {
        "folds_ran": sorted({r["fold"] for r in view_rows}),
        "models": sorted({r["model"] for r in view_rows}),
        "task_mode": args.task_mode,
        "aux_loss_weight": float(args.aux_loss_weight) if args.task_mode == "cls_peak" else None,
    }
    def _best(rows: list[dict[str, Any]], task: str, metric: str) -> dict[str, Any] | None:
        c = [r for r in rows if r.get("task") == task and str(r.get(metric, "nan")) not in {"nan", ""}]
        if not c:
            return None
        return max(c, key=lambda r: float(r[metric]))
    summary["best_cls_view"] = _best(view_rows, "classification", "AUROC")
    summary["best_cls_unique"] = _best(uniq_rows, "classification", "AUROC")
    summary["best_peak_view"] = _best(view_rows, "regression:peak_fz_context_centered", "R2")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.task_mode == "cls_only":
        rep = [
            "# M6 Ablation Report",
            "",
            "- task_mode: cls_only",
            f"- models: {', '.join(sorted({r['model'] for r in view_rows if r.get('task') == 'classification'}))}",
            f"- folds: {', '.join(sorted({r['fold'] for r in view_rows if r.get('task') == 'classification'}))}",
        ]
        (out_dir / "ablation_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print(f"Wrote: {view_path}")
    print(f"Wrote: {uniq_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {log_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
