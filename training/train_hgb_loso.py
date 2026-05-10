#!/usr/bin/env python3
"""Train full-LOSO HGB baselines for M5 main table."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metrics import aggregate_unique_impact, compute_binary_metrics, compute_regression_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--feature-meta", type=Path, required=True)
    p.add_argument("--context-labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--input-sets",
        type=str,
        default="context_only,pose_only,pose_plus_context",
        help="Comma-separated subset of input sets to train (default: all three).",
    )
    p.add_argument(
        "--balanced-pose-plus-context",
        action="store_true",
        help="Per fold, use sklearn balanced sample weights on training labels for pose+context only; model tag hgb_cls_balanced.",
    )
    return p.parse_args()


def _finite(v: Any) -> bool:
    try:
        return np.isfinite(float(v))
    except Exception:
        return False


def _load_features(features_npz: Path, feature_meta_csv: Path) -> tuple[np.ndarray, dict[str, int]]:
    with np.load(features_npz, allow_pickle=True) as d:
        X = np.asarray(d["X"], dtype=float)
        pths = [str(x) for x in d["npz_path"]]
    valid_paths = set()
    with feature_meta_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("feature_valid", "")).lower() in {"true", "1", "yes", "y"}:
                valid_paths.add(r["npz_path"])
    idx = {p: i for i, p in enumerate(pths) if p in valid_paths}
    return X, idx


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


def _onehot(train: list[dict[str, str]], test: list[dict[str, str]], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vocab = {}
    for k in keys:
        for v in sorted({r.get(k, "unknown") for r in train}):
            vocab[(k, v)] = len(vocab)

    def make(rows: list[dict[str, str]]) -> np.ndarray:
        X = np.zeros((len(rows), len(vocab)), dtype=float)
        for i, r in enumerate(rows):
            for k in keys:
                idx = vocab.get((k, r.get(k, "unknown")))
                if idx is not None:
                    X[i, idx] = 1.0
        return X

    return make(train), make(test)


def _design(input_set: str, train: list[dict[str, str]], test: list[dict[str, str]], Xflat: np.ndarray, idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    if input_set == "context_only":
        return _onehot(train, test, ["stage", "fatigue_state"])
    xtr = np.asarray([Xflat[idx[r["npz_path"]]] for r in train], dtype=float)
    xte = np.asarray([Xflat[idx[r["npz_path"]]] for r in test], dtype=float)
    if input_set == "pose_only":
        return xtr, xte
    if input_set == "pose_plus_context":
        mtr, mte = _onehot(train, test, ["stage", "fatigue_state"])
        return np.concatenate([xtr, mtr], axis=1), np.concatenate([xte, mte], axis=1)
    raise ValueError(input_set)


def _fit_cls_model(Xtr: np.ndarray, ytr: np.ndarray, sample_weight: np.ndarray | None = None):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        m = HistGradientBoostingClassifier(max_depth=6, random_state=42)
        model_name = "hgb_cls"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        m = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        model_name = "rf_cls_fallback"
    if sample_weight is not None:
        m.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        m.fit(Xtr, ytr)
    return m, model_name


def _fit_reg_model(Xtr: np.ndarray, ytr: np.ndarray):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        m = HistGradientBoostingRegressor(max_depth=6, random_state=42)
        model_name = "hgb_reg"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor

        m = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        model_name = "rf_reg_fallback"
    m.fit(Xtr, ytr)
    return m, model_name


def _select_threshold_from_val(y_val: np.ndarray, p_val: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (p_val >= t).astype(int)
        f1 = float(f1_score(y_val, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    Xflat, idx = _load_features(args.features, args.feature_meta)
    input_sets = [s.strip() for s in args.input_sets.split(",") if s.strip()]
    allowed = {"context_only", "pose_only", "pose_plus_context"}
    bad = [s for s in input_sets if s not in allowed]
    if bad:
        print(f"Unknown --input-sets entries: {bad}; allowed: {sorted(allowed)}", file=sys.stderr)
        return 2
    if not input_sets:
        print("Empty --input-sets", file=sys.stderr)
        return 2

    with args.context_labels.open("r", encoding="utf-8", newline="") as f:
        all_rows = [r for r in csv.DictReader(f) if r["npz_path"] in idx and r.get("subject_id", "unknown") != "unknown"]
    folds = sorted({r["test_subject"] for r in all_rows})

    view_rows: list[dict[str, Any]] = []
    uniq_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for fold in folds:
        frows = [r for r in all_rows if r["test_subject"] == fold]
        train_full = [r for r in frows if r["role"] == "train" and _finite(r.get("context_high_impact_q75")) and _finite(r.get("peak_fz_context_centered"))]
        test = [r for r in frows if r["role"] == "test" and _finite(r.get("context_high_impact_q75")) and _finite(r.get("peak_fz_context_centered"))]
        if not train_full or not test:
            continue
        train, val = _group_split(train_full, val_ratio=0.15, seed=42)
        if not val:
            continue
        for input_set in input_sets:
            Xtr, Xte = _design(input_set, train, test, Xflat, idx)
            Xtr_val, Xva = _design(input_set, train, val, Xflat, idx)
            ytr = np.asarray([int(float(r["context_high_impact_q75"])) for r in train], dtype=int)
            yva = np.asarray([int(float(r["context_high_impact_q75"])) for r in val], dtype=int)
            yte = np.asarray([int(float(r["context_high_impact_q75"])) for r in test], dtype=int)

            sw_cls: np.ndarray | None = None
            if args.balanced_pose_plus_context and input_set == "pose_plus_context":
                from sklearn.utils.class_weight import compute_sample_weight

                sw_cls = compute_sample_weight("balanced", ytr)
            cls_model, cls_name = _fit_cls_model(Xtr, ytr, sample_weight=sw_cls)
            if sw_cls is not None:
                cls_name = "hgb_cls_balanced"
            if hasattr(cls_model, "predict_proba"):
                p_va = cls_model.predict_proba(Xva)[:, 1]
                p_te = cls_model.predict_proba(Xte)[:, 1]
            else:
                s_va = cls_model.decision_function(Xva)
                s_te = cls_model.decision_function(Xte)
                p_va = (s_va - np.min(s_va)) / (np.ptp(s_va) + 1e-8)
                p_te = (s_te - np.min(s_te)) / (np.ptp(s_te) + 1e-8)
            thr = _select_threshold_from_val(yva, p_va)
            m = compute_binary_metrics(y_true=yte, y_prob=p_te, threshold=thr)
            view_rows.append({"fold": fold, "model": cls_name, "input_set": input_set, "task": "classification", **m})

            cls_val_rows = []
            for r, yt, pp in zip(val, yva, p_va):
                cls_val_rows.append(
                    {
                        "fold": fold,
                        "model": cls_name,
                        "input_set": input_set,
                        "split": "val",
                        "task": "classification",
                        "npz_path": r["npz_path"],
                        "unique_impact_key_candidate": r["unique_impact_key_candidate"],
                        "subject_id": r.get("subject_id", "unknown"),
                        "stage": r.get("stage", "unknown"),
                        "fatigue_state": r.get("fatigue_state", "unknown"),
                        "y_true_cls": float(yt),
                        "y_prob": float(pp),
                    }
                )
            cls_test_rows = []
            for r, yt, pp in zip(test, yte, p_te):
                row = {
                    "fold": fold,
                    "model": cls_name,
                    "input_set": input_set,
                    "split": "test",
                    "task": "classification",
                    "npz_path": r["npz_path"],
                    "unique_impact_key_candidate": r["unique_impact_key_candidate"],
                    "subject_id": r.get("subject_id", "unknown"),
                    "stage": r.get("stage", "unknown"),
                    "fatigue_state": r.get("fatigue_state", "unknown"),
                    "y_true_cls": float(yt),
                    "y_prob": float(pp),
                }
                cls_test_rows.append(row)
                pred_rows.append(row)
            pred_rows.extend(cls_val_rows)
            agg = aggregate_unique_impact(
                [{"unique_impact_key_candidate": r["unique_impact_key_candidate"], "y_true": r["y_true_cls"], "y_prob": r["y_prob"]} for r in cls_test_rows],
                pred_keys=("y_true", "y_prob"),
            )
            um = compute_binary_metrics(agg["y_true"], agg["y_prob"], threshold=thr)
            uniq_rows.append({"fold": fold, "model": cls_name, "input_set": input_set, "task": "classification", **um})

            # Regression
            ytr_r = np.asarray([float(r["peak_fz_context_centered"]) for r in train], dtype=float)
            yte_r = np.asarray([float(r["peak_fz_context_centered"]) for r in test], dtype=float)
            reg_model, reg_name = _fit_reg_model(Xtr, ytr_r)
            yp_te = reg_model.predict(Xte)
            rm = compute_regression_metrics(yte_r, yp_te)
            view_rows.append({"fold": fold, "model": reg_name, "input_set": input_set, "task": "regression:peak_fz_context_centered", **rm})
            reg_val_rows = []
            yp_va = reg_model.predict(Xva)
            yva_r = np.asarray([float(r["peak_fz_context_centered"]) for r in val], dtype=float)
            for r, yt, yp in zip(val, yva_r, yp_va):
                reg_val_rows.append(
                    {
                        "fold": fold,
                        "model": reg_name,
                        "input_set": input_set,
                        "split": "val",
                        "task": "regression:peak_fz_context_centered",
                        "npz_path": r["npz_path"],
                        "unique_impact_key_candidate": r["unique_impact_key_candidate"],
                        "subject_id": r.get("subject_id", "unknown"),
                        "stage": r.get("stage", "unknown"),
                        "fatigue_state": r.get("fatigue_state", "unknown"),
                        "y_true_peak": float(yt),
                        "y_pred_peak": float(yp),
                    }
                )
            reg_rows = []
            for r, yt, yp in zip(test, yte_r, yp_te):
                row = {
                    "fold": fold,
                    "model": reg_name,
                    "input_set": input_set,
                    "split": "test",
                    "task": "regression:peak_fz_context_centered",
                    "npz_path": r["npz_path"],
                    "unique_impact_key_candidate": r["unique_impact_key_candidate"],
                    "subject_id": r.get("subject_id", "unknown"),
                    "stage": r.get("stage", "unknown"),
                    "fatigue_state": r.get("fatigue_state", "unknown"),
                    "y_true_peak": float(yt),
                    "y_pred_peak": float(yp),
                }
                reg_rows.append(row)
                pred_rows.append(row)
            pred_rows.extend(reg_val_rows)
            agg_y = aggregate_unique_impact(
                [{"unique_impact_key_candidate": r["unique_impact_key_candidate"], "y_true": r["y_true_peak"], "y_pred": r["y_pred_peak"]} for r in reg_rows],
                pred_keys=("y_true", "y_pred"),
            )
            urm = compute_regression_metrics(agg_y["y_true"], agg_y["y_pred"])
            uniq_rows.append({"fold": fold, "model": reg_name, "input_set": input_set, "task": "regression:peak_fz_context_centered", **urm})

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = sorted({k for r in rows for k in r.keys()})
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    _write_csv(out_dir / "hgb_metrics_view.csv", view_rows)
    _write_csv(out_dir / "hgb_metrics_unique.csv", uniq_rows)
    _write_csv(out_dir / "hgb_predictions.csv", pred_rows)

    summary = {
        "fold_count": len({r["fold"] for r in view_rows}),
        "input_sets": sorted({r["input_set"] for r in view_rows}),
        "classification_rows": len([r for r in view_rows if r["task"] == "classification"]),
        "regression_rows": len([r for r in view_rows if r["task"] == "regression:peak_fz_context_centered"]),
    }
    (out_dir / "hgb_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote: {out_dir / 'hgb_metrics_view.csv'}")
    print(f"Wrote: {out_dir / 'hgb_metrics_unique.csv'}")
    print(f"Wrote: {out_dir / 'hgb_predictions.csv'}")
    print(f"Wrote: {out_dir / 'hgb_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
