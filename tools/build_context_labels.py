#!/usr/bin/env python3
"""Build fold-safe LOSO context-normalized labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument(
        "--percentile",
        type=float,
        default=75.0,
        help="Train-only quantile for context threshold τ^{(k)} (paper eq. 1). Default 75 ≙ Q0.75.",
    )
    return p.parse_args()


def _as_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _stats(vals: list[float], percentile: float) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    pq = float(np.clip(percentile, 0.0, 100.0))
    thr = float(np.percentile(arr, pq))
    return {
        "threshold": thr,
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr) if np.std(arr) > 1e-8 else 1.0),
        "count": int(arr.size),
        "percentile_used": pq,
    }


def main() -> int:
    args = parse_args()
    pct = float(args.percentile)
    with args.labels.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    data = [
        r
        for r in rows
        if _as_bool(r.get("valid_label")) and _as_bool(r.get("peak_fz_valid")) and r.get("subject_id", "unknown") != "unknown"
    ]
    subjects = sorted({r["subject_id"] for r in data})
    out_rows = []
    fold_json: dict[str, Any] = {}
    total_fallback = 0
    total_assign = 0
    warnings = []

    for i, test_subject in enumerate(subjects):
        fold_id = f"loso_{i:03d}_{test_subject}"
        tr = [r for r in data if r["subject_id"] != test_subject]
        te = [r for r in data if r["subject_id"] == test_subject]
        g_stats = _stats([float(r["peak_fz"]) for r in tr], pct)
        ctx_vals = defaultdict(list)
        stg_vals = defaultdict(list)
        for r in tr:
            ctx = f"{r.get('stage','unknown')}|{r.get('fatigue_state','unknown')}"
            stg = r.get("stage", "unknown")
            v = float(r["peak_fz"])
            ctx_vals[ctx].append(v)
            stg_vals[stg].append(v)
        ctx_stats = {k: _stats(v, pct) for k, v in ctx_vals.items()}
        stg_stats = {k: _stats(v, pct) for k, v in stg_vals.items()}
        fold_ctx_info = {}
        fallback_count = 0

        for role, subset in (("train", tr), ("test", te)):
            for r in subset:
                ctx = f"{r.get('stage','unknown')}|{r.get('fatigue_state','unknown')}"
                stg = r.get("stage", "unknown")
                stat = None
                src = ""
                if ctx in ctx_stats and ctx_stats[ctx]["count"] >= 50:
                    stat = ctx_stats[ctx]
                    src = "context"
                elif stg in stg_stats and stg_stats[stg]["count"] >= 50:
                    stat = stg_stats[stg]
                    src = "stage"
                else:
                    stat = g_stats
                    src = "global"
                if src != "context":
                    fallback_count += 1
                peak = float(r["peak_fz"])
                centered = peak - stat["median"]
                z = (peak - stat["mean"]) / stat["std"]
                thr = float(stat["threshold"])
                row = dict(r)
                row.update(
                    {
                        "fold_id": fold_id,
                        "test_subject": test_subject,
                        "role": role,
                        "context_key_v1": ctx,
                        "context_threshold_q75": thr,
                        "threshold_source": src,
                        "context_high_impact_q75": int(peak >= thr),
                        "peak_fz_context_centered": centered,
                        "peak_fz_context_z": z,
                    }
                )
                out_rows.append(row)
                total_assign += 1
                fold_ctx_info.setdefault(ctx, {"used_source": src, "count": 0})
                fold_ctx_info[ctx]["count"] += 1
        total_fallback += fallback_count
        tr_pos = np.mean([int(float(r["context_high_impact_q75"])) for r in out_rows if r["fold_id"] == fold_id and r["role"] == "train"])
        te_pos = np.mean([int(float(r["context_high_impact_q75"])) for r in out_rows if r["fold_id"] == fold_id and r["role"] == "test"])
        fold_json[fold_id] = {
            "test_subject": test_subject,
            "global_train_stats": g_stats,
            "context_stats_train": ctx_stats,
            "stage_stats_train": stg_stats,
            "context_usage": fold_ctx_info,
            "train_positive_rate": float(tr_pos),
            "test_positive_rate": float(te_pos),
            "fallback_count": fallback_count,
        }
        if any(v["used_source"] != "context" for v in fold_ctx_info.values()):
            warnings.append(f"{fold_id}: some contexts used fallback")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else []
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    out_json = {
        "fold_count": len(subjects),
        "percentile": pct,
        "folds": fold_json,
        "fallback_rate": float(total_fallback / max(total_assign, 1)),
        "warnings": warnings,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_json, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
