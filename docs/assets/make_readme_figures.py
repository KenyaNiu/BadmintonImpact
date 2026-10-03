"""Regenerate the README result charts from a completed run.

Usage: python3 docs/assets/make_readme_figures.py --run-dir outputs/runs/corrected_q75
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from badminton_impact_ai.stats.paired import paired_fold_summary  # noqa: E402

BLUE, ORANGE, GREY, RED = "#0072B2", "#E69F00", "#9AA0A6", "#B23A48"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load_unique_auroc(run: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for path in glob.glob(str(run / "metrics" / "sub_*__*__*.csv")):
        fold, model, mode = Path(path).stem.split("__")
        for row in csv.DictReader(open(path)):
            if row.get("task") == "classification" and row["resolution"] == "unique_impact":
                out[(model, mode)][fold] = float(row["AUROC"])
    return out


def models_chart(auroc, out: Path) -> None:
    names = {"hgb_context": "HGB (context only)", "hgb_pose": "HGB (pose only)", "hgb_pose_context": "HGB (pose + context)",
             "tcn": "TCN", "bigru": "BiGRU", "stgcn_light": "STGCN-Light", "transformer": "Temporal Transformer",
             "cn_hildnet": "CN-HiLDNet"}
    rows = []
    for key, label in names.items():
        values = np.array(list(auroc[(key, "cls_peak")].values()))
        rows.append((label, values.mean(), values.std(ddof=1), key))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = [BLUE if r[3] == "cn_hildnet" else ORANGE if r[3] == "hgb_pose_context" else GREY for r in rows]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], xerr=[r[2] for r in rows], color=colors, ecolor="#333", capsize=3, height=0.62)
    ax.axvline(0.5, color="#555", lw=0.8, ls=":")
    for i, r in enumerate(rows):
        ax.text(0.02, i, f"{r[1]:.3f}", va="center", color="white", fontweight="bold", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Unique-impact AUROC (mean ± SD over 10 LOSO folds); dotted line = chance")
    ax.set_title("Ranking held-out landings: CN-HiLDNet vs. baselines", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def top20_chart(run: Path, out: Path) -> None:
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(open(run / "analysis" / "top_fraction_by_fold.csv")):
        if row["task_mode"] == "cls_peak" and row["model"] in ("hgb_pose_context", "cn_hildnet"):
            for m in ("precision_at_fraction", "recall_at_fraction", "ndcg_at_fraction"):
                data[row["model"]][m].append(float(row[m]))
    labels = ["Precision@20%", "Recall@20%", "NDCG@20%"]
    keys = ["precision_at_fraction", "recall_at_fraction", "ndcg_at_fraction"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for off, (model, lab, col) in zip((-0.19, 0.19), (("hgb_pose_context", "HGB (pose + context)", ORANGE), ("cn_hildnet", "CN-HiLDNet", BLUE))):
        means = [np.mean(data[model][k]) for k in keys]
        sds = [np.std(data[model][k], ddof=1) for k in keys]
        ax.bar(x + off, means, 0.36, yerr=sds, color=col, capsize=3, label=lab, ecolor="#333")
        for xi, m in zip(x + off, means):
            ax.text(xi, 0.03, f"{m:.3f}", ha="center", color="white", fontweight="bold", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Reviewing only the top 20% of landings in each trial", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def ablation_chart(auroc, out: Path) -> None:
    full = auroc[("cn_hildnet", "cls_peak")]
    variants = [("Mean-pool encoder", ("mean_pool", "cls_peak")), ("Sequence only", ("sequence_attention", "cls_only")),
                ("No context", ("no_context", "cls_peak")), ("No pose statistics", ("no_stats", "cls_peak")),
                ("No attention pooling", ("no_attention_pool", "cls_peak")), ("Classification only", ("cn_hildnet", "cls_only"))]
    rows = []
    for label, key in variants:
        s = paired_fold_summary(full, auroc[key])  # full minus variant: positive = full model better
        rows.append((label, s["mean_difference"], *s["bootstrap_ci_95"]))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for i, (label, m, lo, hi) in enumerate(rows):
        sig = lo > 0 or hi < 0
        ax.barh(i, m, color=RED if sig else GREY, height=0.6)
        ax.plot([lo, hi], [i, i], color="#222", lw=1.2)
        ax.text(max(hi, m) + 0.008, i, f"{m:+.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("AUROC drop vs. full CN-HiLDNet (bars = mean; lines = 95% bootstrap CI)")
    ax.set_title("AUROC lost when one component is removed", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    auroc = load_unique_auroc(args.run_dir)
    models_chart(auroc, args.out_dir / "results_models.png")
    top20_chart(args.run_dir, args.out_dir / "results_top20.png")
    ablation_chart(auroc, args.out_dir / "results_ablation.png")


if __name__ == "__main__":
    main()
