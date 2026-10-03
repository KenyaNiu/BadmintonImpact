#!/usr/bin/env python3
"""Regenerate the three paper result figures from one completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from badminton_impact_ai.metrics.calibration import _probability_logistic, _temperature
from badminton_impact_ai.stats import paired_fold_summary


BLUE = "#0072B2"
ORANGE = "#E69F00"
MAGENTA = "#CC79A7"
GRAY = "#6E6E6E"
GRID = "#B8B8B8"
MODEL_COLORS = {
    "hgb_pose_context": ORANGE,
    "cn_hildnet": BLUE,
    "tcn": MAGENTA,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, pdf_path: Path) -> None:
    """Save editable vector outputs and the journal-requested raster export."""
    fig.savefig(pdf_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(pdf_path.with_suffix(".eps"), bbox_inches="tight")
    fig.savefig(pdf_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _metric_rows(run_dir: Path) -> pd.DataFrame:
    paths = sorted((run_dir / "metrics").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no metric files under {run_dir}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def _prediction_paths(run_dir: Path, model: str) -> list[Path]:
    paths = sorted((run_dir / "predictions").glob(f"*__{model}__cls_peak.csv"))
    if len(paths) != 10:
        raise AssertionError(f"expected 10 prediction files for {model}, found {len(paths)}")
    return paths


def _calibrated_unique_predictions(run_dir: Path, model: str, method: str) -> pd.DataFrame:
    """Fit each fold's calibrator on validation views and aggregate held-out views."""
    folds = []
    for path in _prediction_paths(run_dir, model):
        frame = pd.read_csv(path)
        validation = frame[frame["split"] == "val"]
        test = frame[frame["split"] == "test"].copy()
        if validation.empty or test.empty:
            raise AssertionError(f"validation/test predictions missing in {path.name}")
        labels = validation["y_true_cls"].to_numpy(dtype=float)
        if method == "probability_logistic":
            mapping = _probability_logistic(labels, validation["y_prob"].to_numpy(dtype=float))
            test["calibrated_probability"] = mapping(test["y_prob"].to_numpy(dtype=float))
        elif method == "temperature":
            mapping = _temperature(labels, validation["cls_logit"].to_numpy(dtype=float))
            test["calibrated_probability"] = mapping(test["cls_logit"].to_numpy(dtype=float))
        else:
            raise ValueError(f"unsupported reliability-curve method: {method}")
        grouped = test.groupby("unique_impact_key_candidate", sort=False).agg(
            y_true=("y_true_cls", "mean"),
            y_min=("y_true_cls", "min"),
            y_max=("y_true_cls", "max"),
            y_prob=("calibrated_probability", "mean"),
        )
        if not np.allclose(grouped["y_min"], grouped["y_max"]):
            raise AssertionError(f"inconsistent labels within a physical impact in {path.name}")
        folds.append(grouped[["y_true", "y_prob"]].reset_index(drop=True))
    return pd.concat(folds, ignore_index=True)


def _reliability_points(frame: pd.DataFrame, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    probabilities = frame["y_prob"].to_numpy(dtype=float)
    labels = frame["y_true"].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_probability, observed_frequency = [], []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= low) & (probabilities <= high if index == n_bins - 1 else probabilities < high)
        if np.any(mask):
            mean_probability.append(float(np.mean(probabilities[mask])))
            observed_frequency.append(float(np.mean(labels[mask])))
    return np.asarray(mean_probability), np.asarray(observed_frequency)


def calibration_figure(run_dir: Path, output: Path) -> None:
    curves = [
        ("cn_hildnet", "probability_logistic", "CN-HiLDNet, logistic", BLUE, "o", "-"),
        ("cn_hildnet", "temperature", "CN-HiLDNet, temperature", BLUE, "o", "--"),
        ("hgb_pose_context", "probability_logistic", "HGB, logistic", ORANGE, "s", "-"),
    ]
    fig = plt.figure(figsize=(3.35, 3.22), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=(0.50, 2.50), hspace=0.02)
    legend_axis = fig.add_subplot(grid[0])
    axis = fig.add_subplot(grid[1])
    legend_axis.set(xlim=(0, 1), ylim=(0, 1))
    legend_axis.axis("off")
    axis.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.0, label="Perfect calibration")
    source_rows = []
    for model, method, label, color, marker, linestyle in curves:
        x, y = _reliability_points(_calibrated_unique_predictions(run_dir, model, method))
        source_rows.extend(
            {"model": model, "calibration_method": method, "predicted_probability": px, "observed_frequency": py}
            for px, py in zip(x, y)
        )
        axis.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4.2,
            markeredgewidth=0.5,
            label=label,
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed frequency")
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#D8D8D8", linestyle="-", linewidth=0.4)
    axis.set_axisbelow(True)
    heading = {"ha": "left", "va": "center", "fontsize": 9.0, "color": "#555555", "fontweight": "bold"}
    key_text = {"ha": "left", "va": "center", "fontsize": 9.0}
    legend_axis.text(0.01, 0.72, "Model", **heading)
    legend_axis.plot(0.23, 0.72, color=BLUE, marker="o", linestyle="none", markersize=4.2)
    legend_axis.text(0.27, 0.72, "CN-HiLDNet", **key_text)
    legend_axis.plot(0.72, 0.72, color=ORANGE, marker="s", linestyle="none", markersize=4.2)
    legend_axis.text(0.76, 0.72, "HGB", **key_text)
    legend_axis.text(0.01, 0.22, "Curve", **heading)
    for start, end, label, linestyle in [
        (0.20, 0.27, "Logistic", "-"),
        (0.49, 0.56, "Temperature", "--"),
        (0.85, 0.91, "Ideal", ":"),
    ]:
        legend_axis.plot([start, end], [0.22, 0.22], color="#333333", linestyle=linestyle, linewidth=1.3)
        legend_axis.text(end + 0.02, 0.22, label, **key_text)
    pd.DataFrame(source_rows).to_csv(output.with_name(f"{output.stem}_source_data.csv"), index=False)
    _save(fig, output)


def ablation_figure(run_dir: Path, output: Path) -> None:
    frame = _metric_rows(run_dir)
    frame = frame[(frame["task"] == "classification") & (frame["resolution"] == "unique_impact")]
    full = frame[(frame["model"] == "cn_hildnet") & (frame["task_mode"] == "cls_peak")]
    variants = [
        ("mean_pool", "cls_peak", "Mean-pool encoder"),
        ("sequence_attention", "cls_only", "Sequence-only"),
        ("no_context", "cls_peak", "No context"),
        ("no_stats", "cls_peak", "No pose statistics"),
        ("no_attention_pool", "cls_peak", "No attention pooling"),
        ("cn_hildnet", "cls_only", "Classification only"),
    ]
    full_map = dict(zip(full["fold"], full["AUROC"]))
    summaries = []
    for model, task_mode, label in variants:
        rows = frame[(frame["model"] == model) & (frame["task_mode"] == task_mode)]
        summary = paired_fold_summary(
            full_map,
            dict(zip(rows["fold"], rows["AUROC"])),
            bootstrap_draws=10_000,
            seed=42,
        )
        summaries.append((label, summary["mean_difference"], *summary["bootstrap_ci_95"]))
    summaries.sort(key=lambda item: item[1], reverse=True)
    labels, means, lows, highs = map(np.asarray, zip(*summaries))
    y = np.arange(len(labels))[::-1]
    fig, axis = plt.subplots(figsize=(3.35, 2.75))
    axis.barh(y, means, color=MAGENTA, edgecolor="black", linewidth=0.45, height=0.56)
    axis.errorbar(
        means,
        y,
        xerr=np.vstack([means - lows.astype(float), highs.astype(float) - means]),
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
    )
    for yi, mean, high in zip(y, means.astype(float), highs.astype(float)):
        axis.text(max(high, mean, 0.0) + 0.006, yi, f"{mean:+.3f}", va="center", fontsize=9)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.set_xlim(-0.025, 0.33)
    axis.set_xlabel("AUROC drop (full - variant)")
    axis.grid(axis="x", color=GRID, linestyle=":", linewidth=0.6)
    axis.set_axisbelow(True)
    fig.tight_layout(pad=0.55)
    pd.DataFrame(
        {"variant": labels, "mean_drop": means.astype(float), "ci_95_low": lows.astype(float), "ci_95_high": highs.astype(float)}
    ).to_csv(output.with_name(f"{output.stem}_source_data.csv"), index=False)
    _save(fig, output)


def rally_figure(run_dir: Path, output: Path) -> None:
    frame = pd.read_csv(run_dir / "analysis" / "subgroups.csv")
    frame = frame[
        (frame["resolution"] == "unique_impact")
        & (frame["task_mode"] == "cls_peak")
        & frame["model"].isin(MODEL_COLORS)
        & frame["subgroup"].isin(["non_rally", "rally"])
    ]
    models = [
        ("hgb_pose_context", "HGB"),
        ("cn_hildnet", "CN-HiLDNet"),
        ("tcn", "TCN"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.6), sharey=True)
    x = np.arange(len(models))
    width = 0.34
    source_rows = []
    for panel, axis, metric, title in zip(["a", "b", "c"], axes, ["AUROC", "AUPRC", "F1"], ["AUROC", "AUPRC", "F1 score"]):
        for group, offset, hatch in [
            ("non_rally", -width / 2, ""),
            ("rally", width / 2, "//"),
        ]:
            means, stds = [], []
            for model, _ in models:
                values = frame[(frame["model"] == model) & (frame["subgroup"] == group)][metric]
                if len(values) != 10:
                    raise AssertionError(f"expected 10 folds for {model}/{group}/{metric}")
                means.append(values.mean())
                stds.append(values.std(ddof=1))
                source_rows.append(
                    {
                        "panel": panel,
                        "metric": metric,
                        "model": model,
                        "subgroup": group,
                        "n_folds": len(values),
                        "mean": values.mean(),
                        "standard_deviation": values.std(ddof=1),
                    }
                )
            axis.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                color=[MODEL_COLORS[model] for model, _ in models],
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"ecolor": "#333333", "elinewidth": 0.8, "capsize": 2.2, "capthick": 0.8},
            )
        axis.set_title(title, pad=6)
        axis.text(0.0, 1.04, f"({panel})", transform=axis.transAxes, ha="left", va="bottom", fontweight="bold")
        axis.set_xticks(x, [label for _, label in models])
        axis.set_ylim(0, 1.04)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(axis="y", color=GRID, linestyle=":", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    legend_handles = [
        Patch(facecolor="#B8B8B8", edgecolor="black", linewidth=0.5, label="Structured drills"),
        Patch(facecolor="#B8B8B8", edgecolor="black", linewidth=0.5, hatch="//", label="Rally"),
        Line2D([0], [0], color="#333333", marker="|", markersize=8, linewidth=0.8, label="Mean ± fold SD (n = 10)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.4,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.25, pad=0.55)
    pd.DataFrame(source_rows).to_csv(output.with_name(f"{output.stem}_source_data.csv"), index=False)
    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError("result figures require a completed run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = [
        args.output_dir / "fig03_calibration_corrected.pdf",
        args.output_dir / "fig04_ablation_corrected.pdf",
        args.output_dir / "fig05_rally_corrected.pdf",
    ]
    calibration_figure(args.run_dir, outputs[0])
    ablation_figure(args.run_dir, outputs[1])
    rally_figure(args.run_dir, outputs[2])
    for path in outputs:
        for figure_path in (path, path.with_suffix(".svg"), path.with_suffix(".eps"), path.with_suffix(".tiff")):
            if not figure_path.exists() or figure_path.stat().st_size < 1_000:
                raise RuntimeError(f"figure generation failed: {figure_path}")


if __name__ == "__main__":
    main()
