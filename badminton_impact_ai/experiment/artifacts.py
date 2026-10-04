"""Paper-facing tables, generated only from one completed and analysed run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from badminton_impact_ai.io import atomic_write_text, write_csv, write_json


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _results_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & Resolution & AUROC mean & AUROC SD \\",
        r"\midrule",
    ]
    for row in rows:
        model, resolution = _latex_escape(row["model"]), _latex_escape(row["resolution"])
        lines.append(f"{model} & {resolution} & {row['AUROC_mean']:.3f} & {row['AUROC_std']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def make_paper_artifacts(run_dir: Path, out_dir: Path) -> Path:
    """Write the result tables, paired comparisons and a SHA-256 manifest; return the manifest path."""
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError("paper artifacts require a completed run")
    summary_path = run_dir / "analysis" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("run `badminton-impact analyze` first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("cohort_match"):
        raise RuntimeError("paper artifacts require identical test cohorts")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summary["summaries"]
    write_csv(out_dir / "main_results.csv", rows)
    classification = [row for row in rows if row["task"] == "classification" and row["task_mode"] == "cls_peak"]
    atomic_write_text(out_dir / "main_results.tex", _results_table(classification))
    write_json(out_dir / "paired_comparisons.json", summary["paired_comparisons"])

    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out_dir.iterdir()) if path.is_file()
    }
    manifest_path = out_dir / "artifact_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path
