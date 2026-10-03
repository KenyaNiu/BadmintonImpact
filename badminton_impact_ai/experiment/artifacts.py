"""Generate paper-facing tables only from one completed canonical run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def make_paper_artifacts(run_dir: Path, out_dir: Path) -> Path:
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
    csv_path = out_dir / "main_results.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    classification = [row for row in rows if row["task"] == "classification" and row["task_mode"] == "cls_peak"]
    tex = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & Resolution & AUROC mean & AUROC SD \\",
        r"\midrule",
    ]
    for row in classification:
        tex.append(
            f"{row['model'].replace('_', r'\_')} & {row['resolution'].replace('_', r'\_')} & "
            f"{row['AUROC_mean']:.3f} & {row['AUROC_std']:.3f} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "main_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (out_dir / "paired_comparisons.json").write_text(
        json.dumps(summary["paired_comparisons"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    artifacts = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    return manifest_path
