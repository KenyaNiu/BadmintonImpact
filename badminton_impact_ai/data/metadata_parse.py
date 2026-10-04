"""Best-effort metadata parsing from NPZ path and metadata fields."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SUBJECT_RE = re.compile(r"(sub_\d{3})", re.IGNORECASE)
_CAMERA_RE = re.compile(r"(?:cam(?:era)?[_-]?)(\d+)", re.IGNORECASE)
_IMPACT_RE = re.compile(r"impact[_-]?(\d+)", re.IGNORECASE)
_TRIAL_RE = re.compile(
    r"(fatigue_stage[123]_\d+|stage[123]_\d+|rally_\d+|fatigue_stage[123]|stage[123]|rally)",
    re.IGNORECASE,
)


def _normalize_camera(value: str) -> str:
    m = _CAMERA_RE.search(value)
    if not m:
        return "unknown"
    return f"cam{int(m.group(1))}"


def _parse_stage(text: str) -> str:
    lower = text.lower()
    for tag in (
        "fatigue_stage1",
        "fatigue_stage2",
        "fatigue_stage3",
        "stage1",
        "stage2",
        "stage3",
        "rally",
    ):
        if tag in lower:
            return tag
    return "unknown"


def _parse_fatigue_state(stage: str) -> str:
    if stage.startswith("fatigue_"):
        return "fatigued"
    if stage in {"stage1", "stage2", "stage3", "rally"}:
        return "fresh"
    return "unknown"


def parse_npz_metadata(npz_path: str | Path, npz_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    path_obj = Path(npz_path)
    rel_text = path_obj.as_posix()
    basename = path_obj.name
    metadata = npz_metadata or {}

    subject_match = _SUBJECT_RE.search(rel_text)
    subject_id = subject_match.group(1).lower() if subject_match else "unknown"

    camera_candidates = [
        str(metadata.get("camera", "")),
        str(metadata.get("cam", "")),
        basename,
        rel_text,
    ]
    camera_id = "unknown"
    for cand in camera_candidates:
        normalized = _normalize_camera(cand)
        if normalized != "unknown":
            camera_id = normalized
            break

    trial_candidates = [
        str(metadata.get("trial", "")),
        basename,
        rel_text,
    ]
    trial_id = "unknown"
    for cand in trial_candidates:
        m = _TRIAL_RE.search(cand)
        if m:
            trial_id = m.group(1).lower()
            break

    stage = _parse_stage(" ".join(trial_candidates))
    fatigue_state = _parse_fatigue_state(stage)

    impact_id = "unknown"
    impact_match = _IMPACT_RE.search(rel_text)
    if impact_match:
        impact_id = impact_match.group(1)
    elif "impact_id" in metadata:
        impact_id = str(metadata.get("impact_id", "unknown"))

    if subject_id != "unknown" and trial_id != "unknown" and impact_id != "unknown":
        unique_key = f"{subject_id}__{trial_id}__impact_{impact_id}"
    else:
        approx = re.sub(r"cam(?:era)?[_-]?\d+", "", rel_text, flags=re.IGNORECASE)
        unique_key = approx.replace("//", "/").strip("/")

    parse_flags = {
        "subject_parsed": subject_id != "unknown",
        "camera_parsed": camera_id != "unknown",
        "trial_parsed": trial_id != "unknown",
        "stage_parsed": stage != "unknown",
        "impact_parsed": impact_id != "unknown",
    }

    return {
        "subject_id": subject_id,
        "camera_id": camera_id,
        "trial_id": trial_id,
        "stage": stage,
        "fatigue_state": fatigue_state,
        "impact_id": impact_id,
        "unique_impact_key_candidate": unique_key,
        "parse_flags": parse_flags,
    }
