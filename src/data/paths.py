"""Path helpers for read-only access to BadmintonGRF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    source_project_root: Path = Path("/home/nky/BadmintonGRF")
    dataset_root: Path = Path("/home/nky/BadmintonGRF/data")
    project_root: Path = Path("/home/nky/Badminton-Impact-AI")

    @property
    def is_read_only_boundary_valid(self) -> bool:
        return self.source_project_root.exists() and self.dataset_root.exists()
