"""Basic import smoke tests for project scaffolding."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.paths import DatasetPaths
from models.icsi_net import ICSINet


def test_imports_and_basic_objects() -> None:
    paths = DatasetPaths()
    model = ICSINet()

    assert isinstance(paths.project_root.as_posix(), str)
    assert model.name == "icsi_net_placeholder"
