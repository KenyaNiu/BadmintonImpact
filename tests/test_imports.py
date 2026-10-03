"""Basic import smoke test for the paper pipeline."""

from badminton_impact_ai.experiment import derived_seed
from badminton_impact_ai.models import CNHiLDNet


def test_imports_and_basic_objects() -> None:
    model = CNHiLDNet(context_dim=4, stat_dim=16)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert derived_seed(42, "fold", "model") == derived_seed(42, "fold", "model")
