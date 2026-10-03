"""The public model list must construct exactly the declared models."""

from badminton_impact_ai.experiment.models import DEEP_MODELS, build_deep_model


def test_all_declared_deep_models_construct() -> None:
    for name in DEEP_MODELS:
        assert build_deep_model(name, stat_dim=16, context_dim=9, hidden_dim=16) is not None
