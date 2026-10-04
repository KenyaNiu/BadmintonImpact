"""Forward-pass smoke tests for every registered neural model."""

from __future__ import annotations

import pytest
import torch

from badminton_impact_ai.models import DEEP_MODELS, build_deep_model

B, T, D = 4, 120, 34


@pytest.mark.parametrize("name", sorted(DEEP_MODELS))
def test_forward_shapes(name: str) -> None:
    model = build_deep_model(name, stat_dim=16, context_dim=9, hidden_dim=64)
    output = model(torch.randn(B, T, D), torch.ones(B, T), torch.randn(B, 9), torch.randn(B, 16))
    assert set(output) == {"cls_logits", "peak_pred"}
    assert output["cls_logits"].shape == output["peak_pred"].shape == (B,)


def test_full_model_diagnostics() -> None:
    model = build_deep_model("cn_hildnet", stat_dim=16, context_dim=9, hidden_dim=64)
    output = model(
        torch.randn(B, T, D), torch.ones(B, T), torch.randn(B, 9), torch.randn(B, 16), return_diagnostics=True
    )
    assert output["attn_alpha"].shape == (B, T)
    assert output["g_fused"].shape == (B, 64)
