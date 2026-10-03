"""Minimal checks for the learned temporal-attention implementation."""

from __future__ import annotations

import torch

from badminton_impact_ai.models import BiGRUBaseline, CNHiLDNet, SequenceAttentionBaseline


def test_attention_is_normalized_and_has_no_cancelling_token() -> None:
    model = SequenceAttentionBaseline(seq_dim=34, hidden_dim=16)
    x = torch.randn(2, 5, 34)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    _, alpha = model.pool(x, mask)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.equal(alpha[0, 3:], torch.zeros(2))
    assert "contact_token" not in dict(model.named_parameters())


def test_variable_length_predictions_do_not_depend_on_batch_padding() -> None:
    for model in (SequenceAttentionBaseline(seq_dim=34, hidden_dim=16), BiGRUBaseline(seq_dim=34, hidden_dim=16)):
        model.eval()
        short = torch.randn(1, 5, 34)
        single_mask = torch.ones(1, 5)
        padded = torch.zeros(2, 9, 34)
        padded[0, :5] = short[0]
        padded[1] = torch.randn(9, 34)
        batch_mask = torch.zeros(2, 9)
        batch_mask[0, :5] = 1
        batch_mask[1] = 1
        with torch.no_grad():
            single = model(short, single_mask)["cls_logits"][0]
            batched = model(padded, batch_mask)["cls_logits"][0]
        assert torch.allclose(single, batched, atol=1e-6)


def test_full_model_has_no_inactive_parameters() -> None:
    model = CNHiLDNet(seq_dim=34, stat_dim=16, context_dim=9, hidden_dim=16)
    output = model(torch.randn(2, 5, 34), torch.ones(2, 5), torch.randn(2, 9), torch.randn(2, 16))
    (output["cls_logits"].sum() + output["peak_pred"].sum()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
