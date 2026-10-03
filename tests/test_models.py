"""Forward-pass smoke tests for canonical neural models."""

from __future__ import annotations

import torch

from badminton_impact_ai.models import CNHiLDNet, CNHiLDNetMeanPool, DeepTCNBaseline, SequenceAttentionBaseline


def test_models_forward_shapes() -> None:
    b, t, d = 4, 120, 34
    x_seq = torch.randn(b, t, d)
    mask = torch.ones(b, t)
    x_stat = torch.randn(b, 16)
    x_ctx = torch.randn(b, 9)

    models = [
        DeepTCNBaseline(seq_dim=d, hidden_dim=64),
        SequenceAttentionBaseline(seq_dim=d, hidden_dim=64),
        CNHiLDNet(seq_dim=d, stat_dim=16, context_dim=9, hidden_dim=64),
    ]
    for m in models:
        out = m(x_seq=x_seq, seq_mask=mask, x_context=x_ctx, x_stat=x_stat)
        assert set(out.keys()) == {"cls_logits", "peak_pred"}
        for k in out:
            assert out[k].shape[0] == b

    hy = CNHiLDNet(seq_dim=d, stat_dim=16, context_dim=9, hidden_dim=64)
    out_d = hy(
        x_seq=x_seq,
        seq_mask=mask,
        x_context=x_ctx,
        x_stat=x_stat,
        return_diagnostics=True,
    )
    assert out_d["attn_alpha"].shape == (b, x_seq.shape[1])
    assert out_d["g_fused"].shape == (b, 64)
    out_z = hy(x_seq=x_seq, seq_mask=mask, x_context=x_ctx, x_stat=x_stat, zero_context=True)
    assert "cls_logits" in out_z

    mp = CNHiLDNetMeanPool(seq_dim=d, stat_dim=16, context_dim=9, hidden_dim=64)
    out_mp = mp(x_seq=x_seq, seq_mask=mask, x_context=x_ctx, x_stat=x_stat)
    assert set(out_mp.keys()) == {"cls_logits", "peak_pred"}
    assert out_mp["cls_logits"].shape == (b,)
