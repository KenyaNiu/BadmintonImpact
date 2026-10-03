"""Model construction and declared input requirements."""

from __future__ import annotations

import torch.nn as nn

from badminton_impact_ai.models import (
    BiGRUBaseline,
    CNHiLDNet,
    CNHiLDNetMeanPool,
    CNHiLDNetNoAttention,
    CNHiLDNetNoContext,
    CNHiLDNetNoStats,
    DeepTCNBaseline,
    SequenceAttentionBaseline,
    STGCNLightBaseline,
    TemporalTransformerBaseline,
)

DEEP_MODELS = {
    "tcn",
    "sequence_attention",
    "cn_hildnet",
    "bigru",
    "transformer",
    "stgcn_light",
    "no_attention_pool",
    "no_context",
    "no_stats",
    "mean_pool",
}
HGB_MODELS = {"hgb_context", "hgb_pose", "hgb_pose_context"}


def build_deep_model(name: str, stat_dim: int, context_dim: int, hidden_dim: int = 128) -> nn.Module:
    if name == "tcn":
        return DeepTCNBaseline(hidden_dim=hidden_dim)
    if name == "sequence_attention":
        return SequenceAttentionBaseline(hidden_dim=hidden_dim)
    if name == "cn_hildnet":
        return CNHiLDNet(stat_dim=stat_dim, context_dim=context_dim, hidden_dim=hidden_dim)
    if name == "bigru":
        return BiGRUBaseline(hidden_dim=hidden_dim)
    if name == "transformer":
        return TemporalTransformerBaseline(hidden_dim=hidden_dim)
    if name == "stgcn_light":
        return STGCNLightBaseline(hidden_dim=hidden_dim)
    if name == "no_attention_pool":
        return CNHiLDNetNoAttention(stat_dim=stat_dim, context_dim=context_dim, hidden_dim=hidden_dim)
    if name == "no_context":
        return CNHiLDNetNoContext(stat_dim=stat_dim, hidden_dim=hidden_dim)
    if name == "no_stats":
        return CNHiLDNetNoStats(context_dim=context_dim, hidden_dim=hidden_dim)
    if name == "mean_pool":
        return CNHiLDNetMeanPool(stat_dim=stat_dim, context_dim=context_dim, hidden_dim=hidden_dim)
    raise ValueError(f"unknown deep model: {name}")


def uses_context(name: str) -> bool:
    return name in {"cn_hildnet", "no_attention_pool", "no_stats", "mean_pool"}


def uses_stats(name: str) -> bool:
    return name in {"cn_hildnet", "no_attention_pool", "no_context", "mean_pool"}
