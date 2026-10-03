"""Models used by the canonical BadmintonImpact experiments."""

from .cn_hildnet import (
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

__all__ = [
    "BiGRUBaseline",
    "CNHiLDNet",
    "CNHiLDNetMeanPool",
    "CNHiLDNetNoAttention",
    "CNHiLDNetNoContext",
    "CNHiLDNetNoStats",
    "DeepTCNBaseline",
    "SequenceAttentionBaseline",
    "STGCNLightBaseline",
    "TemporalTransformerBaseline",
]
