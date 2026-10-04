"""Name -> model registry shared by the configuration validator, trainer and tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from .backbones import (
    AttentionPoolEncoder,
    BiGRUEncoder,
    STGCNLightEncoder,
    TemporalTCNEncoder,
    TemporalTransformerEncoder,
)
from .networks import CNHiLDNet, SequenceBaseline


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model and which optional inputs it consumes."""

    build: Callable[[int, int, int], nn.Module]  # (stat_dim, context_dim, hidden_dim) -> model
    uses_stats: bool = False
    uses_context: bool = False


def _baseline(factory: Callable[[int], nn.Module]) -> Callable[[int, int, int], nn.Module]:
    return lambda stat_dim, context_dim, hidden: SequenceBaseline(factory(hidden), hidden)


def _fusion(sequence: str, *, stats: bool = True, context: bool = True) -> Callable[[int, int, int], nn.Module]:
    return lambda stat_dim, context_dim, hidden: CNHiLDNet(
        stat_dim=stat_dim if stats else None,
        context_dim=context_dim if context else None,
        hidden_dim=hidden,
        sequence=sequence,
    )


DEEP_MODEL_SPECS: dict[str, ModelSpec] = {
    # sequence-only baselines
    "tcn": ModelSpec(_baseline(lambda h: TemporalTCNEncoder(34, h, depth=4))),
    "bigru": ModelSpec(_baseline(lambda h: BiGRUEncoder(34, h, num_layers=2, dropout=0.1))),
    "transformer": ModelSpec(_baseline(lambda h: TemporalTransformerEncoder(34, d_model=h, nhead=4, num_layers=2))),
    "stgcn_light": ModelSpec(_baseline(lambda h: STGCNLightEncoder(num_joints=17, in_ch=2, hidden_dim=h))),
    "sequence_attention": ModelSpec(_baseline(lambda h: AttentionPoolEncoder(34, h))),
    # full model and its ablations
    "cn_hildnet": ModelSpec(_fusion("attention"), uses_stats=True, uses_context=True),
    "no_attention_pool": ModelSpec(_fusion("tcn_mean"), uses_stats=True, uses_context=True),
    "mean_pool": ModelSpec(_fusion("frame_mean"), uses_stats=True, uses_context=True),
    "no_context": ModelSpec(_fusion("attention", context=False), uses_stats=True),
    "no_stats": ModelSpec(_fusion("attention", stats=False), uses_context=True),
}
DEEP_MODELS = frozenset(DEEP_MODEL_SPECS)
HGB_MODELS = frozenset({"hgb_context", "hgb_pose", "hgb_pose_context"})


def build_deep_model(name: str, stat_dim: int, context_dim: int, hidden_dim: int = 128) -> nn.Module:
    try:
        spec = DEEP_MODEL_SPECS[name]
    except KeyError:
        raise ValueError(f"unknown deep model: {name}") from None
    return spec.build(stat_dim, context_dim, hidden_dim)


def uses_stats(name: str) -> bool:
    return DEEP_MODEL_SPECS[name].uses_stats


def uses_context(name: str) -> bool:
    return DEEP_MODEL_SPECS[name].uses_context
