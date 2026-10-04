"""CN-HiLDNet, its ablations and the sequence-only baselines."""

from __future__ import annotations

import torch
from torch import nn

from .backbones import AttentionPoolEncoder, FrameMeanEncoder, TemporalTCNEncoder

SEQUENCE_ENCODERS = ("attention", "tcn_mean", "frame_mean")


class RankingHeads(nn.Module):
    """Classification logit (the ranking score) and an auxiliary context-centred peak-force output."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.cls_head = nn.Linear(in_dim, 1)
        self.peak_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cls_logits": self.cls_head(x).squeeze(-1), "peak_pred": self.peak_head(x).squeeze(-1)}


def _mlp(in_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())


class SequenceBaseline(nn.Module):
    """A pose-sequence encoder followed by the ranking heads (no statistics, no context)."""

    def __init__(self, encoder: nn.Module, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = RankingHeads(hidden_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        x_context: torch.Tensor | None = None,
        x_stat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(x_seq, seq_mask)
        return self.heads(pooled)

    def pool(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention-pooled embedding and frame weights (attention encoders only)."""
        return self.encoder.pool(x_seq, seq_mask)


class CNHiLDNet(nn.Module):
    """Three-branch network: sequence encoder, window statistics and protocol context, fused before the heads.

    ``sequence`` selects the temporal branch (``"attention"`` is the full model) and ``stat_dim`` /
    ``context_dim`` set to ``None`` drop the corresponding branch, which yields the ablations.
    """

    def __init__(
        self,
        seq_dim: int = 34,
        stat_dim: int | None = 16,
        context_dim: int | None = 9,
        hidden_dim: int = 128,
        sequence: str = "attention",
    ) -> None:
        super().__init__()
        if sequence == "attention":
            self.seq_branch: nn.Module = AttentionPoolEncoder(seq_dim, hidden_dim)
            self.seq_proj: nn.Module | None = nn.Linear(hidden_dim, hidden_dim)
        elif sequence == "tcn_mean":
            self.seq_branch, self.seq_proj = TemporalTCNEncoder(seq_dim, hidden_dim, depth=4), None
        elif sequence == "frame_mean":
            self.seq_branch, self.seq_proj = FrameMeanEncoder(seq_dim, hidden_dim), None
        else:
            raise ValueError(f"unknown sequence encoder {sequence!r}; expected one of {SEQUENCE_ENCODERS}")
        self.stat_branch = _mlp(stat_dim, 64) if stat_dim else None
        self.ctx_branch = _mlp(context_dim, 32) if context_dim else None
        fused_dim = hidden_dim + (64 if stat_dim else 0) + (32 if context_dim else 0)
        self.fusion = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = RankingHeads(hidden_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        x_context: torch.Tensor | None = None,
        x_stat: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        if (self.stat_branch is not None and x_stat is None) or (self.ctx_branch is not None and x_context is None):
            raise ValueError("x_stat and x_context are required by the enabled branches")
        alpha = None
        if self.seq_proj is not None:
            pooled, alpha = self.seq_branch.pool(x_seq, seq_mask)
            parts = [self.seq_proj(pooled)]
        else:
            parts = [self.seq_branch(x_seq, seq_mask)[1]]
        if self.stat_branch is not None:
            parts.append(self.stat_branch(x_stat))
        if self.ctx_branch is not None:
            parts.append(self.ctx_branch(x_context))
        fused = self.fusion(torch.cat(parts, dim=1))
        output = self.heads(fused)
        if return_diagnostics:
            output = {**output, "attn_alpha": alpha, "g_fused": fused}
        return output
