"""CN-HiLDNet and cohort-matched temporal baselines."""

from __future__ import annotations

import torch
import torch.nn as nn

from .temporal_backbones import BiGRUEncoder, STGCNLightEncoder, TemporalTCNEncoder, TemporalTransformerEncoder


class _ClassificationPeakHeads(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.cls_head = nn.Linear(in_dim, 1)
        self.peak_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "cls_logits": self.cls_head(x).squeeze(-1),
            "peak_pred": self.peak_head(x).squeeze(-1),
        }


class DeepTCNBaseline(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTCNEncoder(in_dim=seq_dim, hidden_dim=hidden_dim, depth=4)
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        x_context: torch.Tensor | None = None,
        x_stat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(x_seq, seq_mask)
        return self.heads(pooled)


class BiGRUBaseline(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = BiGRUEncoder(in_dim=seq_dim, hidden_dim=hidden_dim, num_layers=2, dropout=0.1)
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(x_seq, seq_mask)
        return self.heads(pooled)


class TemporalTransformerBaseline(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTransformerEncoder(in_dim=seq_dim, d_model=hidden_dim, nhead=4, num_layers=2)
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(x_seq, seq_mask)
        return self.heads(pooled)


class STGCNLightBaseline(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = STGCNLightEncoder(num_joints=17, in_ch=2, hidden_dim=hidden_dim)
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(x_seq, seq_mask)
        return self.heads(pooled)


class TemporalAttentionEncoder(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTCNEncoder(in_dim=seq_dim, hidden_dim=hidden_dim, depth=4)
        self.attn = nn.Linear(hidden_dim, 1)

    def pool(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return learned temporal-attention pooling and its normalized weights."""
        h_t, _ = self.encoder(x_seq, seq_mask)
        logits = self.attn(h_t).squeeze(-1).masked_fill(seq_mask <= 0, -1e9)
        alpha = torch.softmax(logits, dim=1)
        return (h_t * alpha.unsqueeze(-1)).sum(dim=1), alpha



class SequenceAttentionBaseline(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_branch = TemporalAttentionEncoder(seq_dim=seq_dim, hidden_dim=hidden_dim)
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def pool(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.seq_branch.pool(x_seq, seq_mask)

    def forward(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        x_context: torch.Tensor | None = None,
        x_stat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled, _ = self.pool(x_seq, seq_mask)
        return self.heads(pooled)


class CNHiLDNet(nn.Module):
    def __init__(self, seq_dim: int = 34, stat_dim: int = 16, context_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_branch = TemporalAttentionEncoder(seq_dim=seq_dim, hidden_dim=hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ctx_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 64 + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def _attn_weights_and_seq_feat(
        self, x_seq: torch.Tensor, seq_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected sequence features and learned attention weights."""
        pooled, alpha = self.seq_branch.pool(x_seq, seq_mask)
        seq_feat = self.seq_proj(pooled)
        return seq_feat, alpha

    def _extract_seq_feat(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        seq_feat, _ = self._attn_weights_and_seq_feat(x_seq, seq_mask)
        return seq_feat

    def forward(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        x_context: torch.Tensor | None = None,
        x_stat: torch.Tensor | None = None,
        *,
        zero_context: bool = False,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        if x_context is None:
            raise ValueError("x_context is required for CNHiLDNet")
        if x_stat is None:
            raise ValueError("x_stat is required for CNHiLDNet")
        seq_feat, alpha = self._attn_weights_and_seq_feat(x_seq, seq_mask)
        stat_feat = self.stat_branch(x_stat)
        ctx_feat = self.ctx_branch(x_context)
        if zero_context:
            ctx_feat = torch.zeros_like(ctx_feat)
        fused = self.fusion(torch.cat([seq_feat, stat_feat, ctx_feat], dim=1))
        out = self.heads(fused)
        if return_diagnostics:
            out = dict(out)
            out["attn_alpha"] = alpha
            out["g_fused"] = fused
        return out


class CNHiLDNetMeanPool(nn.Module):
    """Same three-branch fusion as CN-HiLDNet, but no temporal encoder: per-frame MLP + mean over time.

    Replaces the temporal encoder and learned attention pooling with an order-invariant average over valid frames
    (no cross-time mixing in the sequence trunk). Isolates the value of learned temporal dynamics
    versus statistics and context branches alone.
    """

    def __init__(self, seq_dim: int = 34, stat_dim: int = 16, context_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.frame_embed = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ctx_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 64 + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x_context is None or x_stat is None:
            raise ValueError("x_context and x_stat are required")
        h = self.frame_embed(x_seq)
        m = seq_mask.unsqueeze(-1).clamp(0.0, 1.0)
        denom = m.sum(dim=1).clamp(min=1e-6)
        seq_feat = (h * m).sum(dim=1) / denom
        fused = self.fusion(torch.cat([seq_feat, self.stat_branch(x_stat), self.ctx_branch(x_context)], dim=1))
        return self.heads(fused)


class CNHiLDNetNoAttention(nn.Module):
    def __init__(self, seq_dim: int = 34, stat_dim: int = 16, context_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTCNEncoder(in_dim=seq_dim, hidden_dim=hidden_dim, depth=4)
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ctx_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 64 + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x_context is None or x_stat is None:
            raise ValueError("x_context and x_stat are required")
        _, seq_feat = self.encoder(x_seq, seq_mask)
        fused = self.fusion(torch.cat([seq_feat, self.stat_branch(x_stat), self.ctx_branch(x_context)], dim=1))
        return self.heads(fused)


class CNHiLDNetNoContext(nn.Module):
    def __init__(self, seq_dim: int = 34, stat_dim: int = 16, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_branch = TemporalAttentionEncoder(seq_dim=seq_dim, hidden_dim=hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 64, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x_stat is None:
            raise ValueError("x_stat is required")
        pooled, _ = self.seq_branch.pool(x_seq, seq_mask)
        seq_feat = self.seq_proj(pooled)
        fused = self.fusion(torch.cat([seq_feat, self.stat_branch(x_stat)], dim=1))
        return self.heads(fused)


class CNHiLDNetNoStats(nn.Module):
    def __init__(self, seq_dim: int = 34, context_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_branch = TemporalAttentionEncoder(seq_dim=seq_dim, hidden_dim=hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _ClassificationPeakHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x_context is None:
            raise ValueError("x_context is required")
        pooled, _ = self.seq_branch.pool(x_seq, seq_mask)
        seq_feat = self.seq_proj(pooled)
        fused = self.fusion(torch.cat([seq_feat, self.ctx_branch(x_context)], dim=1))
        return self.heads(fused)
