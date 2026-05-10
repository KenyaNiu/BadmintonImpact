"""Model definitions for M4/M6/M6.5 suites."""

from __future__ import annotations

import torch
import torch.nn as nn

from .temporal_backbones import BiGRUEncoder, STGCNLightEncoder, TemporalTCNEncoder, TemporalTransformerEncoder


class _MultiTaskHeads(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.cls_head = nn.Linear(in_dim, 1)
        self.peak_head = nn.Linear(in_dim, 1)
        self.impulse_head = nn.Linear(in_dim, 1)
        self.loglr_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "cls_logits": self.cls_head(x).squeeze(-1),
            "peak_pred": self.peak_head(x).squeeze(-1),
            "impulse_pred": self.impulse_head(x).squeeze(-1),
            "loglr_pred": self.loglr_head(x).squeeze(-1),
        }


class DeepTCNBaseline(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTCNEncoder(in_dim=seq_dim, hidden_dim=hidden_dim, depth=4)
        self.heads = _MultiTaskHeads(hidden_dim)

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
        self.heads = _MultiTaskHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(x_seq, seq_mask)
        return self.heads(pooled)


class TemporalTransformerBaseline(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTransformerEncoder(in_dim=seq_dim, d_model=hidden_dim, nhead=4, num_layers=2)
        self.heads = _MultiTaskHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(x_seq, seq_mask)
        return self.heads(pooled)


class STGCNLightBaseline(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = STGCNLightEncoder(num_joints=17, in_ch=2, hidden_dim=hidden_dim)
        self.heads = _MultiTaskHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(x_seq, seq_mask)
        return self.heads(pooled)


class ICSISeq(nn.Module):
    def __init__(self, seq_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTCNEncoder(in_dim=seq_dim, hidden_dim=hidden_dim, depth=4)
        self.contact_token = nn.Parameter(torch.zeros(hidden_dim))
        self.attn = nn.Linear(hidden_dim, 1)
        self.heads = _MultiTaskHeads(hidden_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        x_context: torch.Tensor | None = None,
        x_stat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_t, _ = self.encoder(x_seq, seq_mask)  # (B,T,H)
        token = self.contact_token.view(1, 1, -1)
        logits = self.attn(h_t + token).squeeze(-1)  # (B,T)
        logits = logits.masked_fill(seq_mask <= 0, -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (h_t * w).sum(dim=1)
        return self.heads(pooled)


class ICSIHybridContext(nn.Module):
    def __init__(self, seq_dim: int = 34, stat_dim: int = 16, context_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_branch = ICSISeq(seq_dim=seq_dim, hidden_dim=hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ctx_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 64 + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _MultiTaskHeads(hidden_dim)

    def _attn_weights_and_seq_feat(
        self, x_seq: torch.Tensor, seq_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (seq_feat, alpha) with alpha_{i,t} = softmax over time (impact-token pooling, paper eq.~4)."""
        h_t, _ = self.seq_branch.encoder(x_seq, seq_mask)
        token = self.seq_branch.contact_token.view(1, 1, -1)
        logits = self.seq_branch.attn(h_t + token).squeeze(-1)
        logits = logits.masked_fill(seq_mask <= 0, -1e9)
        alpha = torch.softmax(logits, dim=1)
        w = alpha.unsqueeze(-1)
        pooled = (h_t * w).sum(dim=1)
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
            raise ValueError("x_context is required for ICSIHybridContext")
        if x_stat is None:
            raise ValueError("x_stat is required for ICSIHybridContext")
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


class ICSIHybridContextStageHeads(nn.Module):
    """Shared ICSI trunk; separate classification logits for rally vs non-rally stages."""

    def __init__(self, seq_dim: int = 34, stat_dim: int = 16, context_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_branch = ICSISeq(seq_dim=seq_dim, hidden_dim=hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ctx_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 64 + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.cls_rally = nn.Linear(hidden_dim, 1)
        self.cls_non = nn.Linear(hidden_dim, 1)
        self.peak_head = nn.Linear(hidden_dim, 1)

    def _extract_seq_feat(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        h_t, _ = self.seq_branch.encoder(x_seq, seq_mask)
        token = self.seq_branch.contact_token.view(1, 1, -1)
        logits = self.seq_branch.attn(h_t + token).squeeze(-1)
        logits = logits.masked_fill(seq_mask <= 0, -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (h_t * w).sum(dim=1)
        return self.seq_proj(pooled)

    def forward(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        x_context: torch.Tensor | None = None,
        x_stat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x_context is None or x_stat is None:
            raise ValueError("x_context and x_stat are required")
        seq_feat = self._extract_seq_feat(x_seq, seq_mask)
        fused = self.fusion(torch.cat([seq_feat, self.stat_branch(x_stat), self.ctx_branch(x_context)], dim=1))
        lr = self.cls_rally(fused).squeeze(-1)
        lnr = self.cls_non(fused).squeeze(-1)
        peak = self.peak_head(fused).squeeze(-1)
        return {"cls_logits_rally": lr, "cls_logits_non": lnr, "peak_pred": peak}


class ICSIHybridMeanPool(nn.Module):
    """Same three-branch fusion as CN-HiLDNet, but no temporal encoder: per-frame MLP + mean over time.

    Replaces the temporal encoder and impact-token pooling with an order-invariant average over valid frames
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
        self.heads = _MultiTaskHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x_context is None or x_stat is None:
            raise ValueError("x_context and x_stat are required")
        h = self.frame_embed(x_seq)
        m = seq_mask.unsqueeze(-1).clamp(0.0, 1.0)
        denom = m.sum(dim=1).clamp(min=1e-6)
        seq_feat = (h * m).sum(dim=1) / denom
        fused = self.fusion(torch.cat([seq_feat, self.stat_branch(x_stat), self.ctx_branch(x_context)], dim=1))
        return self.heads(fused)


class ICSIHybridNoToken(nn.Module):
    def __init__(self, seq_dim: int = 34, stat_dim: int = 16, context_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTCNEncoder(in_dim=seq_dim, hidden_dim=hidden_dim, depth=4)
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ctx_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 64 + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _MultiTaskHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x_context is None or x_stat is None:
            raise ValueError("x_context and x_stat are required")
        _, seq_feat = self.encoder(x_seq, seq_mask)
        fused = self.fusion(torch.cat([seq_feat, self.stat_branch(x_stat), self.ctx_branch(x_context)], dim=1))
        return self.heads(fused)


class ICSIHybridNoContext(nn.Module):
    def __init__(self, seq_dim: int = 34, stat_dim: int = 16, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_branch = ICSISeq(seq_dim=seq_dim, hidden_dim=hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 64, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _MultiTaskHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x_stat is None:
            raise ValueError("x_stat is required")
        h_t, _ = self.seq_branch.encoder(x_seq, seq_mask)
        token = self.seq_branch.contact_token.view(1, 1, -1)
        logits = self.seq_branch.attn(h_t + token).squeeze(-1).masked_fill(seq_mask <= 0, -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        seq_feat = self.seq_proj((h_t * w).sum(dim=1))
        fused = self.fusion(torch.cat([seq_feat, self.stat_branch(x_stat)], dim=1))
        return self.heads(fused)


class ICSIHybridNoStats(nn.Module):
    def __init__(self, seq_dim: int = 34, context_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_branch = ICSISeq(seq_dim=seq_dim, hidden_dim=hidden_dim)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = _MultiTaskHeads(hidden_dim)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor, x_context: torch.Tensor | None = None, x_stat: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x_context is None:
            raise ValueError("x_context is required")
        h_t, _ = self.seq_branch.encoder(x_seq, seq_mask)
        token = self.seq_branch.contact_token.view(1, 1, -1)
        logits = self.seq_branch.attn(h_t + token).squeeze(-1).masked_fill(seq_mask <= 0, -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        seq_feat = self.seq_proj((h_t * w).sum(dim=1))
        fused = self.fusion(torch.cat([seq_feat, self.ctx_branch(x_context)], dim=1))
        return self.heads(fused)


class ICSINet:
    """Backward-compatible placeholder shim."""

    def __init__(self) -> None:
        self.name = "icsi_net_placeholder"

