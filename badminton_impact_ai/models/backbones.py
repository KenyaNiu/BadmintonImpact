"""Temporal and spatio-temporal sequence encoders.

Every encoder maps a padded pose batch ``(B, T, D)`` and a validity mask ``(B, T)`` to
``(frame_features (B, T, H), pooled (B, H))``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# COCO-17 skeleton (undirected edges) used by the lightweight ST-GCN baseline.
COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)  # fmt: skip


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average ``(B, T, H)`` features over the valid frames of each sequence."""
    weights = mask.unsqueeze(-1)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class ResidualTCNBlock(nn.Module):
    """Two dilated 1-D convolutions with a residual connection; padding frames stay zero."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.conv1(x) * mask)
        residual = self.conv2(residual) * mask
        return self.act(x + residual) * mask


class TemporalTCNEncoder(nn.Module):
    """Dilated residual TCN (dilations 1, 2, 4, ...) with masked mean pooling."""

    def __init__(self, in_dim: int = 34, hidden_dim: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(ResidualTCNBlock(hidden_dim, dilation=2**i) for i in range(depth))

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channel_mask = seq_mask.unsqueeze(1)
        h = self.in_proj(x_seq.transpose(1, 2)) * channel_mask
        for block in self.blocks:
            h = block(h, channel_mask)
        frames = h.transpose(1, 2)
        return frames, masked_mean(frames, seq_mask)


class AttentionPoolEncoder(nn.Module):
    """TCN followed by learned, masked temporal-attention pooling."""

    def __init__(self, in_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TemporalTCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, depth=4)
        self.attn = nn.Linear(hidden_dim, 1)

    def _attend(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames, _ = self.encoder(x_seq, seq_mask)
        logits = self.attn(frames).squeeze(-1).masked_fill(seq_mask <= 0, -1e9)
        alpha = torch.softmax(logits, dim=1)
        return frames, (frames * alpha.unsqueeze(-1)).sum(dim=1), alpha

    def pool(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the attention-pooled embedding and the normalised frame weights ``(B, T)``."""
        _, pooled, alpha = self._attend(x_seq, seq_mask)
        return pooled, alpha

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames, pooled, _ = self._attend(x_seq, seq_mask)
        return frames, pooled


class FrameMeanEncoder(nn.Module):
    """Order-invariant ablation: a per-frame MLP followed by a masked mean over time."""

    def __init__(self, in_dim: int = 34, hidden_dim: int = 128) -> None:
        super().__init__()
        self.frame_embed = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self.frame_embed(x_seq)
        return frames, masked_mean(frames, seq_mask)


class BiGRUEncoder(nn.Module):
    def __init__(self, in_dim: int = 34, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = seq_mask.sum(dim=1).to(dtype=torch.int64)
        if torch.any(lengths <= 0):
            raise ValueError("BiGRUEncoder received an empty sequence")
        packed = nn.utils.rnn.pack_padded_sequence(x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        frames, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x_seq.shape[1])
        return frames, masked_mean(frames, seq_mask)


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 34,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x_seq) + self.pos_emb[:, : x_seq.shape[1], :]
        frames = self.encoder(h, src_key_padding_mask=seq_mask <= 0)
        return frames, masked_mean(frames, seq_mask)


class STGCNLightEncoder(nn.Module):
    """Lightweight spatial graph convolution over the COCO-17 skeleton plus a temporal convolution."""

    def __init__(self, num_joints: int = 17, in_ch: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.register_buffer("adj", self._normalized_adjacency(num_joints), persistent=False)
        self.spatial_fc1 = nn.Linear(in_ch, 32)
        self.spatial_fc2 = nn.Linear(32, 64)
        self.temporal_conv = nn.Conv1d(num_joints * 64, hidden_dim, kernel_size=3, padding=1)

    @staticmethod
    def _normalized_adjacency(num_joints: int) -> torch.Tensor:
        adjacency = torch.eye(num_joints, dtype=torch.float32)
        for i, j in COCO17_EDGES:
            if i < num_joints and j < num_joints:
                adjacency[i, j] = adjacency[j, i] = 1.0
        return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x_seq.shape
        x = x_seq[..., : self.num_joints * 2].reshape(b, t, self.num_joints, 2)
        x = torch.einsum("ij,btjd->btid", self.adj, x)
        x = F.relu(self.spatial_fc2(F.relu(self.spatial_fc1(x))))
        x = x.reshape(b, t, self.num_joints * 64).transpose(1, 2)
        frames = F.relu(self.temporal_conv(x)).transpose(1, 2)
        return frames, masked_mean(frames, seq_mask)
