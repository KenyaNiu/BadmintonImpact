"""Temporal and spatiotemporal backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.conv1(x) * mask)
        residual = self.conv2(residual) * mask
        return self.act(x + residual) * mask


class TemporalTCNEncoder(nn.Module):
    """Simple temporal convolution stack with masked mean pooling."""

    def __init__(self, in_dim: int = 34, hidden_dim: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        blocks = []
        for i in range(depth):
            blocks.append(ResidualTCNBlock(hidden_dim, kernel_size=3, dilation=2**i))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_seq: (B,T,D)
        channel_mask = seq_mask.unsqueeze(1)
        h = x_seq.transpose(1, 2)  # (B,D,T)
        h = self.in_proj(h) * channel_mask
        for block in self.blocks:
            h = block(h, channel_mask)
        h_t = h.transpose(1, 2)  # (B,T,H)
        mask = seq_mask.unsqueeze(-1)  # (B,T,1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (h_t * mask).sum(dim=1) / denom
        return h_t, pooled


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
        h_t, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x_seq.shape[1])
        mask = seq_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (h_t * mask).sum(dim=1) / denom
        return h_t, pooled


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
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x_seq.shape
        h = self.in_proj(x_seq)
        h = h + self.pos_emb[:, :t, :]
        pad_mask = seq_mask <= 0
        h_t = self.encoder(h, src_key_padding_mask=pad_mask)
        mask = seq_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (h_t * mask).sum(dim=1) / denom
        return h_t, pooled


class STGCNLightEncoder(nn.Module):
    """Lightweight ST-GCN encoder over COCO-17 graph."""

    def __init__(self, num_joints: int = 17, in_ch: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.register_buffer("adj", self._build_adj(num_joints), persistent=False)
        self.spatial_fc1 = nn.Linear(in_ch, 32)
        self.spatial_fc2 = nn.Linear(32, 64)
        self.temporal_conv = nn.Conv1d(num_joints * 64, hidden_dim, kernel_size=3, padding=1)

    @staticmethod
    def _build_adj(num_joints: int) -> torch.Tensor:
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (0, 5), (0, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 6),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]
        a = torch.eye(num_joints, dtype=torch.float32)
        for i, j in edges:
            if i < num_joints and j < num_joints:
                a[i, j] = 1.0
                a[j, i] = 1.0
        deg = a.sum(dim=1, keepdim=True).clamp_min(1.0)
        return a / deg

    def forward(self, x_seq: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x_seq.shape
        x = x_seq[..., : self.num_joints * 2].reshape(b, t, self.num_joints, 2)
        x = torch.einsum("ij,btjd->btid", self.adj, x)
        x = F.relu(self.spatial_fc1(x))
        x = F.relu(self.spatial_fc2(x))
        x = x.reshape(b, t, self.num_joints * 64).transpose(1, 2)
        h = F.relu(self.temporal_conv(x)).transpose(1, 2)
        mask = seq_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (h * mask).sum(dim=1) / denom
        return h, pooled
