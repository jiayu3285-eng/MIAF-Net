"""Low-rank Multimodal Fusion baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ModalityProjector


class LMF(nn.Module):
    def __init__(self, text_dim=512, image_dim=768, audio_dim=256, hidden_dim=128, rank=4, num_classes=3, dropout=0.3):
        super().__init__()
        self.project = ModalityProjector(text_dim, image_dim, audio_dim, hidden_dim, dropout)
        self.rank = rank
        self.factor_t = nn.Parameter(torch.randn(rank, hidden_dim + 1, num_classes) * 0.02)
        self.factor_i = nn.Parameter(torch.randn(rank, hidden_dim + 1, num_classes) * 0.02)
        self.factor_a = nn.Parameter(torch.randn(rank, hidden_dim + 1, num_classes) * 0.02)
        self.fusion_weights = nn.Parameter(torch.ones(rank))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, text, image, audio):
        t, i, a = self.project(text, image, audio)
        ones = torch.ones(t.size(0), 1, device=t.device, dtype=t.dtype)
        t = torch.cat([ones, t], dim=1)
        i = torch.cat([ones, i], dim=1)
        a = torch.cat([ones, a], dim=1)
        out_t = torch.einsum("bd,rdc->brc", t, self.factor_t)
        out_i = torch.einsum("bd,rdc->brc", i, self.factor_i)
        out_a = torch.einsum("bd,rdc->brc", a, self.factor_a)
        fused = out_t * out_i * out_a
        return torch.einsum("brc,r->bc", fused, self.fusion_weights) + self.bias

