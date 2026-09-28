"""Fusion-stage ablation modules for MIAF-Net.

Inputs are projected modality vectors:

- text:  [B, D]
- image: [B, D]
- audio: [B, D]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DirectConcatFusion(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 152, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, text, image, audio):
        return self.classifier(torch.cat([text, image, audio], dim=1))


class DCMIFOnlyFusion(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 152, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim * 6, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    @staticmethod
    def tokens(text, image, audio):
        return [
            text * image,
            torch.abs(text - image),
            text * audio,
            torch.abs(text - audio),
            image * audio,
            torch.abs(image - audio),
        ]

    def forward(self, text, image, audio):
        return self.classifier(torch.cat(self.tokens(text, image, audio), dim=1))


class TokenAttentionFusion(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 152, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1, bias=False))
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, text, image, audio):
        tokens = torch.stack(DCMIFOnlyFusion.tokens(text, image, audio), dim=1)
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1)
        return self.classifier(pooled)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4, hidden: int = 152, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, text, image, audio):
        query = text.unsqueeze(1)
        key_value = torch.stack([image, audio], dim=1)
        out, _ = self.attn(query, key_value, key_value, need_weights=False)
        out = self.norm(out + query).squeeze(1)
        return self.classifier(out)


class BilinearGateFusion(nn.Module):
    def __init__(self, dim: int = 256, rank: int = 64, hidden: int = 152, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.u = nn.Linear(dim, rank, bias=False)
        self.v = nn.Linear(dim, rank, bias=False)
        self.gate = nn.Sequential(nn.Linear(rank * 3, dim * 3), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _bilinear(self, x, y):
        return self.u(x) * self.v(y)

    def forward(self, text, image, audio):
        bilinear = torch.cat(
            [self._bilinear(text, image), self._bilinear(text, audio), self._bilinear(image, audio)],
            dim=1,
        )
        concat = torch.cat([text, image, audio], dim=1)
        return self.classifier(concat * self.gate(bilinear))


class DCMIFMHSAFusion(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4, hidden: int = 152, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, text, image, audio):
        tokens = torch.stack(DCMIFOnlyFusion.tokens(text, image, audio), dim=1)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        out = self.norm(out + tokens).mean(dim=1)
        return self.classifier(out)


FUSION_VARIANTS = {
    "direct_concat": DirectConcatFusion,
    "dcmif_only": DCMIFOnlyFusion,
    "token_attention": TokenAttentionFusion,
    "cross_attention": CrossAttentionFusion,
    "bilinear_gate": BilinearGateFusion,
    "dcmif_mhsa": DCMIFMHSAFusion,
}
