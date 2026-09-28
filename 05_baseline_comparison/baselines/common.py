"""Common blocks for baseline comparison models.

The baseline modules operate on aligned modality representations:

- text:  [B, D_t]
- image: [B, D_i]
- audio: [B, D_a]

This keeps the comparison focused on multimodal fusion while sharing the same
data split, preprocessing, and evaluation protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, audio_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.image = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.audio = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, text: torch.Tensor, image: torch.Tensor, audio: torch.Tensor):
        return self.text(text), self.image(image), self.audio(audio)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

