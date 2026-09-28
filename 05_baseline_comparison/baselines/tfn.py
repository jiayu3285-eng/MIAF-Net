"""Tensor Fusion Network baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLPClassifier, ModalityProjector


class TFN(nn.Module):
    def __init__(self, text_dim=512, image_dim=768, audio_dim=256, hidden_dim=64, fusion_hidden=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.project = ModalityProjector(text_dim, image_dim, audio_dim, hidden_dim, dropout)
        self.classifier = MLPClassifier((hidden_dim + 1) ** 3, fusion_hidden, num_classes, dropout)

    def forward(self, text, image, audio):
        t, i, a = self.project(text, image, audio)
        ones = torch.ones(t.size(0), 1, device=t.device, dtype=t.dtype)
        t = torch.cat([ones, t], dim=1)
        i = torch.cat([ones, i], dim=1)
        a = torch.cat([ones, a], dim=1)
        fusion = torch.einsum("bi,bj,bk->bijk", t, i, a).flatten(1)
        return self.classifier(fusion)

