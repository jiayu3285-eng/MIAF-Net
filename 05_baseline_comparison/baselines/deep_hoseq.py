"""Deep Higher Order Sequence Fusion baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLPClassifier, ModalityProjector


class DeepHOSeq(nn.Module):
    def __init__(self, text_dim=512, image_dim=768, audio_dim=256, hidden_dim=128, fusion_hidden=512, num_classes=3, dropout=0.3):
        super().__init__()
        self.project = ModalityProjector(text_dim, image_dim, audio_dim, hidden_dim, dropout)
        in_dim = hidden_dim * 7
        self.classifier = MLPClassifier(in_dim, fusion_hidden, num_classes, dropout)

    def forward(self, text, image, audio):
        t, i, a = self.project(text, image, audio)
        parts = [t, i, a, t * i, t * a, i * a, t * i * a]
        return self.classifier(torch.cat(parts, dim=1))

