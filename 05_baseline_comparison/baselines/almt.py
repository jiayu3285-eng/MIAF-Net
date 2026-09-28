"""Adaptive Language-guided Multimodal Transformer baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLPClassifier, ModalityProjector


class ALMT(nn.Module):
    def __init__(self, text_dim=512, image_dim=768, audio_dim=256, hidden_dim=128, num_heads=4, layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.project = ModalityProjector(text_dim, image_dim, audio_dim, hidden_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.language_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.Sigmoid())
        self.classifier = MLPClassifier(hidden_dim, hidden_dim, num_classes, dropout)

    def forward(self, text, image, audio):
        t, i, a = self.project(text, image, audio)
        gates = self.language_gate(t)
        gi, ga = gates.chunk(2, dim=1)
        tokens = torch.stack([t, gi * i, ga * a], dim=1)
        encoded = self.encoder(tokens)
        return self.classifier(encoded.mean(dim=1))

