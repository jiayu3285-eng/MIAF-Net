"""Bi-Bimodal Fusion Network baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLPClassifier, ModalityProjector


class BBFN(nn.Module):
    def __init__(self, text_dim=512, image_dim=768, audio_dim=256, hidden_dim=128, fusion_hidden=512, num_classes=3, dropout=0.3):
        super().__init__()
        self.project = ModalityProjector(text_dim, image_dim, audio_dim, hidden_dim, dropout)
        self.pair_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.classifier = MLPClassifier(hidden_dim * 3, fusion_hidden, num_classes, dropout)

    def fuse_pair(self, x, y):
        gate = self.pair_gate(torch.cat([x, y], dim=1))
        return gate * x + (1.0 - gate) * y

    def forward(self, text, image, audio):
        t, i, a = self.project(text, image, audio)
        ti = self.fuse_pair(t, i)
        ta = self.fuse_pair(t, a)
        ia = self.fuse_pair(i, a)
        return self.classifier(torch.cat([ti, ta, ia], dim=1))

