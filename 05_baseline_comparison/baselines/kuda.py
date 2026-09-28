"""Knowledge-guided Dynamic Modality Attention baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLPClassifier, ModalityProjector


class KuDA(nn.Module):
    def __init__(self, text_dim=512, image_dim=768, audio_dim=256, hidden_dim=128, fusion_hidden=512, num_classes=3, dropout=0.3):
        super().__init__()
        self.project = ModalityProjector(text_dim, image_dim, audio_dim, hidden_dim, dropout)
        self.knowledge_proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Tanh())
        self.modality_scorer = nn.Linear(hidden_dim, 1)
        self.classifier = MLPClassifier(hidden_dim, fusion_hidden, num_classes, dropout)

    def forward(self, text, image, audio):
        t, i, a = self.project(text, image, audio)
        knowledge = self.knowledge_proj(torch.cat([t, i, a], dim=1))
        tokens = torch.stack([t + knowledge, i + knowledge, a + knowledge], dim=1)
        weights = torch.softmax(self.modality_scorer(tokens).squeeze(-1), dim=1).unsqueeze(-1)
        fused = (tokens * weights).sum(dim=1)
        return self.classifier(fused)

