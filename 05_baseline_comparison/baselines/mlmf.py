"""Multitask Low-rank Multimodal Fusion baseline."""

from __future__ import annotations

import torch.nn as nn

from .lmf import LMF


class MLMF(nn.Module):
    def __init__(self, text_dim=512, image_dim=768, audio_dim=256, hidden_dim=128, rank=8, num_classes=3, dropout=0.3):
        super().__init__()
        self.fusion = LMF(text_dim, image_dim, audio_dim, hidden_dim, rank, num_classes, dropout)
        self.aux_text = nn.Linear(hidden_dim, num_classes)
        self.aux_image = nn.Linear(hidden_dim, num_classes)
        self.aux_audio = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, image, audio, return_aux=False):
        logits = self.fusion(text, image, audio)
        if not return_aux:
            return logits
        t, i, a = self.fusion.project(text, image, audio)
        return {
            "logits": logits,
            "text_logits": self.aux_text(t),
            "image_logits": self.aux_image(i),
            "audio_logits": self.aux_audio(a),
        }

