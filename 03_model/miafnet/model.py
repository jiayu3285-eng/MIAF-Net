"""Main MIAF-Net architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoConfig, AutoModel

from .audio_mamba import AudioMambaEncoder


@dataclass
class MIAFNetConfig:
    num_classes: int = 3

    text_backbone: str = "roberta-base"
    text_rnn_hidden: int = 256
    text_rnn_layers: int = 1
    text_attention_size: int = 128

    audio_feature_dim: int = 64
    audio_hidden_size: int = 64
    audio_layers: int = 6

    fusion_dim: int = 256
    fusion_hidden: int = 152
    mhsa_heads: int = 4
    dropout: float = 0.3

    use_pretrained_text: bool = True
    use_pretrained_image: bool = True


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    mask = mask.to(dtype=scores.dtype)
    scores = scores.masked_fill(mask <= 0, float("-inf"))
    attn = torch.softmax(scores, dim=dim)
    attn = attn * mask
    attn = attn / attn.sum(dim=dim, keepdim=True).clamp_min(eps)
    return attn


class AdditiveAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, attention_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_size, attention_size)
        self.v = nn.Linear(attention_size, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.proj(hidden_states))
        h = self.drop(h)
        scores = self.v(h).squeeze(-1)
        weights = masked_softmax(scores, mask)
        return torch.einsum("bl,bld->bd", weights, hidden_states)


class TextEncoder(nn.Module):
    """RoBERTa + BiLSTM + Additive Attention."""

    def __init__(self, cfg: MIAFNetConfig):
        super().__init__()
        if cfg.use_pretrained_text:
            self.backbone = AutoModel.from_pretrained(cfg.text_backbone)
        else:
            text_config = AutoConfig.from_pretrained(cfg.text_backbone)
            self.backbone = AutoModel.from_config(text_config)
        hidden = int(self.backbone.config.hidden_size)

        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=cfg.text_rnn_hidden,
            num_layers=cfg.text_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if cfg.text_rnn_layers > 1 else 0.0,
        )
        self.pool = AdditiveAttentionPooling(
            hidden_size=cfg.text_rnn_hidden * 2,
            attention_size=cfg.text_attention_size,
            dropout=cfg.dropout,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.out_dim = cfg.text_rnn_hidden * 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence = output.last_hidden_state
        rnn_output, _ = self.bilstm(sequence)
        pooled = self.pool(rnn_output, attention_mask)
        return self.drop(pooled)


class ECABlock(nn.Module):
    """Efficient Channel Attention over pooled visual channels."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.conv(x.unsqueeze(1))
        weights = self.sigmoid(weights).squeeze(1)
        return x * weights


class SwinTECAEncoder(nn.Module):
    """Swin-T visual encoder followed by ECA channel recalibration."""

    def __init__(self, cfg: MIAFNetConfig):
        super().__init__()
        weights = models.Swin_T_Weights.DEFAULT if cfg.use_pretrained_image else None
        swin = models.swin_t(weights=weights)
        self.features = swin.features
        self.norm = swin.norm
        self.permute = swin.permute
        self.avgpool = swin.avgpool
        self.out_dim = int(swin.head.in_features)
        self.eca = ECABlock(self.out_dim, kernel_size=3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.eca(x)


class AudioEncoder(nn.Module):
    """Audio Mamba encoder for 64-bin log-Mel spectrograms."""

    def __init__(self, cfg: MIAFNetConfig):
        super().__init__()
        self.encoder = AudioMambaEncoder(
            feature_dim=cfg.audio_feature_dim,
            hidden_size=cfg.audio_hidden_size,
            num_layers=cfg.audio_layers,
            dropout=cfg.dropout,
        )
        self.out_dim = cfg.audio_hidden_size

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)


class DCMIFMHSAFusion(nn.Module):
    """DCMIF interaction tokens followed by MHSA aggregation."""

    def __init__(self, cfg: MIAFNetConfig):
        super().__init__()
        self.fusion_dim = cfg.fusion_dim
        self.mhsa = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.mhsa_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.fusion_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    @staticmethod
    def pair_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y

    @staticmethod
    def pair_absdiff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.abs(x - y)

    def forward(self, text: torch.Tensor, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.pair_mul(text, image),
                self.pair_absdiff(text, image),
                self.pair_mul(text, audio),
                self.pair_absdiff(text, audio),
                self.pair_mul(image, audio),
                self.pair_absdiff(image, audio),
            ],
            dim=1,
        )
        attended, _ = self.mhsa(tokens, tokens, tokens, need_weights=False)
        attended = self.norm(attended + tokens)
        pooled = attended.mean(dim=1)
        pooled = self.drop(pooled)
        return self.mlp(pooled)


class MIAFNet(nn.Module):
    """Multimodal Interactive Attention Fusion Network."""

    def __init__(self, cfg: MIAFNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or MIAFNetConfig()

        self.text_encoder = TextEncoder(self.cfg)
        self.image_encoder = SwinTECAEncoder(self.cfg)
        self.audio_encoder = AudioEncoder(self.cfg)

        self.text_proj = nn.Linear(self.text_encoder.out_dim, self.cfg.fusion_dim)
        self.image_proj = nn.Linear(self.image_encoder.out_dim, self.cfg.fusion_dim)
        self.audio_proj = nn.Linear(self.audio_encoder.out_dim, self.cfg.fusion_dim)

        self.fusion = DCMIFMHSAFusion(self.cfg)
        self.classifier = nn.Linear(self.cfg.fusion_dim, self.cfg.num_classes)

    def encode_modalities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        mel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_feature = self.text_encoder(input_ids, attention_mask)
        text_feature = self.text_proj(text_feature)

        batch, frames, channels, height, width = images.shape
        image_input = images.reshape(batch * frames, channels, height, width)
        image_feature = self.image_encoder(image_input)
        image_feature = image_feature.reshape(batch, frames, -1).mean(dim=1)
        image_feature = self.image_proj(image_feature)

        audio_feature = self.audio_encoder(mel)
        audio_feature = self.audio_proj(audio_feature)

        return text_feature, image_feature, audio_feature

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        mel: torch.Tensor,
        return_features: bool = False,
    ):
        text_feature, image_feature, audio_feature = self.encode_modalities(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            mel=mel,
        )
        fused = self.fusion(text_feature, image_feature, audio_feature)
        logits = self.classifier(fused)

        if return_features:
            return {
                "logits": logits,
                "text_feature": text_feature,
                "image_feature": image_feature,
                "audio_feature": audio_feature,
                "fusion_feature": fused,
            }
        return logits
