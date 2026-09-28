"""Perturbation utilities for MIAF-Net robustness evaluation."""

from __future__ import annotations

import torch


ROBUSTNESS_MODES = [
    "clean",
    "visual_gaussian_noise",
    "visual_center_occlusion",
    "audio_gaussian_noise",
    "visual_missing",
    "audio_missing",
    "text_missing",
    "image_audio_missing",
]


MODE_LABELS = {
    "clean": "Clean test set",
    "visual_gaussian_noise": "Visual Gaussian noise",
    "visual_center_occlusion": "Visual center occlusion",
    "audio_gaussian_noise": "Audio Gaussian noise",
    "visual_missing": "Visual modality missing",
    "audio_missing": "Audio modality missing",
    "text_missing": "Text modality missing",
    "image_audio_missing": "Image and audio modalities missing",
}


def apply_perturbation(batch: dict[str, torch.Tensor], mode: str, device: torch.device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    images = batch["images"].to(device, non_blocking=True)
    mel = batch["mel"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    if mode == "clean":
        pass
    elif mode == "visual_gaussian_noise":
        images = images + torch.randn_like(images) * 0.10
    elif mode == "visual_center_occlusion":
        images = images.clone()
        _, _, _, height, width = images.shape
        occ_h, occ_w = height // 4, width // 4
        h0, w0 = (height - occ_h) // 2, (width - occ_w) // 2
        images[:, :, :, h0 : h0 + occ_h, w0 : w0 + occ_w] = 0
    elif mode == "audio_gaussian_noise":
        mel = mel + torch.randn_like(mel) * 0.10
    elif mode == "visual_missing":
        images = torch.zeros_like(images)
    elif mode == "audio_missing":
        mel = torch.zeros_like(mel)
    elif mode == "text_missing":
        input_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(attention_mask)
    elif mode == "image_audio_missing":
        images = torch.zeros_like(images)
        mel = torch.zeros_like(mel)
    else:
        raise ValueError(f"Unknown robustness mode: {mode}")

    return input_ids, attention_mask, images, mel, labels

