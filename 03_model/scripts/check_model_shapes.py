"""Check MIAF-Net input and output tensor shapes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from miafnet import MIAFNet, MIAFNetConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight MIAF-Net shape check.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mel-steps", type=int, default=201)
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    args = parse_args()
    cfg = MIAFNetConfig(use_pretrained_text=False, use_pretrained_image=False)
    model = MIAFNet(cfg)
    model.eval()

    batch = args.batch_size
    input_ids = torch.ones(batch, 128, dtype=torch.long)
    attention_mask = torch.ones(batch, 128, dtype=torch.long)
    images = torch.randn(batch, 3, 3, 224, 224)
    mel = torch.randn(batch, 64, args.mel_steps)

    with torch.no_grad():
        output = model(input_ids, attention_mask, images, mel, return_features=True)

    total, trainable = count_parameters(model)
    print("logits:", tuple(output["logits"].shape))
    print("text_feature:", tuple(output["text_feature"].shape))
    print("image_feature:", tuple(output["image_feature"].shape))
    print("audio_feature:", tuple(output["audio_feature"].shape))
    print("fusion_feature:", tuple(output["fusion_feature"].shape))
    print(f"parameters_total: {total / 1e6:.2f}M")
    print(f"parameters_trainable: {trainable / 1e6:.2f}M")


if __name__ == "__main__":
    main()
