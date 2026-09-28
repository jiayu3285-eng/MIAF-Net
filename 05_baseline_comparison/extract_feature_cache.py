"""Extract aligned modality features for baseline comparison.

The baseline models in this group consume fixed modality vectors:

- text:  [B, D_t]
- image: [B, D_i]
- audio: [B, D_a]

This script reuses the released MIAF-Net encoders to export those vectors for
the same train/validation/test split used in Table 12. A trained checkpoint can
be provided when available; otherwise the script only serves as an interface
check and should not be used for manuscript numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


CURRENT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(RELEASE_ROOT / "04_training_evaluation"))
sys.path.insert(0, str(RELEASE_ROOT / "03_model"))

from miafnet import MIAFNet, MIAFNetConfig  # noqa: E402
from train import TEXT_BACKBONE, make_loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract aligned modality feature caches for baselines.")
    parser.add_argument("--checkpoint", default=None, help="Optional MIAF-Net checkpoint.")
    parser.add_argument("--split-dir", default=str(RELEASE_ROOT / "01_data_split" / "data" / "splits"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def extract_split(model: MIAFNet, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    text_features, image_features, audio_features, labels = [], [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        images = batch["images"].to(device, non_blocking=True)
        mel = batch["mel"].to(device, non_blocking=True)

        text, image, audio = model.encode_modalities(input_ids, attention_mask, images, mel)
        text_features.append(text.detach().cpu())
        image_features.append(image.detach().cpu())
        audio_features.append(audio.detach().cpu())
        labels.append(batch["label"].detach().cpu())

    return {
        "text": torch.cat(text_features, dim=0),
        "image": torch.cat(image_features, dim=0),
        "audio": torch.cat(audio_features, dim=0),
        "label": torch.cat(labels, dim=0),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE, use_fast=True)

    model = MIAFNet(MIAFNetConfig()).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        loader = make_loader(args, split, tokenizer, shuffle=False)
        obj = extract_split(model, loader, device)
        torch.save(obj, out_dir / f"{split}.pt")
        print(f"Saved {split}: {out_dir / f'{split}.pt'}")


if __name__ == "__main__":
    main()

