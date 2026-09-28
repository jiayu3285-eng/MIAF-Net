"""Evaluate a saved MIAF-Net checkpoint on validation or independent test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(RELEASE_ROOT / "03_model"))

from miafnet import MIAFNet, MIAFNetConfig  # noqa: E402
from train import TEXT_BACKBONE, make_loader, run_epoch  # noqa: E402
from utils.logger import write_json  # noqa: E402
from utils.metrics import detailed_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIAF-Net checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--split-dir", default=str(RELEASE_ROOT / "01_data_split" / "data" / "splits"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE, use_fast=True)
    loader = make_loader(args, args.split, tokenizer, shuffle=False)

    model = MIAFNet(MIAFNetConfig()).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()

    loss, metrics, y_true, y_pred = run_epoch(model, loader, criterion, device, amp=args.amp)
    result = {"split": args.split, "loss": float(loss), **metrics.to_dict()}
    print(result)
    print(detailed_report(y_true, y_pred))

    if args.output_json:
        write_json(args.output_json, result)


if __name__ == "__main__":
    main()
