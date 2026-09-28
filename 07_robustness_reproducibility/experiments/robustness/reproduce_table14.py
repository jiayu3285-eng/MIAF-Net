"""Run robustness evaluation and reproduce Table 14."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "04_training_evaluation"))
sys.path.insert(0, str(RELEASE_ROOT / "03_model"))

from miafnet import MIAFNet, MIAFNetConfig  # noqa: E402
from train import TEXT_BACKBONE, make_loader  # noqa: E402
from perturbations import MODE_LABELS, ROBUSTNESS_MODES, apply_perturbation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIAF-Net robustness and reproduce Table 14.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-dir", default=str(RELEASE_ROOT / "01_data_split" / "data" / "splits"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-csv", default="table14_reproduced.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--modes", nargs="*", default=ROBUSTNESS_MODES)
    return parser.parse_args()


@torch.no_grad()
def evaluate_mode(model, loader, criterion, mode: str, device: torch.device, amp: bool = True) -> dict:
    model.eval()
    total_loss = 0.0
    preds, labels = [], []

    for batch in loader:
        input_ids, attention_mask, images, mel, y = apply_perturbation(batch, mode, device)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(input_ids, attention_mask, images, mel)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    return {
        "mode": mode,
        "setting": MODE_LABELS.get(mode, mode),
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE, use_fast=True)
    loader = make_loader(args, "test", tokenizer, shuffle=False)

    model = MIAFNet(MIAFNetConfig()).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()

    rows = [evaluate_mode(model, loader, criterion, mode, device, amp=args.amp) for mode in args.modes]
    clean = next((row for row in rows if row["mode"] == "clean"), rows[0])
    for row in rows:
        row["acc_drop"] = clean["acc"] - row["acc"]
        row["macro_f1_drop"] = clean["macro_f1"] - row["macro_f1"]
        for key in ["loss", "acc", "macro_f1", "acc_drop", "macro_f1_drop"]:
            row[key] = f"{row[key]:.4f}"

    write_csv(args.out_csv, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

