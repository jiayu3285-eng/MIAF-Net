"""Train MIAF-Net for one random seed.

The training schedule is fixed to 100 epochs. The checkpoint is selected by
the best validation Macro-F1. Training is not terminated early.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(RELEASE_ROOT / "03_model"))

from miafnet import MIAFNet, MIAFNetConfig  # noqa: E402
from utils.logger import append_csv_row, write_json  # noqa: E402
from utils.metrics import compute_metrics, detailed_report  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def load_preprocessing_module():
    path = RELEASE_ROOT / "02_preprocessing" / "miafnet" / "preprocessing.py"
    spec = importlib.util.spec_from_file_location("miafnet_release_preprocessing", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preprocessing module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preproc = load_preprocessing_module()
FRAME_TAGS = preproc.FRAME_TAGS
LABEL2ID = preproc.LABEL2ID
TEXT_BACKBONE = preproc.TEXT_BACKBONE
build_image_index = preproc.build_image_index
encode_text = preproc.encode_text
load_log_mel = preproc.load_log_mel
load_three_frames = preproc.load_three_frames
normalize_label = preproc.normalize_label
read_transcript = preproc.read_transcript


class TriModalDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str, tokenizer, raw_root: str | Path, image_dir: str | Path):
        self.df = pd.read_csv(csv_path)
        self.df["video_id"] = self.df["video_id"].astype(str).str.strip()
        self.df["label"] = self.df["label_3"].apply(normalize_label)
        self.split = split
        self.tokenizer = tokenizer
        raw_root = Path(raw_root)
        self.transcript_dir = raw_root / "transcript_full"
        self.segmented_transcript_dir = raw_root / "transcript_segmented"
        self.audio_dir = raw_root / "WAV_16000"
        self.image_index = build_image_index(image_dir, FRAME_TAGS)

    def __len__(self) -> int:
        return len(self.df)

    def _read_text(self, video_id: str) -> str:
        text = read_transcript(self.transcript_dir, video_id)
        if not text:
            text = read_transcript(self.segmented_transcript_dir, video_id)
        return text

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        text = self._read_text(video_id)
        encoded = encode_text(self.tokenizer, text)
        images = load_three_frames(self.image_index, self.split, video_id)
        mel = load_log_mel(self.audio_dir, video_id)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "images": images,
            "mel": mel,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MIAF-Net for one seed.")
    parser.add_argument("--split-dir", default=str(RELEASE_ROOT / "01_data_split" / "data" / "splits"))
    parser.add_argument("--raw-root", required=True, help="raw_extracted folder root.")
    parser.add_argument("--image-dir", required=True, help="Prepared image_dataset_8_1_1 folder.")
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def make_loader(args: argparse.Namespace, split: str, tokenizer, shuffle: bool) -> DataLoader:
    ds = TriModalDataset(
        csv_path=Path(args.split_dir) / f"{split}.csv",
        split=split,
        tokenizer=tokenizer,
        raw_root=args.raw_root,
        image_dir=args.image_dir,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp: bool = True):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    y_true, y_pred = [], []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            images = batch["images"].to(device, non_blocking=True)
            mel = batch["mel"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(input_ids, attention_mask, images, mel)
                loss = criterion(logits, labels)

            if train_mode:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    return total_loss / len(loader.dataset), metrics, y_true, y_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dir = Path(args.output_dir) / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(seed_dir / "config.json", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE, use_fast=True)
    train_loader = make_loader(args, "train", tokenizer, shuffle=True)
    val_loader = make_loader(args, "val", tokenizer, shuffle=False)
    test_loader = make_loader(args, "test", tokenizer, shuffle=False)

    model = MIAFNet(MIAFNetConfig()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_val_macro_f1 = -1.0
    best_path = seed_dir / "best_val_macro_f1.pt"
    log_path = seed_dir / "train_log.csv"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, amp=args.amp
        )
        val_loss, val_metrics, _, _ = run_epoch(model, val_loader, criterion, device, amp=args.amp)

        is_best = val_metrics.macro_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_metrics.macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_state": model.state_dict(),
                    "val_macro_f1": val_metrics.macro_f1,
                    "val_acc": val_metrics.acc,
                    "config": vars(args),
                    "label2id": LABEL2ID,
                },
                best_path,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics.acc,
            "train_macro_f1": train_metrics.macro_f1,
            "val_loss": val_loss,
            "val_acc": val_metrics.acc,
            "val_macro_f1": val_metrics.macro_f1,
            "is_best_val_macro_f1": int(is_best),
        }
        append_csv_row(log_path, row)
        print(row)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_metrics, y_true, y_pred = run_epoch(model, test_loader, criterion, device, amp=args.amp)

    final = {
        "seed": args.seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_val_macro_f1": float(checkpoint["val_macro_f1"]),
        "test_loss": float(test_loss),
        "test_acc": test_metrics.acc,
        "test_macro_f1": test_metrics.macro_f1,
        "test_macro_precision": test_metrics.macro_precision,
        "test_macro_recall": test_metrics.macro_recall,
        "test_neg_f1": test_metrics.neg_f1,
        "test_neu_f1": test_metrics.neu_f1,
        "test_pos_f1": test_metrics.pos_f1,
    }
    write_json(seed_dir / "result.json", final)
    (seed_dir / "test_classification_report.txt").write_text(detailed_report(y_true, y_pred), encoding="utf-8")
    print(final)


if __name__ == "__main__":
    main()
