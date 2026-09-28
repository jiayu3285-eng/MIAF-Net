"""Unified baseline training and evaluation entry point.

This script trains one baseline on cached aligned modality features and reports
validation/test ACC and Macro-F1 using the same split protocol as Table 12.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from baselines import BASELINE_REGISTRY  # noqa: E402


def load_yaml_like(path: str | Path) -> dict:
    cfg = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        cfg[key.strip()] = parsed
    return cfg


def load_feature_split(feature_dir: str | Path, split: str) -> TensorDataset:
    obj = torch.load(Path(feature_dir) / f"{split}.pt", map_location="cpu")
    text = obj.get("text", obj.get("t"))
    image = obj.get("image", obj.get("i"))
    audio = obj.get("audio", obj.get("a"))
    label = obj.get("label", obj.get("y"))
    if any(x is None for x in [text, image, audio, label]):
        raise KeyError("Feature cache must contain text/image/audio/label or t/i/a/y keys.")
    return TensorDataset(text.float(), image.float(), audio.float(), label.long())


def compute(loader, model, criterion, device, optimizer=None, amp=True):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    preds, gts = [], []
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for text, image, audio, y in loader:
            text = text.to(device)
            image = image.to(device)
            audio = audio.to(device)
            y = y.to(device)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                out = model(text, image, audio)
                logits = out["logits"] if isinstance(out, dict) else out
                loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            gts.extend(y.detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(gts, preds),
        "macro_f1": f1_score(gts, preds, average="macro", zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate one baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-dir", required=True, help="Directory with train.pt, val.pt, test.pt.")
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_like(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = str(cfg["model_key"])
    model_cls = BASELINE_REGISTRY[model_name]
    model = model_cls(
        text_dim=int(cfg["text_dim"]),
        image_dim=int(cfg["image_dim"]),
        audio_dim=int(cfg["audio_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_classes=3,
        dropout=float(cfg["dropout"]),
    ).to(device)

    train_ds = load_feature_split(args.feature_dir, "train")
    val_ds = load_feature_split(args.feature_dir, "val")
    test_ds = load_feature_split(args.feature_dir, "test")
    batch_size = int(cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    start = time.perf_counter()
    best_state = None
    best_val_f1 = -1.0
    best_val_acc = -1.0
    epochs = int(cfg["epochs"])
    for _ in range(epochs):
        compute(train_loader, model, criterion, device, optimizer=optimizer, amp=bool(cfg["amp"]))
        val = compute(val_loader, model, criterion, device, amp=bool(cfg["amp"]))
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_val_acc = val["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = compute(test_loader, model, criterion, device, amp=bool(cfg["amp"]))
    minutes = (time.perf_counter() - start) / 60.0

    result = {
        "model": cfg["model_name"],
        "val_acc": [best_val_acc],
        "val_macro_f1": [best_val_f1],
        "test_acc": [test["acc"]],
        "test_macro_f1": [test["macro_f1"]],
        "params_m": float(cfg["table_params_m"]),
        "training_time_min_per_run": round(minutes, 2),
        "table_training_time_min_per_run": float(cfg["table_training_time_min_per_run"]),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()

