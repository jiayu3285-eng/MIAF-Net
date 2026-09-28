"""Aggregate regularization and lightweight-control results for Table 8."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from utils.aggregate import read_json, scalar_or_mean, write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Table 8 regularization-control summary.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-csv", default="table8_regularization_control.csv")
    parser.add_argument("--out-json", default="table8_regularization_control.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        obj = read_json(path)
        rows.append(
            {
                "Setting": obj["setting"],
                "Frozen encoders": obj.get("frozen_encoders", ""),
                "Classifier": obj.get("classifier", ""),
                "Dropout": obj.get("dropout", ""),
                "Weight decay": obj.get("weight_decay", ""),
                "Hidden size": obj.get("hidden_size", ""),
                "Best epoch": obj.get("best_epoch", ""),
                "Val ACC": scalar_or_mean(obj["val_acc"]),
                "Val Macro-F1": scalar_or_mean(obj["val_macro_f1"]),
                "Test ACC": scalar_or_mean(obj.get("test_acc", "")),
                "Test Macro-F1": scalar_or_mean(obj.get("test_macro_f1", "")),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No regularization result JSON files found under {args.results_dir}")
    write_csv(args.out_csv, rows)
    write_json(args.out_json, {"rows": rows})
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

