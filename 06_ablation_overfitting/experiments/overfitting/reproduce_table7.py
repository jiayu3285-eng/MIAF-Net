"""Aggregate training/validation/test gap results for Table 7."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from utils.aggregate import read_json, scalar_or_mean, write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Table 7 overfitting summary.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-csv", default="table7_overfitting_summary.csv")
    parser.add_argument("--out-json", default="table7_overfitting_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        obj = read_json(path)
        train_f1 = obj["train_macro_f1"]
        val_f1 = obj["val_macro_f1"]
        test_f1 = obj.get("test_macro_f1", "")
        row = {
            "Setting": obj["setting"],
            "Best epoch": obj.get("best_epoch", ""),
            "Train ACC": scalar_or_mean(obj["train_acc"]),
            "Train Macro-F1": scalar_or_mean(train_f1),
            "Val ACC": scalar_or_mean(obj["val_acc"]),
            "Val Macro-F1": scalar_or_mean(val_f1),
            "Test ACC": scalar_or_mean(obj.get("test_acc", "")),
            "Test Macro-F1": scalar_or_mean(test_f1),
            "Train-Val Macro-F1 gap": obj.get("train_val_f1_gap", ""),
            "Train-Test Macro-F1 gap": obj.get("train_test_f1_gap", ""),
        }
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No overfitting result JSON files found under {args.results_dir}")
    write_csv(args.out_csv, rows)
    write_json(args.out_json, {"rows": rows})
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

