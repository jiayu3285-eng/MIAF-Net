"""Aggregate fusion ablation results for Table 11."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from utils.aggregate import read_json, scalar_or_mean, write_csv, write_json  # noqa: E402


ORDER = [
    ("direct_concat", "Direct concatenation"),
    ("dcmif_only", "DCMIF only"),
    ("token_attention", "DCMIF + token attention"),
    ("cross_attention", "DCMIF + cross attention"),
    ("bilinear_gate", "DCMIF + bilinear gate"),
    ("dcmif_mhsa", "DCMIF + MHSA"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Table 11 fusion ablation summary.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-csv", default="table11_fusion_ablation.csv")
    parser.add_argument("--out-json", default="table11_fusion_ablation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for key, label in ORDER:
        path = Path(args.results_dir) / f"{key}.json"
        if not path.exists():
            continue
        obj = read_json(path)
        rows.append(
            {
                "Variant": label,
                "Val ACC": scalar_or_mean(obj["val_acc"]),
                "Val Macro-F1": scalar_or_mean(obj["val_macro_f1"]),
                "Test ACC": scalar_or_mean(obj.get("test_acc", "")),
                "Test Macro-F1": scalar_or_mean(obj.get("test_macro_f1", "")),
                "Component tested": obj.get("component_tested", ""),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No fusion ablation result JSON files found under {args.results_dir}")
    write_csv(args.out_csv, rows)
    write_json(args.out_json, {"rows": rows})
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

