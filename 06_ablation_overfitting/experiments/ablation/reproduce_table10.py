"""Aggregate progressive replacement results for Table 10."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from models.progressive_variants import PROGRESSIVE_VARIANTS  # noqa: E402
from utils.aggregate import read_json, scalar_or_mean, write_csv, write_json  # noqa: E402


ORDER = ["miaf_base", "miaf_t", "miaf_ti", "miaf_tia", "miaf_net"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Table 10 progressive replacement summary.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-csv", default="table10_progressive_replacement.csv")
    parser.add_argument("--out-json", default="table10_progressive_replacement.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for key in ORDER:
        variant = PROGRESSIVE_VARIANTS[key]
        path = Path(args.results_dir) / f"{key}.json"
        if not path.exists():
            continue
        obj = read_json(path)
        rows.append(
            {
                "Variant": variant.name,
                "Active modalities": "+".join(
                    name
                    for name, active in [
                        ("T", variant.use_text_branch),
                        ("I", variant.use_image_branch),
                        ("A", variant.use_audio_branch),
                    ]
                    if active
                ),
                "Fusion": variant.fusion_type,
                "Val ACC": scalar_or_mean(obj["val_acc"]),
                "Val Macro-F1": scalar_or_mean(obj["val_macro_f1"]),
                "Test ACC": scalar_or_mean(obj.get("test_acc", "")),
                "Test Macro-F1": scalar_or_mean(obj.get("test_macro_f1", "")),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No progressive result JSON files found under {args.results_dir}")
    write_csv(args.out_csv, rows)
    write_json(args.out_json, {"rows": rows})
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

