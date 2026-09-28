"""Verify the released MIAF-Net fixed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


LABEL_ORDER = ["pos", "neu", "neg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify MIAF-Net split files.")
    parser.add_argument("--sample-list", required=True, help="Filtered multimodal sample list CSV.")
    parser.add_argument("--split-dir", required=True, help="Directory containing train.csv, val.csv, test.csv.")
    parser.add_argument("--write-json", default=None, help="Optional path for statistics JSON.")
    return parser.parse_args()


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label_3"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in LABEL_ORDER}


def main() -> None:
    args = parse_args()
    sample_list = pd.read_csv(args.sample_list)
    split_dir = Path(args.split_dir)
    splits = {name: pd.read_csv(split_dir / f"{name}.csv") for name in ["train", "val", "test"]}

    sample_ids = set(sample_list["video_id"].astype(str))
    split_ids = {name: set(df["video_id"].astype(str)) for name, df in splits.items()}
    merged_ids = set().union(*split_ids.values())

    overlaps = {
        "train_val": len(split_ids["train"] & split_ids["val"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "val_test": len(split_ids["val"] & split_ids["test"]),
    }

    missing_from_splits = sorted(sample_ids - merged_ids)
    unknown_in_splits = sorted(merged_ids - sample_ids)

    stats = {
        "sample_list": {
            "samples": int(len(sample_list)),
            "label_distribution": label_counts(sample_list),
        },
        "splits": {
            name: {
                "samples": int(len(df)),
                "label_distribution": label_counts(df),
            }
            for name, df in splits.items()
        },
        "overlaps": overlaps,
        "missing_from_splits": len(missing_from_splits),
        "unknown_in_splits": len(unknown_in_splits),
    }

    print(json.dumps(stats, indent=2))

    if any(overlaps.values()):
        raise RuntimeError(f"Split overlap detected: {overlaps}")
    if missing_from_splits or unknown_in_splits:
        raise RuntimeError(
            f"Split/sample-list mismatch. Missing={len(missing_from_splits)}, unknown={len(unknown_in_splits)}"
        )

    if args.write_json:
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Statistics saved to: {out_path}")


if __name__ == "__main__":
    main()

