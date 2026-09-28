"""Create the fixed stratified 8:1:1 train/validation/test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = ["video_id", "sentiment_score", "label_3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the MIAF-Net fixed 8:1:1 split.")
    parser.add_argument("--input-csv", required=True, help="Filtered multimodal sample list CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory for train.csv, val.csv, test.csv.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for stratified splitting.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def check_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Existing columns: {list(df.columns)}")


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    df = pd.read_csv(args.input_csv)
    check_columns(df)
    df = df[REQUIRED_COLUMNS].dropna().copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["label_3"] = df["label_3"].astype(str).str.strip()

    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=df["label_3"],
    )
    val_share_of_temp = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share_of_temp,
        random_state=args.seed,
        stratify=temp_df["label_3"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df = split_df.sort_values("video_id").reset_index(drop=True)
        split_df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(split_df)}")
        print(split_df["label_3"].value_counts().reindex(["pos", "neu", "neg"], fill_value=0))


if __name__ == "__main__":
    main()

