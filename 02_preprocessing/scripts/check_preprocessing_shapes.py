"""Check text, image, and audio preprocessing outputs for a few samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from miafnet.preprocessing import (  # noqa: E402
    FRAME_TAGS,
    MAX_TEXT_LENGTH,
    N_MELS,
    TEXT_BACKBONE,
    build_image_index,
    encode_text,
    load_log_mel,
    load_three_frames,
    read_transcript,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MIAF-Net preprocessing tensor shapes.")
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--transcript-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.split_csv).head(args.num_samples)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE, use_fast=True)
    image_index = build_image_index(args.image_dir, FRAME_TAGS)

    for _, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        text = read_transcript(args.transcript_dir, video_id)
        encoded = encode_text(tokenizer, text)
        images = load_three_frames(image_index, args.split, video_id)
        mel = load_log_mel(args.audio_dir, video_id)

        print(
            {
                "video_id": video_id,
                "input_ids": tuple(encoded["input_ids"].shape),
                "attention_mask": tuple(encoded["attention_mask"].shape),
                "expected_text_length": MAX_TEXT_LENGTH,
                "images": tuple(images.shape),
                "mel": tuple(mel.shape),
                "expected_n_mels": N_MELS,
            }
        )


if __name__ == "__main__":
    main()

