"""Build the filtered tri-modal CMU-MOSEI sample list used by MIAF-Net.

The script follows the manuscript setting:
1. Aggregate MTurk sentiment annotations from clip level to video level.
2. Convert the averaged sentiment score to three classes.
3. Keep only samples with aligned label, transcript, audio, and video resources.
4. Save a compact metadata CSV with video_id, sentiment_score, and label_3.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd


YT_ID_RE = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")
NEEDED_LABEL_COLUMNS = ["Input.VIDEO_ID", "Input.CLIP", "Answer.sentiment"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the filtered tri-modal MIAF-Net sample list.")
    parser.add_argument("--raw-root", required=True, help="Root folder of extracted CMU-MOSEI files.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--label-dir-name", default="Labels")
    parser.add_argument("--transcript-full-dir-name", default="transcript_full")
    parser.add_argument("--transcript-segmented-dir-name", default="transcript_segmented")
    parser.add_argument("--audio-dir-name", default="WAV_16000")
    parser.add_argument("--video-dir-name", default="videos_segmented")
    return parser.parse_args()


def to_3class(score: float) -> str:
    if score <= -0.5:
        return "neg"
    if score >= 0.5:
        return "pos"
    return "neu"


def extract_video_id(name_without_ext: str) -> str | None:
    name = str(name_without_ext).strip()
    if len(name) == 11 and re.fullmatch(r"[A-Za-z0-9_-]{11}", name):
        return name
    match = YT_ID_RE.search(name)
    return match.group(1) if match else None


def collect_transcript_ids(full_dir: Path, segmented_dir: Path) -> set[str]:
    ids: set[str] = set()
    for folder in [full_dir, segmented_dir]:
        if not folder.is_dir():
            continue
        for ext in ("*.txt", "*.transcript", "*.vtt", "*.srt"):
            for path in glob.glob(str(folder / "**" / ext), recursive=True):
                video_id = extract_video_id(Path(path).stem)
                if video_id:
                    ids.add(video_id)
    return ids


def collect_audio_ids(audio_dir: Path) -> set[str]:
    if not audio_dir.is_dir():
        return set()
    return {p.stem.strip() for p in audio_dir.glob("*.wav")}


def collect_video_ids(video_dir: Path) -> set[str]:
    if not video_dir.is_dir():
        return set()
    ids: set[str] = set()
    for path in video_dir.glob("*.mp4"):
        stem = path.stem
        video_id = None
        if "_" in stem:
            base, maybe_index = stem.rsplit("_", 1)
            if maybe_index.isdigit():
                video_id = extract_video_id(base)
        if video_id is None:
            video_id = extract_video_id(stem)
        if video_id:
            ids.add(video_id)
    return ids


def load_video_level_labels(label_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(label_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if all(col in df.columns for col in NEEDED_LABEL_COLUMNS):
            df = df[NEEDED_LABEL_COLUMNS].dropna().copy()
            df["Input.VIDEO_ID"] = df["Input.VIDEO_ID"].astype(str).str.strip()
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "No MTurk-format label CSV was found. Required columns: "
            + ", ".join(NEEDED_LABEL_COLUMNS)
        )

    all_df = pd.concat(frames, ignore_index=True)
    clip_df = (
        all_df.groupby(["Input.VIDEO_ID", "Input.CLIP"], as_index=False)["Answer.sentiment"]
        .mean()
        .rename(columns={"Answer.sentiment": "clip_sentiment"})
    )
    video_df = (
        clip_df.groupby("Input.VIDEO_ID", as_index=False)["clip_sentiment"]
        .mean()
        .rename(columns={"Input.VIDEO_ID": "video_id", "clip_sentiment": "sentiment_score"})
    )
    video_df["video_id"] = video_df["video_id"].astype(str).str.strip()
    video_df["label_3"] = video_df["sentiment_score"].apply(to_3class)
    return video_df[["video_id", "sentiment_score", "label_3"]]


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)

    label_dir = raw_root / args.label_dir_name
    full_dir = raw_root / args.transcript_full_dir_name
    segmented_dir = raw_root / args.transcript_segmented_dir_name
    audio_dir = raw_root / args.audio_dir_name
    video_dir = raw_root / args.video_dir_name

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    video_df = load_video_level_labels(label_dir)
    transcript_ids = collect_transcript_ids(full_dir, segmented_dir)
    audio_ids = collect_audio_ids(audio_dir)
    video_ids = collect_video_ids(video_dir)

    available_ids = transcript_ids & audio_ids
    if video_ids:
        available_ids &= video_ids

    out_df = video_df[video_df["video_id"].isin(available_ids)].copy()
    out_df = out_df.sort_values("video_id").reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Samples: {len(out_df)}")
    print(out_df["label_3"].value_counts().reindex(["pos", "neu", "neg"], fill_value=0))


if __name__ == "__main__":
    main()
