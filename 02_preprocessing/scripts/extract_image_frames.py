"""Extract deterministic image frames for the final tri-modal MIAF-Net experiments."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


SEGMENT_RE = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)\.mp4$", re.IGNORECASE)
RATIOS = {"t25": 0.25, "t50": 0.50, "t75": 0.75}
LABELS = ["neg", "neu", "pos"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image_dataset_8_1_1 from segmented CMU-MOSEI videos.")
    parser.add_argument("--video-dir", required=True, help="Folder containing segmented .mp4 files.")
    parser.add_argument("--split-dir", required=True, help="Folder containing train.csv, val.csv, test.csv.")
    parser.add_argument("--out-dir", required=True, help="Output image dataset folder.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--loglevel", default="error")
    return parser.parse_args()


def build_segment_index(video_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    index: dict[str, list[tuple[int, Path]]] = {}
    for path in video_dir.glob("*.mp4"):
        match = SEGMENT_RE.match(path.name)
        if not match:
            continue
        base = match.group("base")
        segment_idx = int(match.group("idx"))
        index.setdefault(base, []).append((segment_idx, path))

    for base in index:
        index[base].sort(key=lambda item: item[0])
    return index


def has_video_stream(ffprobe: str, video_path: Path) -> bool:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        return "video" in output
    except Exception:
        return False


def duration_seconds(ffprobe: str, video_path: Path) -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    return float(subprocess.check_output(cmd, text=True).strip())


def extract_frame(ffmpeg: str, loglevel: str, video_path: Path, second: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-i",
        str(video_path),
        "-ss",
        f"{second:.3f}",
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-an",
        "-sn",
        "-dn",
        "-y",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def pick_segment(args: argparse.Namespace, video_id: str, index: dict[str, list[tuple[int, Path]]]) -> Path | None:
    for _, path in index.get(video_id, []):
        if not has_video_stream(args.ffprobe, path):
            continue
        tmp = path.with_name(f"__tmp_check_{video_id}.jpg")
        ok = extract_frame(args.ffmpeg, args.loglevel, path, 0.0, tmp)
        if tmp.exists():
            tmp.unlink()
        if ok:
            return path
    return None


def check_split_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = {"video_id", "label_3"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")


def process_split(args: argparse.Namespace, split: str, index: dict[str, list[tuple[int, Path]]]) -> dict[str, int]:
    csv_path = Path(args.split_dir) / f"{split}.csv"
    df = pd.read_csv(csv_path)
    check_split_columns(df, csv_path)

    stats = {"total": len(df), "done": 0, "missing_segment": 0, "failed_extract": 0}
    for _, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        label = str(row["label_3"]).strip()
        if label not in LABELS:
            raise ValueError(f"Unknown label {label} in {csv_path}")

        segment = pick_segment(args, video_id, index)
        if segment is None:
            stats["missing_segment"] += 1
            continue

        try:
            duration = duration_seconds(args.ffprobe, segment)
        except Exception:
            stats["failed_extract"] += 1
            continue

        ok_all = True
        for tag, ratio in RATIOS.items():
            out_path = Path(args.out_dir) / split / label / f"{video_id}_{tag}.jpg"
            if out_path.exists():
                continue
            second = max(0.0, min(duration * ratio, max(duration - 0.05, 0.0)))
            ok = extract_frame(args.ffmpeg, args.loglevel, segment, second, out_path)
            if not ok:
                ok = extract_frame(args.ffmpeg, args.loglevel, segment, 0.0, out_path)
            ok_all = ok_all and ok

        if ok_all:
            stats["done"] += 1
        else:
            stats["failed_extract"] += 1

    return stats


def main() -> None:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    segment_index = build_segment_index(Path(args.video_dir))
    print(f"Indexed base videos: {len(segment_index)}")

    for split in ["train", "val", "test"]:
        stats = process_split(args, split, segment_index)
        print(f"{split}: {stats}")


if __name__ == "__main__":
    main()

