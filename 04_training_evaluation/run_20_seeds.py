"""Run the manuscript 20-seed MIAF-Net training protocol."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIAF-Net across 20 seeds.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_py = Path(__file__).resolve().parent / "train.py"
    for seed in args.seeds:
        cmd = [
            args.python,
            str(train_py),
            "--raw-root",
            args.raw_root,
            "--image-dir",
            args.image_dir,
            "--output-dir",
            args.output_dir,
            "--seed",
            str(seed),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

