"""Table aggregation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def fmt(values: list[float], digits: int = 3) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f}±{std:.{digits}f}"

