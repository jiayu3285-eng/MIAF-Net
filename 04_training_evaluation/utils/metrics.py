"""Metrics used for MIAF-Net validation, testing, and Table 9 reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


LABELS = ["neg", "neu", "pos"]


@dataclass
class MetricResult:
    acc: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    neg_f1: float
    neu_f1: float
    pos_f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> MetricResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return MetricResult(
        acc=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        neg_f1=float(class_f1[0]),
        neu_f1=float(class_f1[1]),
        pos_f1=float(class_f1[2]),
    )


def detailed_report(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> str:
    return classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABELS, digits=4, zero_division=0)


def summarize_seed_results(rows: list[dict]) -> dict[str, dict[str, float]]:
    keys = ["test_acc", "test_macro_f1", "test_macro_precision", "test_macro_recall", "test_neg_f1", "test_neu_f1", "test_pos_f1"]
    summary = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

