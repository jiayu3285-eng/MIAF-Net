"""Reproduce the revised Table 12 from baseline result JSON files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from utils.table import fmt, load_json, mean_std, write_json


MODEL_ORDER = ["TFN", "LMF", "MulT", "Deep-HOSeq", "BBFN", "MLMF", "ALMT", "KuDA", "MIAF-Net"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate baseline comparison results for Table 12.")
    parser.add_argument("--results-dir", required=True, help="Directory containing one JSON result file per model.")
    parser.add_argument("--out-csv", default="table12_reproduced.csv")
    parser.add_argument("--out-json", default="table12_reproduced.json")
    return parser.parse_args()


def model_sort_key(row: dict) -> int:
    try:
        return MODEL_ORDER.index(row["Model"])
    except ValueError:
        return len(MODEL_ORDER)


def format_minutes(value: int | float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def build_row(obj: dict) -> dict:
    model = obj["model"]
    val_acc_mean, val_acc_std = mean_std(obj["val_acc"])
    val_f1_mean, val_f1_std = mean_std(obj["val_macro_f1"])
    test_acc_mean, test_acc_std = mean_std(obj["test_acc"])
    test_f1_mean, test_f1_std = mean_std(obj["test_macro_f1"])

    return {
        "Model": model,
        "Val ACC": fmt(obj["val_acc"]),
        "Val Macro-F1": fmt(obj["val_macro_f1"]),
        "Test ACC": fmt(obj["test_acc"]),
        "Test Macro-F1": fmt(obj["test_macro_f1"]),
        "Params (M)": obj["params_m"],
        "Training time": (
            f'{format_minutes(obj.get("table_training_time_min_per_run", obj["training_time_min_per_run"]))}min/run'
        ),
        "_numeric": {
            "val_acc_mean": val_acc_mean,
            "val_acc_std": val_acc_std,
            "val_macro_f1_mean": val_f1_mean,
            "val_macro_f1_std": val_f1_std,
            "test_acc_mean": test_acc_mean,
            "test_acc_std": test_acc_std,
            "test_macro_f1_mean": test_f1_mean,
            "test_macro_f1_std": test_f1_std,
        },
    }


def main() -> None:
    args = parse_args()
    result_files = sorted(Path(args.results_dir).glob("*.json"))
    if not result_files:
        raise FileNotFoundError(f"No JSON result files found under {args.results_dir}")

    rows = [build_row(load_json(path)) for path in result_files]
    rows.sort(key=model_sort_key)

    output_rows = [{k: v for k, v in row.items() if k != "_numeric"} for row in rows]
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    write_json(args.out_json, {"rows": rows})
    for row in output_rows:
        print(row)


if __name__ == "__main__":
    main()
