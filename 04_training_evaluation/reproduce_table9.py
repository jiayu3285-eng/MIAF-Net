"""Aggregate 20-seed results for Table 9 reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.logger import read_json, write_json
from utils.metrics import format_mean_std, summarize_seed_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MIAF-Net seed results.")
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_files = sorted(Path(args.results_dir).glob("seed_*/result.json"))
    if not result_files:
        raise FileNotFoundError(f"No result.json files found under {args.results_dir}")

    rows = [read_json(path) for path in result_files]
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    summary = summarize_seed_results(rows)

    table9 = {
        "seeds": [int(row["seed"]) for row in rows],
        "num_seeds": len(rows),
        "Test_ACC": format_mean_std(summary["test_acc"]["mean"], summary["test_acc"]["std"]),
        "Test_Macro_F1": format_mean_std(summary["test_macro_f1"]["mean"], summary["test_macro_f1"]["std"]),
        "Test_Macro_Precision": format_mean_std(
            summary["test_macro_precision"]["mean"], summary["test_macro_precision"]["std"]
        ),
        "Test_Macro_Recall": format_mean_std(summary["test_macro_recall"]["mean"], summary["test_macro_recall"]["std"]),
        "summary": summary,
    }

    print(json.dumps(table9, indent=2))

    out_json = args.out_json or str(Path(args.results_dir) / "table9_summary.json")
    out_csv = args.out_csv or str(Path(args.results_dir) / "seed_results.csv")
    write_json(out_json, table9)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()

