"""Experiment log: append evaluation results to a CSV and optionally to W&B."""

import argparse
import csv
import json
import os
from datetime import datetime

LOG_COLS = [
    "timestamp", "arch", "input_mode", "seed", "split", "n_samples",
    "accuracy", "accuracy_ci",
    "precision", "precision_ci",
    "recall", "recall_ci",
    "f1_macro", "f1_macro_ci",
    "roc_auc", "roc_auc_ci",
    "pr_auc", "pr_auc_ci",
    "checkpoint",
]

DEFAULT_LOG = "results/experiment_log.csv"


def append_to_log(result: dict, log_path: str = DEFAULT_LOG):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    write_header = not os.path.exists(log_path)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "arch": result["arch"],
        "input_mode": result["input_mode"],
        "seed": result["seed"],
        "split": result["split"],
        "n_samples": result["n_samples"],
        "checkpoint": result["checkpoint"],
    }

    for metric_name in ["accuracy", "precision", "recall", "f1_macro", "roc_auc", "pr_auc"]:
        m = result["metrics"].get(metric_name, {})
        row[metric_name] = m.get("value", "")
        row[f"{metric_name}_ci"] = f"[{m.get('ci_lo', '')}, {m.get('ci_hi', '')}]"

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged to {log_path}")


def main():
    p = argparse.ArgumentParser(description="Append evaluation JSON to experiment log CSV")
    p.add_argument("json_file", help="Path to evaluation result JSON (from evaluate.py --out-json)")
    p.add_argument("--log", default=DEFAULT_LOG, help="Path to experiment log CSV")
    args = p.parse_args()

    with open(args.json_file) as f:
        result = json.load(f)

    append_to_log(result, args.log)


if __name__ == "__main__":
    main()
