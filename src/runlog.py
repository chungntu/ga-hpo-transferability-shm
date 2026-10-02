"""
Result bookkeeping for the revision (HANDOFF_REVISION.md section 5.5).

Every training run appends one row to revision/results/runs.csv and dumps its
history / confusion matrix / per-class recall to revision/results/details/<run_id>.json.
Nothing is ever overwritten: a run_id collision raises.
"""

from __future__ import annotations

import csv
import json
import os
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repository root
RESULTS_DIR = os.path.join(ROOT, "results")
DETAIL_DIR = os.path.join(RESULTS_DIR, "details")
CSV_PATH = os.path.join(RESULTS_DIR, "runs.csv")

COLUMNS = [
    "run_id", "timestamp", "experiment",
    "dataset_train", "dataset_eval", "n_classes", "n_sensors", "window_length",
    "split_scheme", "split_digest",
    "config_source", "model", "scaler",
    "lr", "filters", "res_per_block", "n_blocks", "kernel_size",
    "epochs", "epochs_run", "batch_size", "head", "n_params",
    "seed",
    "val_acc", "val_acc_best", "test_acc", "test_macro_f1", "test_balanced_acc",
    "stopped_early", "train_time_sec", "gpu_name", "notes",
]


def new_run_id(prefix="run"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def log_run(row: dict, extras: dict | None = None):
    os.makedirs(DETAIL_DIR, exist_ok=True)
    run_id = row.get("run_id") or new_run_id()
    row = dict(row)
    row["run_id"] = run_id
    row.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown result columns: {sorted(unknown)}")

    detail_path = os.path.join(DETAIL_DIR, f"{run_id}.json")
    if os.path.exists(detail_path):
        raise FileExistsError(detail_path)

    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})

    if extras is not None:
        with open(detail_path, "w", encoding="utf-8") as fh:
            json.dump({"run_id": run_id, **row, **extras}, fh)
    return run_id


def read_runs():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
