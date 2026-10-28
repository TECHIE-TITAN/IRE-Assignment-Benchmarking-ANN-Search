"""Append one row per benchmark run to results/results.csv.

results/ lives next to code/ (not inside it), matching the submission layout:
    <root>/code/src/results_log.py  ->  <root>/results/
"""
import csv
import os
from datetime import datetime
from pathlib import Path

# run_all.sh --quick sets A3_RESULTS_DIR so a quick run never touches the real results
RESULTS_DIR = Path(os.environ.get("A3_RESULTS_DIR")
                   or Path(__file__).resolve().parent.parent.parent / "results")
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_CSV = RESULTS_DIR / "results.csv"

COLUMNS = [
    "timestamp",
    "experiment",    # e.g. part1, part2, part3, part4
    "dataset",       # sift, glove, or a synthetic dataset label
    "n_vectors",
    "split",         # tune or final
    "index",         # exact, hnsw, ivfflat, ivfpq
    "build_params",  # e.g. "M=16;efConstruction=200"
    "search_params", # e.g. "efSearch=64"
    "mode",          # single (one query at a time) or batch
    "recall_at_10",
    "qps",
    "build_time_s",
    "index_bytes",
    "bytes_per_vector",
    "notes",
]


def log_run(path=RESULTS_CSV, **fields):
    """Append a row. Unknown field names raise; missing ones are left blank."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown result columns: {sorted(unknown)}")
    row = {c: fields.get(c, "") for c in COLUMNS}
    if not row["timestamp"]:
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
