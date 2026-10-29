"""Compare two result folders produced by the same pipeline (run-to-run variation).

For every benchmark row that appears in both results.csv files (same experiment,
dataset, size, index, build settings, search setting and mode) it reports how far
recall, QPS and build time moved. It also compares the headline numbers: QPS at
recall 0.90 and 0.95, the Part 3 cost at 0.95, and the Part 1 exact timings.

Usage: python src/compare_runs.py ../results ../results_v2
Writes <second folder>/tables/run_comparison.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scaling_experiment import INDEXES as SCALE_INDEXES
from scaling_experiment import SIZES, cost_ms, label, load_points
from tables import DATASETS, KINDS, NAMES, TARGETS, best_at, load_results, md_table

KEY = ["experiment", "dataset", "n_vectors", "split", "index", "build_params",
       "search_params", "mode_key"]


def pct(a, b):
    return 100.0 * (b - a) / a


def with_mode_key(df):
    """Exact search logs two batch rows (1 thread, 8 threads) that differ only in notes."""
    df = df.copy()
    df["mode_key"] = df["mode"].astype(str)
    exact = df["index"] == "exact"
    df.loc[exact, "mode_key"] = df.loc[exact, "mode"].astype(str) + "|" + df.loc[exact, "notes"].astype(str)
    df[KEY] = df[KEY].fillna("")
    return df


def matched_rows(a, b):
    return with_mode_key(a).merge(with_mode_key(b), on=KEY, suffixes=("_a", "_b"))


def row_summary(a, b):
    both = matched_rows(a, b)
    rows = []
    for (exp, ds, kind), g in both.groupby(["experiment", "dataset", "index"]):
        n_a = len(a[(a.experiment == exp) & (a.dataset == ds) & (a["index"] == kind)])
        n_b = len(b[(b.experiment == exp) & (b.dataset == ds) & (b["index"] == kind)])
        d_recall = (g.recall_at_10_b - g.recall_at_10_a).abs()
        d_qps = pct(g.qps_a, g.qps_b)
        rows.append([exp, ds, NAMES.get(kind, kind), n_a, n_b, len(g),
                     f"{d_recall.max():.4f}", f"{d_recall.mean():.4f}",
                     f"{np.median(d_qps):+.1f}%", f"{np.median(d_qps.abs()):.1f}%",
                     f"{d_qps.abs().max():.1f}%"])
    return md_table(["experiment", "dataset", "index", "rows A", "rows B", "matched",
                     "max |d recall|", "mean |d recall|", "median d QPS", "median |d QPS|",
                     "max |d QPS|"], rows)


def headline(dir_a, dir_b):
    pa, ea = load_results(Path(dir_a) / "results.csv")
    pb, eb = load_results(Path(dir_b) / "results.csv")
    rows = []
    for ds, name in DATASETS.items():
        if ds in ea and ds in eb:
            rows.append([name, "Exact (1 thread, one query)", "-", f"{ea[ds]:.1f}",
                         f"{eb[ds]:.1f}", f"{pct(ea[ds], eb[ds]):+.1f}%"])
        for kind in KINDS:
            ga = pa[(pa.dataset == ds) & (pa["index"] == kind)]
            gb = pb[(pb.dataset == ds) & (pb["index"] == kind)]
            for t in TARGETS:
                ra = best_at(ga, t) if not ga.empty else None
                rb = best_at(gb, t) if not gb.empty else None
                if ra is None or rb is None:
                    rows.append([name, NAMES[kind], f"{t:.2f}",
                                 "not reached" if ra is None else f"{ra.qps:.0f}",
                                 "not reached" if rb is None else f"{rb.qps:.0f}", "-"])
                else:
                    rows.append([name, NAMES[kind], f"{t:.2f}", f"{ra.qps:.0f}",
                                 f"{rb.qps:.0f}", f"{pct(ra.qps, rb.qps):+.1f}%"])
    return md_table(["Dataset", "Index", "recall target", "QPS run A", "QPS run B",
                     "change"], rows)


def scaling(a, b):
    pa, pb = load_points(a), load_points(b)
    rows = []
    for kind in SCALE_INDEXES:
        for n in SIZES:
            ca, cb = cost_ms(pa[(kind, n)]), cost_ms(pb[(kind, n)])
            if ca is None or cb is None:
                continue
            rows.append([NAMES[kind], label(n), f"{ca:.3f}", f"{cb:.3f}", f"{pct(ca, cb):+.1f}%"])
    growth = []
    for kind in SCALE_INDEXES:
        vals = [(cost_ms(p[(kind, SIZES[-1])]), cost_ms(p[(kind, SIZES[0])])) for p in (pa, pb)]
        if all(x is not None and y is not None for x, y in vals):
            growth.append([NAMES[kind], f"{vals[0][0] / vals[0][1]:.1f}x",
                           f"{vals[1][0] / vals[1][1]:.1f}x"])
    return (md_table(["Index", "n", "ms/query A", "ms/query B", "change"], rows)
            + f"\nGrowth {label(SIZES[-1])} vs {label(SIZES[0])}\n\n"
            + md_table(["Index", "run A", "run B"], growth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", help="first results folder (for example ../results)")
    ap.add_argument("dir_b", help="second results folder (for example ../results_v2)")
    args = ap.parse_args()

    a = pd.read_csv(Path(args.dir_a) / "results.csv")
    b = pd.read_csv(Path(args.dir_b) / "results.csv")
    text = (
        f"Run A: {args.dir_a}   Run B: {args.dir_b}\n\n"
        "Every benchmark row present in both runs: how far recall, QPS and build time moved\n\n"
        + row_summary(a, b)
        + "\nHeadline numbers (QPS = one thread, one query per call)\n\n"
        + headline(args.dir_a, args.dir_b)
        + "\nPart 3: cost at recall >= 0.95 (ms per query)\n\n"
        + scaling(a, b)
    )
    out = Path(args.dir_b) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_comparison.md").write_text(text)
    print(text)
    print(f"wrote {out / 'run_comparison.md'}")


if __name__ == "__main__":
    main()
