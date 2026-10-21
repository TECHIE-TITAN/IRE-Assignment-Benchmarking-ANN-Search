"""Part 4: tables and the main figure for the drift experiment.

Reads experiment=part4_drift rows from results/results.csv and writes
  results/tables/drift_table.md
  results/figures/drift_main.{png,pdf}   (recall and relative speed vs shift)

Usage: python src/drift_analysis.py
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from plots import GRID, INK, MUTED, STYLE
from results_log import FIGURES_DIR, RESULTS_CSV
from tables import NAMES, TABLES_DIR, md_table

KINDS = ["hnsw", "ivfflat", "ivfpq"]
VARIANTS = ["stale", "refreshed"]


def load():
    df = pd.read_csv(RESULTS_CSV)
    df = df[df.experiment == "part4_drift"].copy()
    df["shift"] = df.build_params.str.extract(r"shift=([\d.]+)", expand=False).astype(float)
    df["variant"] = df.build_params.str.extract(r"variant=(\w+)", expand=False)
    for key in ("max_list", "mean_list", "scanned"):
        df[key] = df.notes.str.extract(rf"{key}=([\d.]+)", expand=False).astype(float)
    return df


def main():
    df = load()
    shifts = sorted(df["shift"].unique())
    row = lambda kind, variant, s: df[(df["index"] == kind) & (df.variant == variant)
                                      & (df["shift"] == s)].iloc[0]

    recall_rows, speed_rows, list_rows = [], [], []
    for kind in KINDS:
        for variant in VARIANTS:
            rows = [row(kind, variant, s) for s in shifts]
            ctrl = row(kind, "stale", 0.0)
            recall_rows.append([NAMES[kind], variant] + [f"{r.recall_at_10:.3f}" for r in rows])
            speed_rows.append([NAMES[kind], variant]
                              + [f"{r.qps:.0f} ({r.qps / ctrl.qps:.2f}x)" for r in rows])
            if kind != "hnsw":
                list_rows.append([NAMES[kind], variant] + [
                    f"{r.max_list:.0f} / {r.mean_list:.0f} / {r.scanned:.0f}" for r in rows])

    head = ["Index", "Variant"] + [f"shift {s:g}" for s in shifts]
    text = (
        "Recall@10 against exact search over all vectors (final queries)\n\n"
        + md_table(head, recall_rows)
        + "\nQPS, one thread, one query per call (in brackets: relative to the same index "
          "on the control, stale)\n\n"
        + md_table(head, speed_rows)
        + "\nIVF lists: largest list / mean list / mean vectors scanned per query\n\n"
        + md_table(head, list_rows)
        + "\nSearch settings: "
        + ", ".join(f"{NAMES[k]} {df[df['index'] == k].search_params.iloc[0]}" for k in KINDS)
        + "\n"
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "drift_table.md").write_text(text)
    print(text)

    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for kind in KINDS:
        ctrl_qps = row(kind, "stale", 0.0).qps
        for variant in VARIANTS:
            rows = [row(kind, variant, s) for s in shifts]
            style = dict(STYLE[kind], linestyle="-" if variant == "stale" else "--")
            label = f"{NAMES[kind]}, {variant}"
            ax_r.plot(shifts, [r.recall_at_10 for r in rows], label=label, linewidth=2,
                      markersize=6, markeredgecolor="#fcfcfb", markeredgewidth=1.2, **style)
            ax_s.plot(shifts, [r.qps / ctrl_qps for r in rows], label=label, linewidth=2,
                      markersize=6, markeredgecolor="#fcfcfb", markeredgewidth=1.2, **style)
    ax_r.set_ylabel("recall@10", color=INK)
    ax_s.set_ylabel("QPS relative to the index on the control", color=INK)
    ax_s.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax_s.yaxis.set_major_formatter(plain)
    ax_s.yaxis.set_minor_formatter(plain)
    for ax, title in ((ax_r, "Recall as the data drifts"), (ax_s, "Speed as the data drifts")):
        ax.set_xlabel("shift of added vectors and queries (cluster radii)", color=INK)
        ax.set_xticks(shifts)
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    ax_r.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"drift_main.{ext}", dpi=200)
    plt.close(fig)
    print("wrote drift_main.png/.pdf")


if __name__ == "__main__":
    main()
