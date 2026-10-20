"""Part 3: scaling on SIFT. Analysis of the 100K, 300K and 1M runs.

Cost = 1 / QPS (milliseconds per query, one thread, one query per call) of the
fastest measured configuration with recall@10 >= 0.95, on the final queries.
Runs at 100K and 300K are labelled experiment=part3_scaling in results.csv.
The 1M point reuses the Part 1 (exact) and Part 2 (HNSW, IVF-Flat) rows.

Also reports, per index, the growth of cost relative to 100K and a scaling
exponent: log(cost ratio) / log(n ratio). An exponent of 1 means cost grows in
direct proportion to the data, 0 means it does not grow.

Writes results/tables/scaling_table.md and results/figures/scaling_sift.{png,pdf}.

Usage: python src/scaling_experiment.py
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots import GRID, INK, MUTED, STYLE
from results_log import FIGURES_DIR, RESULTS_CSV
from tables import NAMES, TABLES_DIR, best_at, md_table

SIZES = [100_000, 300_000, 1_000_000]
INDEXES = ["exact", "hnsw", "ivfflat"]
TARGET = 0.95
BASE_N = SIZES[0]


def load_points(df):
    """Best (highest QPS, recall >= TARGET) row per index and size, or None."""
    sift = df[(df.dataset == "sift") & (df.split == "final") & (df["mode"] == "single")]
    sift = sift[sift.experiment.isin(["part1", "part2", "part3_scaling"])]
    points = {}
    for kind in INDEXES:
        for n in SIZES:
            g = sift[(sift["index"] == kind) & (sift.n_vectors == n)]
            points[(kind, n)] = best_at(g, TARGET) if not g.empty else None
    return points


def cost_ms(row):
    return None if row is None else 1000.0 / row.qps


def fmt(x, spec=".2f"):
    return "n/a" if x is None else format(x, spec)


def main():
    points = load_points(pd.read_csv(RESULTS_CSV))
    cost = {k: cost_ms(v) for k, v in points.items()}

    rows, detail = [], []
    for kind in INDEXES:
        c0 = cost[(kind, BASE_N)]
        line = [NAMES[kind]]
        for n in SIZES:
            line.append(fmt(cost[(kind, n)], ".3f"))
        for n in SIZES[1:]:
            cn = cost[(kind, n)]
            line.append("n/a" if c0 is None or cn is None else f"{cn / c0:.1f}x")
        cn = cost[(kind, SIZES[-1])]
        if c0 and cn:
            line.append(f"{math.log(cn / c0) / math.log(SIZES[-1] / BASE_N):.2f}")
        else:
            line.append("n/a")
        rows.append(line)
        for n in SIZES:
            p = points[(kind, n)]
            if p is not None:
                detail.append([NAMES[kind], f"{n:,}", f"{p.build_params}; {p.search_params}",
                               f"{p.recall_at_10:.4f}", f"{p.build_time_s:.1f}",
                               f"{p.index_bytes / 1e6:.1f}"])
            else:
                detail.append([NAMES[kind], f"{n:,}", "recall 0.95 not reached or missing", "", "", ""])

    speed = []
    for kind in ("hnsw", "ivfflat"):
        line = [NAMES[kind]]
        for n in SIZES:
            e, a = cost[("exact", n)], cost[(kind, n)]
            line.append("n/a" if e is None or a is None else f"{e / a:.0f}x")
        speed.append(line)

    header = ["Index"] + [f"ms/query @ {n // 1000}K" for n in SIZES] \
        + [f"growth {n // 1000}K vs {BASE_N // 1000}K" for n in SIZES[1:]] + ["scaling exponent"]
    text = (
        f"Cost at recall@10 >= {TARGET}, one thread, one query per call (SIFT, final queries)\n\n"
        + md_table(header, rows)
        + "\nSpeedup over exact search at the same size\n\n"
        + md_table(["Index"] + [f"{n // 1000}K" for n in SIZES], speed)
        + "\nConfiguration, recall, build time and size behind each point\n\n"
        + md_table(["Index", "n", "Config", "recall@10", "Build time (s)", "Size (MB)"], detail)
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "scaling_table.md").write_text(text)
    print(text)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for kind in INDEXES:
        xs = [n for n in SIZES if cost[(kind, n)] is not None]
        ys = [cost[(kind, n)] for n in xs]
        style = STYLE.get(kind, dict(color=MUTED, marker="*"))
        ax.plot(xs, ys, label=NAMES[kind], linewidth=2, markersize=7,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2, **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"{n // 1000}K" if n < 1_000_000 else "1M" for n in SIZES])
    ax.minorticks_off()
    ax.set_xlabel("number of vectors", color=INK)
    ax.set_ylabel(f"ms per query at recall@10 >= {TARGET} (1 thread, log scale)", color=INK)
    ax.set_title("SIFT: cost of reaching recall 0.95 as the data grows", fontsize=10, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"scaling_sift.{ext}", dpi=200)
    plt.close(fig)
    print("wrote scaling_sift.png/.pdf")


if __name__ == "__main__":
    main()
