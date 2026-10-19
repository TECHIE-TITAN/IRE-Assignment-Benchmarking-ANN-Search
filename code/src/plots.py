"""Part 2 figures: recall@10 vs QPS, one per dataset.

Each curve is the best (Pareto) trade-off over all build settings of that
index, from the final split in results/results.csv. Exact search is shown as
a single point at recall 1.0. Recall is on the x axis and QPS (log scale) on
the y axis, the usual convention for ANN benchmarks.

Usage: python src/plots.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_log import FIGURES_DIR
from tables import DATASETS, KINDS, NAMES, load_results, pareto

# categorical slots 1-3 of the reference palette (light surface); marker
# shape is a second encoding so identity never relies on colour alone
STYLE = {
    "hnsw": dict(color="#2a78d6", marker="o"),
    "ivfflat": dict(color="#eb6834", marker="s"),
    "ivfpq": dict(color="#1baf7a", marker="^"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"


def plot_dataset(ds, part2, exact_qps):
    sub = part2[part2.dataset == ds]
    n = int(sub.n_vectors.iloc[0])
    fig, ax = plt.subplots(figsize=(6.5, 4.3))

    for kind in KINDS:
        g = sub[sub["index"] == kind]
        if g.empty:
            continue
        front = pareto(g)
        ax.plot(front.recall_at_10, front.qps, label=NAMES[kind], linewidth=2,
                markersize=6, markeredgecolor="#fcfcfb", markeredgewidth=1.2, **STYLE[kind])

    if ds in exact_qps:
        ax.axhline(exact_qps[ds], color=MUTED, linestyle="--", linewidth=1)
        ax.plot([1.0], [exact_qps[ds]], marker="*", color=MUTED, markersize=11,
                linestyle="none", label=f"{NAMES['exact']} ({exact_qps[ds]:.0f} QPS)")

    for t in (0.90, 0.95):
        ax.axvline(t, color=GRID, linewidth=1, zorder=0)

    ax.set_yscale("log")
    ax.set_xlim(right=1.01)
    ax.set_xlabel("recall@10", color=INK)
    ax.set_ylabel("queries per second (1 thread, log scale)", color=INK)
    ax.set_title(f"{DATASETS[ds]}: recall vs speed, best trade-off per index (n={n:,})",
                 fontsize=10, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"recall_vs_qps_{ds}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote recall_vs_qps_{ds}.png/.pdf")


def main():
    part2, exact_qps = load_results()
    for ds in DATASETS:
        if not part2[part2.dataset == ds].empty:
            plot_dataset(ds, part2, exact_qps)


if __name__ == "__main__":
    main()
