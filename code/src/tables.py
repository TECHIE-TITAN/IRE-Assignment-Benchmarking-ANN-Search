"""Part 2 tables, computed from results/results.csv (final split only).

  qps_table.md   : QPS each index reaches at recall 0.90 and 0.95
  build_table.md : build time, index size and bytes/vector at recall 0.95

"At recall T" means: among all measured points (all build settings, all sweep
values) with recall@10 >= T, the one with the highest QPS. No interpolation.
If an index never reaches T, the table says so and, for the build table,
reports the configuration with the highest recall instead.

Usage: python src/tables.py
"""
import pandas as pd
from results_log import RESULTS_CSV, RESULTS_DIR

TABLES_DIR = RESULTS_DIR / "tables"
KINDS = ["hnsw", "ivfflat", "ivfpq"]
NAMES = {"hnsw": "HNSW", "ivfflat": "IVF-Flat", "ivfpq": "IVF-PQ", "exact": "Exact (Flat)"}
DATASETS = {"sift": "SIFT-1M", "glove": "GloVe-100"}
TARGETS = (0.90, 0.95)


def load_results(path=RESULTS_CSV):
    """Return (part2 final rows, {dataset: exact single-query QPS})."""
    df = pd.read_csv(path)
    part2 = df[(df.experiment == "part2") & (df.split == "final") & (df["mode"] == "single")]
    exact = df[(df.experiment == "part1") & (df["index"] == "exact") & (df["mode"] == "single")]
    exact_qps = {}
    for name, g in exact.groupby("dataset"):
        exact_qps[name] = float(g.sort_values("n_vectors").iloc[-1].qps)
    return part2, exact_qps


def pareto(df):
    """Points not beaten on both recall and QPS, sorted by recall ascending."""
    df = df.sort_values(["recall_at_10", "qps"], ascending=[False, False])
    keep, best = [], -1.0
    for idx, row in df.iterrows():
        if row.qps > best:
            keep.append(idx)
            best = row.qps
    return df.loc[keep].sort_values("recall_at_10")


def best_at(df, target):
    """Highest-QPS row with recall >= target, or None."""
    ok = df[df.recall_at_10 >= target]
    return None if ok.empty else ok.loc[ok.qps.idxmax()]


def config(row):
    return f"{row.build_params}; {row.search_params}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def qps_table(part2, exact_qps):
    rows = []
    for ds, label in DATASETS.items():
        sub = part2[part2.dataset == ds]
        if sub.empty:
            continue
        if ds in exact_qps:
            rows.append([label, NAMES["exact"], f"{exact_qps[ds]:.0f} (recall 1.0)",
                         f"{exact_qps[ds]:.0f} (recall 1.0)", "IndexFlatL2"])
        for kind in KINDS:
            g = sub[sub["index"] == kind]
            if g.empty:
                continue
            cells = []
            for t in TARGETS:
                b = best_at(g, t)
                cells.append(f"{b.qps:.0f}" if b is not None else f"not reached (max {g.recall_at_10.max():.3f})")
            b95 = best_at(g, 0.95)
            rows.append([label, NAMES[kind], cells[0], cells[1], config(b95) if b95 is not None else "-"])
    return md_table(["Dataset", "Index", "QPS @ recall 0.90", "QPS @ recall 0.95",
                     "Config at 0.95"], rows)


def build_table(part2):
    rows = []
    for ds, label in DATASETS.items():
        sub = part2[part2.dataset == ds]
        for kind in KINDS:
            g = sub[sub["index"] == kind]
            if g.empty:
                continue
            b = best_at(g, 0.95)
            note = ""
            if b is None:
                b = g.loc[g.recall_at_10.idxmax()]
                note = f" (0.95 not reached; best recall {b.recall_at_10:.3f})"
            rows.append([label, NAMES[kind], config(b) + note, f"{b.build_time_s:.1f}",
                         f"{b.index_bytes / 1e6:.1f}", f"{b.bytes_per_vector:.1f}"])
    return md_table(["Dataset", "Index", "Config", "Build time (s)", "Index size (MB)",
                     "Bytes / vector"], rows)


def main():
    part2, exact_qps = load_results()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for fname, text in [("qps_table.md", qps_table(part2, exact_qps)),
                        ("build_table.md", build_table(part2))]:
        (TABLES_DIR / fname).write_text(text)
        print(f"--- {fname} ---\n{text}")


if __name__ == "__main__":
    main()
