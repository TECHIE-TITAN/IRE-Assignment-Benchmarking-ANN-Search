"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ.

For each dataset and index kind, build every configuration in the grid, then
sweep the search-time parameter (efSearch / nprobe). For each sweep point:
  recall@10 : all queries of the split, batch search (results do not depend on threads)
  QPS       : one query per call, single thread, median of --repeats passes

--split tune  uses the first 5,000 queries and logs to data/tuning.csv
              (used to check that grids reach recall ~0.95; not reported).
--split final uses the last 5,000 queries and logs to ../results/results.csv.

Usage:
  python src/benchmark.py --split tune --n 100000 --timing-queries 500 --repeats 1
  python src/benchmark.py --split final
"""
import argparse
import math

import faiss
from data import DATA_DIR, FINAL, TUNE, load_dataset, split_queries
from groundtruth import kth_dist_sq, load_or_compute_gt
from indexes import KINDS, build_index, params_str, set_search_param
from recall import recall_at_k
from results_log import RESULTS_CSV, log_run
from timing import set_single_thread, time_single_queries

K = 10

EF_SWEEP = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
NPROBE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# stop a sweep once recall is high enough, or after two sweep points in a row
# that improve recall by less than MIN_GAIN (the index has plateaued)
STOP_RECALL = 0.99
MIN_GAIN = 0.002

PQ_M = {128: (16, 32, 64), 100: (10, 20, 50)}  # sub-vector counts per dimension


def build_grid(kind, n, d, level):
    """Build-parameter settings to try. level: 'full' or 'small'."""
    full = level == "full"
    base = 2 ** round(math.log2(math.sqrt(n)))  # ~sqrt(n) rounded to a power of 2
    if kind == "hnsw":
        return [{"M": m, "efConstruction": 200} for m in ([8, 16, 32] if full else [16])]
    if kind == "ivfflat":
        return [{"nlist": nl} for nl in ([base, 2 * base, 4 * base] if full else [base])]
    if kind == "ivfpq":
        nlists = [base, 4 * base] if full else [base]
        ms = PQ_M[d] if full else PQ_M[d][:1]
        return [{"nlist": nl, "m": m} for nl in nlists for m in ms]
    raise ValueError(kind)


def sweep_values(built):
    if built.kind == "hnsw":
        return EF_SWEEP
    return [v for v in NPROBE_SWEEP if v <= built.params["nlist"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sift", "glove"])
    ap.add_argument("--indexes", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--split", choices=["tune", "final"], default="final")
    ap.add_argument("--n", type=int, default=None, help="use first N base vectors")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timing-queries", type=int, default=None,
                    help="time only the first T queries of the split (recall still uses all)")
    ap.add_argument("--grid", choices=["full", "small"], default="full")
    ap.add_argument("--experiment", default=None,
                    help="label for the experiment column (default: part2, or part2_tune)")
    args = ap.parse_args()

    max_threads = faiss.omp_get_max_threads()
    sel = TUNE if args.split == "tune" else FINAL
    if args.split == "tune":
        out_csv, experiment = DATA_DIR / "tuning.csv", "part2_tune"
    else:
        out_csv, experiment = RESULTS_CSV, "part2"
    if args.experiment:
        experiment = args.experiment

    for name in args.datasets:
        d = load_dataset(name, n=args.n)
        train, queries = d["train"], split_queries(d["test"])[0 if args.split == "tune" else 1]
        n, dim = train.shape
        D_all, _ = load_or_compute_gt(name, train, d["test"], k=100)
        kth = kth_dist_sq(D_all, K)[sel]
        timed = queries if args.timing_queries is None else queries[:args.timing_queries]
        print(f"\n##### {name}: n={n} d={dim} split={args.split} "
              f"queries={len(queries)} timed={len(timed)}", flush=True)

        for kind in args.indexes:
            for bp in build_grid(kind, n, dim, args.grid):
                faiss.omp_set_num_threads(max_threads)
                built = build_index(kind, train, **bp)
                print(f"\n[{kind}] {params_str(built.params)}  build={built.build_time_s:.1f}s "
                      f"size={built.size_bytes / 1e6:.1f}MB", flush=True)

                prev, stall, best = None, 0, 0.0
                for v in sweep_values(built):
                    set_search_param(built, v)
                    faiss.omp_set_num_threads(max_threads)
                    ids = built.index.search(queries, K)[1]
                    rec = recall_at_k(ids, kth, train, queries, k=K)

                    set_single_thread()
                    res = time_single_queries(
                        lambda q: built.index.search(q, K), timed, repeats=args.repeats)

                    log_run(
                        path=out_csv, experiment=experiment, dataset=name, n_vectors=n,
                        split=args.split, index=kind, build_params=params_str(built.params),
                        search_params=f"{built.search_param_name}={v}", mode="single",
                        recall_at_10=round(rec, 5), qps=round(res["qps"], 2),
                        build_time_s=round(built.build_time_s, 3),
                        index_bytes=built.size_bytes,
                        bytes_per_vector=round(built.size_bytes / n, 2),
                        notes=f"repeats={args.repeats};timed_queries={len(timed)}",
                    )
                    print(f"  {built.search_param_name}={v:<5} recall={rec:.4f} "
                          f"qps={res['qps']:9.1f}", flush=True)

                    best = max(best, rec)
                    if rec >= STOP_RECALL:
                        break
                    stall = stall + 1 if prev is not None and rec - prev < MIN_GAIN else 0
                    if stall >= 2:
                        print("  (recall plateaued, stopping sweep)", flush=True)
                        break
                    prev = rec
                print(f"  best recall for this build: {best:.4f}", flush=True)
                del built


if __name__ == "__main__":
    main()
