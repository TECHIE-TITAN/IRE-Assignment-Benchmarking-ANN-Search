"""Part 1: exact baseline.

1. Build IndexFlatL2 and compute exact ground truth (cached).
2. Check agreement with the neighbours stored in the data file.
3. Time exact search: one query at a time, and all queries in one batch.

Usage: python src/exact_baseline.py [--datasets sift glove] [--n N] [--repeats R]
"""
import argparse
import time

import numpy as np

import faiss
from data import FINAL, load_dataset, split_queries
from groundtruth import kth_dist_sq, load_or_compute_gt
from recall import recall_at_k
from results_log import log_run
from timing import set_single_thread, time_batch, time_single_queries

K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sift", "glove"])
    ap.add_argument("--n", type=int, default=None, help="use first N base vectors")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--experiment", default="part1", help="label for the experiment column")
    args = ap.parse_args()

    max_threads = faiss.omp_get_max_threads()

    for name in args.datasets:
        print(f"\n=== {name} ===")
        faiss.omp_set_num_threads(max_threads)
        d = load_dataset(name, n=args.n)
        train, test = d["train"], d["test"]
        _, final = split_queries(test)
        n = len(train)

        t0 = time.perf_counter()
        index = faiss.IndexFlatL2(train.shape[1])
        index.add(train)
        build_time = time.perf_counter() - t0
        index_bytes = train.nbytes

        D_all, I_all = load_or_compute_gt(name, train, test, k=100)
        kth_final = kth_dist_sq(D_all, K)[FINAL]

        # compare our exact results with the neighbours stored in the HDF5 file
        if d["neighbors"] is not None:
            stored = d["neighbors"][:, :K]
            tie_aware = recall_at_k(stored, kth_dist_sq(D_all, K), train, test, k=K)
            ours = I_all[:, :K]
            same_set = np.mean(
                [len(set(a) & set(b)) == K for a, b in zip(stored, ours)]
            )
            print(f"agreement with stored neighbours (all {len(test)} queries):")
            print(f"  tie-aware recall@{K} of stored vs ours : {tie_aware:.6f}")
            print(f"  queries with identical top-{K} id sets : {same_set:.4%}")
        else:
            print("subset run: stored neighbours not valid, skipping agreement check")

        base_row = dict(
            experiment=args.experiment, dataset=name, n_vectors=n, split="final",
            index="exact", build_params="IndexFlatL2", build_time_s=round(build_time, 4),
            index_bytes=index_bytes, bytes_per_vector=index_bytes / n,
        )

        def search(q):
            return index.search(q, K)

        # ids from a batch run (all threads) for the recall figure
        _, ids = index.search(final, K)
        rec = recall_at_k(ids, kth_final, train, final, k=K)
        print(f"exact recall@{K} on final queries: {rec:.6f}")

        set_single_thread()
        single = time_single_queries(search, final, repeats=args.repeats)
        batch1 = time_batch(search, final, repeats=args.repeats)
        faiss.omp_set_num_threads(max_threads)
        batchN = time_batch(search, final, repeats=args.repeats)

        for mode, res, note in [
            ("single", single, "1 thread, one query per call"),
            ("batch", batch1, "1 thread, all queries in one call"),
            ("batch", batchN, f"{max_threads} threads, all queries in one call"),
        ]:
            log_run(**base_row, mode=mode, recall_at_10=rec,
                    qps=round(res["qps"], 2), notes=note)
            print(f"  {note:42s} {res['qps']:12.1f} QPS "
                  f"({1000 / res['qps']:.3f} ms/query)")


if __name__ == "__main__":
    main()
