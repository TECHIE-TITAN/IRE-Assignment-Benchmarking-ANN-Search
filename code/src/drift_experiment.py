"""Part 4: drifting data on synthetic clusters (see hypothesis.md).

For each index (HNSW, IVF-Flat, IVF-PQ):
  1. Build it once on the base vectors and keep a serialised copy, so every shift
     starts from the identical index.
  2. Calibrate the search setting on the control (shift 0): the smallest efSearch
     or nprobe with recall@10 >= 0.95 on the first half of the queries, or the
     setting with the highest recall if 0.95 is never reached. That setting is
     then fixed for every shift.
  3. For each shift (0 = control, 1, 2, 4 cluster radii):
       stale     : copy of the base index, then the shifted vectors are added
                   without retraining
       refreshed : IVF indexes retrained on all vectors, HNSW rebuilt from scratch
     Recall@10 is measured against exact search over base + added vectors, on the
     second half of the queries. QPS is one thread, one query per call, median of
     --repeats passes.

Rows go to ../results/results.csv with experiment=part4_drift. build_time_s is the
time to add the new vectors (stale) or to retrain/rebuild (refreshed); notes says which.

Usage:
  python src/drift_experiment.py            # full: 100K base + 50K added
  python src/drift_experiment.py --quick    # small sizes for run_all.sh --quick
"""
import argparse
import time
from dataclasses import replace

import faiss
import numpy as np
from benchmark import EF_SWEEP, NPROBE_SWEEP
from groundtruth import exact_knn
from indexes import build_index, params_str, set_search_param
from recall import recall_at_k
from results_log import log_run
from synthetic_data import DriftConfig, make_drift_dataset
from timing import set_single_thread, time_single_queries

K = 10
TARGET = 0.95
SHIFTS = [0, 1, 2, 4]  # cluster radii; 0 is the unchanged control
DATASET = "synthetic_gmm"


def index_params(quick):
    nlist = 128 if quick else 256
    return {
        "hnsw": {"M": 16, "efConstruction": 200},
        "ivfflat": {"nlist": nlist},
        "ivfpq": {"nlist": nlist, "m": 16},
    }


def sweep_values(kind, params):
    if kind == "hnsw":
        return EF_SWEEP
    return [v for v in NPROBE_SWEEP if v <= params["nlist"]]


def prepare(cfg, shift):
    """Vectors, queries and exact ground truth for one shift."""
    data = make_drift_dataset(cfg, shift)
    everything = np.vstack([data["base"], data["new"]])
    dist, _ = exact_knn(everything, data["queries"], K)
    return data, everything, dist[:, K - 1]


def stale_index(base_built, blob, new):
    """Copy of the base index with the shifted vectors added. Returns (Built, add seconds)."""
    index = faiss.deserialize_index(blob)
    t0 = time.perf_counter()
    index.add(new)
    seconds = time.perf_counter() - t0
    size = int(faiss.serialize_index(index).nbytes)
    return replace(base_built, index=index, size_bytes=size), seconds


def calibrate(built, values, queries, kth, everything):
    """Smallest search value reaching TARGET on the control, else the best one."""
    best = None
    for v in values:
        set_search_param(built, v)
        ids = built.index.search(queries, K)[1]
        rec = recall_at_k(ids, kth, everything, queries, k=K)
        if rec >= TARGET:
            return v, rec
        if best is None or rec > best[1] + 1e-9:
            best = (v, rec)
    return best


def list_diagnostics(built, queries, nprobe):
    """Largest and mean inverted-list size, and mean vectors scanned per query."""
    ivf = faiss.extract_index_ivf(built.index)
    sizes = np.array([ivf.invlists.list_size(i) for i in range(ivf.nlist)])
    lists = ivf.quantizer.search(queries, nprobe)[1]
    return int(sizes.max()), float(sizes.mean()), float(sizes[lists].sum(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes for run_all.sh --quick")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        cfg = DriftConfig(n_base=20_000, n_new=10_000, n_queries=2_000, seed=args.seed)
        repeats = 1
    else:
        cfg = DriftConfig(seed=args.seed)
        repeats = args.repeats
    params = index_params(args.quick)
    max_threads = faiss.omp_get_max_threads()
    half = cfg.n_queries // 2
    print(f"config: {cfg}\nshifts (cluster radii): {SHIFTS}, radius = {cfg.radius:.2f}", flush=True)

    # control data: base is identical for every shift, so indexes are built once
    control, control_all, control_kth = prepare(cfg, 0)
    base = control["base"]
    base_built, blobs, setting = {}, {}, {}
    for kind, p in params.items():
        faiss.omp_set_num_threads(max_threads)
        base_built[kind] = build_index(kind, base, **p)
        blobs[kind] = faiss.serialize_index(base_built[kind].index)
        print(f"[{kind}] built on base: {params_str(base_built[kind].params)} "
              f"in {base_built[kind].build_time_s:.1f}s", flush=True)

        calib, _ = stale_index(base_built[kind], blobs[kind], control["new"])
        value, rec = calibrate(calib, sweep_values(kind, p), control["queries"][:half],
                               control_kth[:half], control_all)
        setting[kind] = value
        print(f"[{kind}] calibrated on control (tune queries): "
              f"{calib.search_param_name}={value} recall={rec:.4f}", flush=True)

    for shift in SHIFTS:
        data, everything, kth = prepare(cfg, shift)
        assert np.array_equal(data["base"], base)
        final, kth_final = data["queries"][half:], kth[half:]
        print(f"\n===== shift = {shift} cluster radii =====", flush=True)

        for kind, p in params.items():
            for variant in ("stale", "refreshed"):
                faiss.omp_set_num_threads(max_threads)
                if variant == "stale":
                    built, seconds = stale_index(base_built[kind], blobs[kind], data["new"])
                    time_note = "add_new"
                else:
                    built = build_index(kind, everything, **p)
                    seconds, time_note = built.build_time_s, "rebuild"
                set_search_param(built, setting[kind])
                ids = built.index.search(final, K)[1]
                rec = recall_at_k(ids, kth_final, everything, final, k=K)

                notes = f"time={time_note}"
                if kind != "hnsw":
                    biggest, mean, scanned = list_diagnostics(built, final, setting[kind])
                    notes += f";max_list={biggest};mean_list={mean:.1f};scanned={scanned:.0f}"

                set_single_thread()
                res = time_single_queries(lambda q: built.index.search(q, K), final,
                                          repeats=repeats)
                faiss.omp_set_num_threads(max_threads)

                log_run(
                    experiment="part4_drift", dataset=DATASET, n_vectors=len(everything),
                    split="final", index=kind,
                    build_params=f"shift={shift};variant={variant};{params_str(built.params)}",
                    search_params=f"{built.search_param_name}={setting[kind]}", mode="single",
                    recall_at_10=round(rec, 5), qps=round(res["qps"], 2),
                    build_time_s=round(seconds, 3), index_bytes=built.size_bytes,
                    bytes_per_vector=round(built.size_bytes / len(everything), 2), notes=notes,
                )
                print(f"  {kind:8s} {variant:9s} recall={rec:.4f} qps={res['qps']:9.1f} "
                      f"{notes}", flush=True)
                del built


if __name__ == "__main__":
    main()
