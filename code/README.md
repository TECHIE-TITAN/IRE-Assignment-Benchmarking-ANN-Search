# Assignment 3: Benchmarking Approximate Nearest Neighbour Search

Benchmarks three approximate nearest neighbour indexes (HNSW, IVF-Flat, IVF-PQ, all from FAISS) against exact search on SIFT-1M and GloVe-100, and runs two experiments: scaling on SIFT (Part 3) and drifting synthetic data (Part 4). Everything in the report is produced by one command, `bash run_all.sh`.

The recall function, timing code and plots are written from scratch. FAISS is used only for the indexes and for exact search.

## Folder layout

`results/` sits next to `code/`, not inside it:

```
<root>/
  code/        this repository
    src/       library and experiment code
    tests/     unit tests (recall function, synthetic data generator)
    data/      datasets and caches (not tracked, created by you, see Setup)
    run_all.sh the one-command reproduction
  results/     results.csv and figures/ (written by run_all.sh)
```

## Setup

Requires Python 3.9 or newer on Linux or macOS. Run everything from inside `code/`.

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the reported results: faiss-cpu 1.13.0, numpy 2.0.2, h5py 3.14.0, pandas 2.3.3, matplotlib 3.9.4, pytest 8.4.2.

### Datasets

Download the two files from ann-benchmarks into `code/data/` (about 0.5 GB each). They are not included in the submission.

```bash
mkdir -p data
curl -L -o data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5
curl -L -o data/glove-100-angular.hdf5  http://ann-benchmarks.com/glove-100-angular.hdf5
```

Each file holds the vectors to index, 10,000 queries and the true neighbours. GloVe vectors are L2-normalised on load and Euclidean distance is used for everything. The queries are split in half: the first 5,000 are for tuning and the last 5,000 are for every reported number.

## Running

Make sure the virtual environment is active, then, from `code/`:

```bash
bash run_all.sh             # full run on the complete datasets
bash run_all.sh --quick     # same pipeline on the first 100,000 vectors
```

| Option | Effect |
|---|---|
| `--quick` | Uses the first 100,000 vectors, a reduced Part 2 grid, one timing repeat, scaling sizes 10K, 30K and 100K, and a smaller drift experiment. Designed to finish in under 15 minutes on a laptop. Writes to `../results_quick`, which is deleted and recreated on every quick run. |
| `--results-dir DIR` | Write results to `DIR`. Refuses to run if `DIR` already contains a `results.csv`. |
| `--with-tuning` | Also runs the tuning sweeps on the first 5,000 queries (logged to `data/tuning.csv`, not used for any reported number). |
| `--help` | Prints the usage. |

Where a full run writes: `../results` if that folder has no `results.csv` yet, otherwise a new `../results_rerun_<timestamp>`. Existing results are never moved or overwritten.

A full run takes about 3 hours on the machine described below (the long part is the Part 2 sweeps at 1M vectors). Each stage prints its own timing, and the logs go to `data/logs_full/` (or `data/logs_quick/`).

### What the script runs

| Stage | What it does |
|---|---|
| tests | Unit tests for the recall function and the synthetic data generator |
| env | Records the machine and library versions to `env.json` |
| Part 1 | Exact search: agreement with the stored neighbours, one-query-at-a-time versus batch timing |
| Part 2 | HNSW, IVF-Flat and IVF-PQ on both datasets: build settings times search-parameter sweeps, tables and the recall-versus-QPS figures |
| Part 3 | Scaling on SIFT at 100K, 300K and 1M vectors: cost of reaching recall 0.95 for exact search, HNSW and IVF-Flat |
| Part 4 | Drifting synthetic data: indexes built on one distribution, then shifted vectors added; stale versus retrained or rebuilt |

### Running one piece

All scripts are in `src/` and take `--help`. For example:

```bash
python src/exact_baseline.py --datasets sift
python src/benchmark.py --split final --datasets glove --indexes hnsw
python src/drift_experiment.py
python src/tables.py && python src/plots.py
python src/compare_runs.py ../results ../results_rerun_<timestamp>
```

Scripts write to `../results` by default. To redirect any of them, set `A3_RESULTS_DIR` to a folder. `run_all.sh` does this for you.

## Outputs

| Path | Contents |
|---|---|
| `results/results.csv` | One row per benchmark run behind a reported number: experiment, dataset, size, split, index, build and search parameters, recall@10, QPS, build time, index size |
| `results/figures/` | `recall_vs_qps_sift`, `recall_vs_qps_glove`, `scaling_sift`, `drift_main` (`.png` and `.pdf`) |
| `results/tables/` | The Part 2 tables (QPS at recall 0.90 and 0.95; build time, size and bytes per vector), the Part 3 scaling table and the Part 4 drift table |
| `results/env.json` | Machine and library versions of the run |

In `results.csv`, `experiment` is `part1` (exact baseline), `part2` (main benchmark), `part3_scaling` (the 100K and 300K runs; the 1M point reuses the `part1` and `part2` rows) or `part4_drift`. All QPS values are one thread, one query per call, median of three passes.

## Source files

| File | Purpose |
|---|---|
| `src/data.py` | Dataset loading, GloVe normalisation, tune/final query split |
| `src/recall.py` | Tie-aware recall@10 (a result tied with the 10th neighbour is not a mistake) |
| `src/timing.py` | Single-thread timing: one query per call, warm-up, median over repeats |
| `src/groundtruth.py` | Exact top-100 from `IndexFlatL2`, cached under `data/` |
| `src/indexes.py` | Builders for HNSW, IVF-Flat and IVF-PQ; build time and index size |
| `src/exact_baseline.py` | Part 1 |
| `src/benchmark.py` | Part 2 benchmark (also used for the Part 3 sizes) |
| `src/tables.py`, `src/plots.py` | Part 2 tables and recall-versus-QPS figures |
| `src/scaling_experiment.py` | Part 3 analysis, table and figure |
| `src/synthetic_data.py` | Seeded generator for the Part 4 clustered data |
| `src/drift_experiment.py`, `src/drift_analysis.py` | Part 4 experiment, table and figure |
| `src/compare_runs.py` | Compares two result folders (run-to-run variation) |
| `src/results_log.py`, `src/env_info.py` | Result logging and machine information |
| `hypothesis.md` | The hypotheses for Part 3 and Part 4 |
| `slurm_*.sh` | Slurm job scripts used on the Ada cluster (not needed elsewhere; `slurm_all.sh` wraps `run_all.sh`) |

## Reproducibility

The reported results come from one full `bash run_all.sh` run on a single node:

| | |
|---|---|
| CPU | Intel Xeon E5-2640 v4 @ 2.40 GHz (8 CPUs allocated to the job) |
| OS | Ubuntu 22.04 (Linux 5.15, glibc 2.35) |
| Python | 3.10.12 |
| Libraries | as pinned in `requirements.txt` |

Rerunning on the same machine reproduces the results as follows:

- **Recall:** identical for exact search, IVF-Flat and IVF-PQ (k-means and PQ training are seeded and the synthetic data is seeded). HNSW recall moves by at most 0.004, because an HNSW graph built on several threads differs slightly from run to run.
- **Speed and build time:** these depend on the machine and on what else is running on it. Between two full runs on the same shared node, the median QPS of a group of runs differed by 3% to 50% depending on the experiment and the index, always in the same direction (the second run was faster). Compare QPS within one run, not across runs.
- **Quick mode** can be run on any machine. Its numbers are for checking that the pipeline works end to end and are not comparable with the full results.

`src/compare_runs.py` prints these differences for any two result folders.
