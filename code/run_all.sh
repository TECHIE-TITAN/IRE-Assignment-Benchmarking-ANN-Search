#!/bin/bash
# Regenerates every number, figure and table of the report from the data files.
#
#   bash run_all.sh                     full run on the complete datasets (hours)
#   bash run_all.sh --quick             same pipeline on the first 100,000 vectors
#   bash run_all.sh --results-dir DIR   write results to DIR (must not already hold a results.csv)
#   bash run_all.sh --with-tuning       also run the tuning sweeps on the first 5,000 queries
#                                       (logged to data/tuning.csv, not part of the reported numbers)
#
# Where results go: ../results if it has no results.csv yet, otherwise a new
# ../results_rerun_<timestamp>; quick mode uses ../results_quick (recreated each time).
# Existing results are never moved or overwritten.
#
# Run it from code/ with the virtual environment active. Needs data/sift-128-euclidean.hdf5 and
# data/glove-100-angular.hdf5 (see README.md).
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODE=full
TUNING=0
RESULTS_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) MODE=quick ;;
        --with-tuning) TUNING=1 ;;
        --results-dir) RESULTS_ARG=${2:?--results-dir needs a directory}; shift ;;
        -h|--help) sed -n '2,/^set /p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown option: $1 (try --help)"; exit 1 ;;
    esac
    shift
done

PY=${PYTHON:-python}
for f in data/sift-128-euclidean.hdf5 data/glove-100-angular.hdf5; do
    [ -f "$f" ] || { echo "missing $f: download it first (see README.md)"; exit 1; }
done
"$PY" -c "import faiss, numpy, h5py, pandas, matplotlib" \
    || { echo "python packages missing: activate the venv and run pip install -r requirements.txt"; exit 1; }

PARENT=$(cd .. && pwd)
if [ -n "$RESULTS_ARG" ]; then
    TARGET=$RESULTS_ARG
elif [ "$MODE" = quick ]; then
    TARGET="$PARENT/results_quick"
    rm -rf "$TARGET"
else
    TARGET="$PARENT/results"
    if [ -e "$TARGET/results.csv" ]; then
        TARGET="$PARENT/results_rerun_$(date +%Y%m%d-%H%M%S)"
        echo "$PARENT/results already holds results; this run writes to $TARGET instead"
    fi
fi
if [ -e "$TARGET/results.csv" ]; then
    echo "error: $TARGET already contains results.csv; choose a new directory"
    exit 1
fi
mkdir -p "$TARGET"
export A3_RESULTS_DIR=$(cd "$TARGET" && pwd)

if [ "$MODE" = quick ]; then
    NARG=(--n 100000); GRID=(--grid small); REPEATS=(--repeats 1); DRIFT=(--quick)
    SCALE_SMALL="10000 30000"; SCALE_ALL="10000 30000 100000"
else
    NARG=(); GRID=(); REPEATS=(); DRIFT=()
    SCALE_SMALL="100000 300000"; SCALE_ALL="100000 300000 1000000"
fi

LOG_DIR="data/logs_$MODE"
mkdir -p "$LOG_DIR"
START=$(date +%s)

# stage <name> <command...>: run, tee to a log, report the time taken
stage() {
    local name=$1; shift
    local t0; t0=$(date +%s)
    echo; echo "=== $name ==="
    "$@" 2>&1 | tee "$LOG_DIR/$name.txt"
    echo "--- $name finished in $(( $(date +%s) - t0 )) s"
}

echo "mode: $MODE   results: $A3_RESULTS_DIR"

stage tests            "$PY" -m pytest -q
stage env              "$PY" src/env_info.py

# Part 1: exact baseline
stage part1_exact      "$PY" -u src/exact_baseline.py ${NARG[@]+"${NARG[@]}"} ${REPEATS[@]+"${REPEATS[@]}"}

# Part 2: HNSW, IVF-Flat and IVF-PQ on both datasets, final 5,000 queries
if [ "$TUNING" = 1 ]; then
    stage part2_tuning "$PY" -u src/benchmark.py --split tune --timing-queries 500 --repeats 1 \
        ${NARG[@]+"${NARG[@]}"} ${GRID[@]+"${GRID[@]}"}
fi
stage part2_benchmark  "$PY" -u src/benchmark.py --split final \
    ${NARG[@]+"${NARG[@]}"} ${GRID[@]+"${GRID[@]}"} ${REPEATS[@]+"${REPEATS[@]}"}
stage part2_tables     "$PY" src/tables.py
stage part2_plots      "$PY" src/plots.py

# Part 3: scaling on SIFT (the largest size reuses the Part 1 and Part 2 rows)
for n in $SCALE_SMALL; do
    stage "part3_exact_$n" "$PY" -u src/exact_baseline.py --datasets sift --n "$n" \
        --experiment part3_scaling ${REPEATS[@]+"${REPEATS[@]}"}
    stage "part3_bench_$n" "$PY" -u src/benchmark.py --datasets sift --split final --n "$n" \
        --indexes hnsw ivfflat --experiment part3_scaling \
        ${GRID[@]+"${GRID[@]}"} ${REPEATS[@]+"${REPEATS[@]}"}
done
stage part3_analysis   "$PY" src/scaling_experiment.py --sizes $SCALE_ALL

# Part 4: drifting data on synthetic clusters
stage part4_drift      "$PY" -u src/drift_experiment.py ${DRIFT[@]+"${DRIFT[@]}"} ${REPEATS[@]+"${REPEATS[@]}"}
stage part4_analysis   "$PY" src/drift_analysis.py

ELAPSED=$(( $(date +%s) - START ))
echo
echo "done in $(( ELAPSED / 60 )) min $(( ELAPSED % 60 )) s. Results: $A3_RESULTS_DIR"
if [ "$MODE" = quick ] && [ "$ELAPSED" -gt 900 ]; then
    echo "warning: quick mode took longer than the 15 minute limit"
fi
