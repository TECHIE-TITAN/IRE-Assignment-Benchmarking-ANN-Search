#!/bin/bash
# Part 3 (scaling on SIFT): exact, HNSW and IVF-Flat at 100K and 300K vectors, then the analysis.
# The 1M point reuses the Part 1 / Part 2 rows already in results.csv.
# Submit from the code/ directory on the same node as Part 2:  sbatch -w gnode007 slurm_scaling.sh
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p data ../results/logs
hostname
python src/env_info.py

for N in 100000 300000; do
    python -u src/exact_baseline.py --datasets sift --n "$N" --experiment part3_scaling \
        2>&1 | tee "data/scaling_exact_${N}.txt"
    python -u src/benchmark.py --datasets sift --split final --n "$N" \
        --indexes hnsw ivfflat --experiment part3_scaling \
        2>&1 | tee "data/scaling_bench_${N}.txt"
done

python src/scaling_experiment.py
cp data/scaling_*.txt ../results/logs/
cp "slurm-${SLURM_JOB_ID}.out" "../results/logs/slurm-scaling-${SLURM_JOB_ID}.out" || true
echo "done"
