#!/bin/bash
# Runs every benchmark of Part 1 and Part 2 on a single node, then builds tables and figures.
# Submit from the code/ directory:  sbatch slurm_job.sh
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load u18/python/3.10.2
source .venv/bin/activate

# builds and batch searches use the CPUs allocated to this job; timing runs are forced to 1 thread in code
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p data
hostname
lscpu | head -20
python src/env_info.py
python -m pytest -q

python -u src/exact_baseline.py 2>&1 | tee data/exact_baseline.txt
python -u src/benchmark.py --datasets sift  --split final 2>&1 | tee data/final_sift.txt
python -u src/benchmark.py --datasets glove --split final 2>&1 | tee data/final_glove.txt

python src/tables.py
python src/plots.py

mkdir -p ../results/logs
cp data/exact_baseline.txt data/final_sift.txt data/final_glove.txt ../results/logs/
cp "slurm-${SLURM_JOB_ID}.out" ../results/logs/ || true
echo "done"
