#!/bin/bash
# Part 4 (drifting data on synthetic clusters), then the tables and figure.
# Submit from the code/ directory on the same node as Part 2:  sbatch -w gnode007 slurm_drift.sh
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
python -m pytest -q

python -u src/drift_experiment.py 2>&1 | tee data/drift_experiment.txt
python src/drift_analysis.py

cp data/drift_experiment.txt ../results/logs/
cp "slurm-${SLURM_JOB_ID}.out" "../results/logs/slurm-drift-${SLURM_JOB_ID}.out" || true
echo "done"
