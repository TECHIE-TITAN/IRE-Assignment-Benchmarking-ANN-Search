#!/bin/bash
# Full pipeline (bash run_all.sh) on one fixed node, writing to ../results_v2.
# Submit from the code/ directory:  sbatch slurm_all.sh
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH --nodelist=gnode007
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

hostname
lscpu | head -20

bash run_all.sh --results-dir ../results_v2

mkdir -p ../results_v2/logs
cp data/logs_full/*.txt ../results_v2/logs/
cp "slurm-${SLURM_JOB_ID}.out" "../results_v2/logs/slurm-all-${SLURM_JOB_ID}.out" || true
echo "done"
