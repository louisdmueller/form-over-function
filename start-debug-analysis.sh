#!/bin/bash
#SBATCH --partition="cpu"
#SBATCH --job-name="Reasoning analysis"
#SBATCH --output=%j-%x.out
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END

set -e # exit on error
set -u # treat unset variables as an error

# export HF_HOME=/pfs/work9/workspace/scratch/hd_dg324-models

echo "Time is $(date +"%H:%M %d-%m-%y") ($(date +%s))"
echo "Endtime is $(date -d "@${SLURM_JOB_END_TIME}" '+%H:%M %d-%m-%y') (${SLURM_JOB_END_TIME})"


module load devel/cuda/12.8

# currently we use python 3.12.x
module load devel/python/3.12.3-gnu-11.4

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python src/analyze_reasonings.py