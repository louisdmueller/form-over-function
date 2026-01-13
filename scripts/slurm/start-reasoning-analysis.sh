#!/bin/bash
#SBATCH --partition="cpu"
#SBATCH --job-name="reasoning-analysis"
#SBATCH --output=outputs/slurm/job-%x/%j.out
#SBATCH --error=outputs/slurm/job-%x/%j.err
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL,END

set -e # exit on error
set -u # treat unset variables as an error

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Time is $(date +"%H:%M %d-%m-%y") ($(date +%s))"
echo "Endtime is $(date -d "@${SLURM_JOB_END_TIME}" '+%H:%M %d-%m-%y') (${SLURM_JOB_END_TIME})"


module load devel/cuda/12.8

# currently we use python 3.12.x
module load devel/python/3.12.3-gnu-11.4

VENV_DIR="$MAIN_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -r "$MAIN_DIR/requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

python src/evaluation/analyze_reasonings.py