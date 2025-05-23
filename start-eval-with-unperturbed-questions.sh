#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --job-name=unperturbed-questions
#SBATCH --output=%j-unperturbed-questions.out
#SBATCH --error=%j-unperturbed-questions.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
set -e

echo Time is `date +"%H:%M %d-%m-%y"`

export PYTHONPATH=${PWD}/src/

## we have to decide which cuda version to use
# module load devel/cuda/11.7
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

python src/main.py