#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --job-name=unperturbed-questions
#SBATCH --output=%j-unperturbed-questions.out
#SBATCH --error=%j-unperturbed-questions.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
set -e

echo Time is `date +"%H:%M %d-%m-%y"`

export PYTHONPATH=${PWD}/src/

## we have to decide which cuda version to use
# module load devel/cuda/11.7
module load devel/cuda/12.8

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    pip install -r requirements.txt
fi
source venv/bin/activate

python main.py