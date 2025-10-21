#!/bin/bash
#SBATCH --partition="dev_gpu_h100"
#SBATCH --job-name="Debug-vllm"
#SBATCH --output=%j-%x.out
#SBATCH --error=%j-%x.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
##SBATCH --signal=SIGUSR1@180    # Send SIGUSR1 signal 3 minutes before the job ends to allow for graceful shutdown

set -e
set -u

# export HF_HOME=/pfs/work9/workspace/scratch/hd_dg324-models

echo Time is `date +"%H:%M %d-%m-%y"`
start_time=$(date +%s)
remaining_time=$(squeue -h -j $SLURM_JOB_ID -O TimeLeft | \
awk -F '[-:]' '{
    if (NF == 1) { print $1 }
    else if (NF == 2) { print ($1 * 60) + $2 }
    else if (NF == 3) { print ($1 * 3600) + ($2 * 60) + $3 }
    else if (NF == 4) { print ($1 * 86400) + ($2 * 3600) + ($3 * 60) + $4 }
}')
end_time=$((start_time + remaining_time))

module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-11.4
module load devel/miniforge
eval "$(conda shell.bash hook)"
# conda activate vllm-env
conda activate vLLM-test-clone


# if [ ! -d "vllm-venv" ]; then
#     if [ ! -d "uv" ]; then
#         echo "Installing uv..."
#         curl -LsSf https://astral.sh/uv/install.sh | sh
#         uv init --python 3.12.3
#         uv python pin 3.12
#     fi
#     echo "Creating virtual environment..."
#     uv venv vllm-venv --python 3.12 --seed
#     source vllm-venv/bin/activate
#     # I am currently using a lockfile since I cannot get torch gpu
#     # to install with `uv add`
#     # And I can also not transform the requirements-vllm.txt to a uv lockfile
#     # `uv add -r requirements-vllm.txt --active` does not work
#     # So for now, just use pip install
#     uv pip install -r requirements-vllm.txt
# else
#     source vllm-venv/bin/activate
# fi
# source vllm-venv/bin/activate

python src-new/generate_judgements.py

conda deactivate
module unload devel/miniforge