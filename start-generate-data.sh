#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --job-name=LLM-Judge-Bias
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user="cluster-notifications.7fo8i@simplelogin.com"

set -e # Exit on error
set -u # Treat unset variables as an error

: ' 
Usage: sbatch start-generate_data.sh [answer_generation_model]
Arguments:
    answer_generation_model (optional) Name / path of model used to generate answers
                                Example models:
                                - gemini-1.5-flash
                                - meta-llama/Llama-3.1-8B-Instruct
                                - openai-community/gpt2
                                - EleutherAI/gpt-neo-1.3B
                                - Qwen/Qwen2-0.5B-Instruct

Notes: 
    The optional command line argument will overwrite the value specified in the script.
'

echo Time is `date +"%H:%M %d-%m-%y"`

export PYTHONPATH=${PWD}/src/

module load devel/cuda/12.8

module load devel/python/3.12.3-gnu-11.4

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

answer_generation_model="gemini-1.5-flash"

if [[ $# -ge 1 ]]; then
    answer_generation_model=$1
fi

echo "Running generation of answers."
echo "Chosen answer generation model: $answer_generation_model"
srun python src/generate_answers.py \
    --answer_generation_model_name_or_path $answer_generation_model \
    --output_path "data/generated_answers/$answer_generation_model-answers.json"
