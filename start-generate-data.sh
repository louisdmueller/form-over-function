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

answer_generation_model="gemini-1.5-flash"
aae_conversion_model=""

if [[ $# -ge 1 ]]; then
    answer_generation_model=$1
fi

if [[ $# -ge 2 ]]; then
    aae_conversion_model=$2
fi

if [[ -n "$aae_conversion_model" ]]; then
    echo "Converting SAE answers to AAE, since conversion model was given.
    Chosen answer generation model: $answer_generation_model
    Chosen AAE conversion model: $aae_conversion_model"
    python src/generate_answers.py \
        --answer_generation_model_name_or_path "$answer_generation_model" \
        --prompt_model_name_or_path "$aae_conversion_model" \
        --output_path "data/$answer_generation_model-answers.json" \
        --aae
        
else
   echo "Running generation without converting of answers to AAE."
    echo "Chosen answer generation model: $answer_generation_model"
    python src/generate_answers.py \
        --answer_generation_model_name_or_path $answer_generation_model \
        --output_path "data/$answer_generation_model-answers.json"
fi
