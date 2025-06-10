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

# answer_generation_model_name_or_path="gemini-1.5-flash"  # model that generates answers to the questions
# answer_generation_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
# answer_generation_model_name_or_path="openai-community/gpt2"
# answer_generation_model_name_or_path="EleutherAI/gpt-neo-1.3B"
answer_generation_model_name_or_path="EleutherAI/gpt-neox-20b"
# answer_generation_model_name_or_path="huggyllama/llama-13b"
prompt_model_name_or_path="gpt-4o-mini" # model that converts answers to aae answers # only used if --aae is set

python src/generate_answers.py \
    --answer_generation_model_name_or_path "$answer_generation_model_name_or_path" \
    --prompt_model_name_or_path "$prompt_model_name_or_path" \
    --output_path "data/$answer_generation_model_name_or_path-answers.json" \
    # --aae
    # TODO: implement input_path so already generated answers can be translated to aae
