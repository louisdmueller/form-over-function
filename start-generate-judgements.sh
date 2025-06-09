#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --job-name=FULL-LLM-Judge-Bias
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=01:00:00
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

judge_model_name="meta-llama/Llama-3.3-70B-Instruct"
# judge_model_name="RandomAnswer"
# judge_model_name="meta-llama/Llama-3.1-8B-Instruct"

### Compare answers from the worse model to the answers from the better model
python src/compare_model_answers_batched.py \
    --judge_model_name_or_path "$judge_model_name" \
    --data_1_path "data/gpt-4.1-answers_aae.json" \
    --data_2_path "data/gpt-neox-20b-answers-temperature-0.5.json" \
    --start_index 0.0 \
    # --end_index 0.5
    # --question_switching # TODO
    # --prompt_switchging # TODO
