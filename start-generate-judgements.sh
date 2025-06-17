#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --job-name="JudgeAnswers"
#SBATCH --output=%j-%x.out
#SBATCH --error=%j-%x.err
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

data_path="data/"

judge_model_name="meta-llama/Llama-3.3-70B-Instruct"
# judge_model_name="meta-llama/Llama-3.1-8B-Instruct"
# judge_model_name="Qwen/Qwen2.5-72B-Instruct"
# judge_model_name="mistralai/Mistral-7B-Instruct-v0.2"
# judge_model_name="mistralai/Mistral-Large-Instruct-2411"

model_1_file="gpt-4.1-answers_aae.json"
model_2_file="Mistral-7B-Instruct-v0.2-answers.json"

model_1=$(echo "$model_1_file" | sed -E 's/-answers(_[a-z]+)?\.json$/\1/')
model_2=$(echo "$model_2_file" | sed -E 's/-answers(_[a-z]+)?\.json$/\1/')

# Output directory is the combination of the models and the judge model
experiment_name="$(basename "${judge_model_name}")---$model_1-vs-$model_2"
output_dir="$data_path/$experiment_name"
mkdir -p "$output_dir"

### Compare answers from the worse model to the answers from the better model
python src/compare_model_answers_batched.py \
    --judge_model_name_or_path "$judge_model_name" \
    --data_1_path "data/$model_1_file" \
    --data_2_path "data/$model_2_file" \
    --output_path "$output_dir" \
    --start_index "auto" \
    --step_size 71 # optional, default is 64
    # --question_switching # uncomment to switch questions between e.g. AAE and SAE style (depends if questions in files differ)
    # --introductionary_beginning # uncomment to add an introductionary beginning to the prompt e.g. "Hi there, I am kind of stuck on this question..."
    # --prompt_switching # TODO


# This checks whether all answers have been generated
# If script returns 0, we will continue with merging the results
# data_1_path is just used to compare how many answers there should be
if python src/check_if_all_data_processed.py --data_1_path "$model_1_file" --input_dir "$output_dir/"; then
    echo "All answers have been generated. Proceeding to merge results."
    python src/merge_json_files.py \
        --merge_path "$output_dir"
    
    echo "Converting merged JSON to XLSX..."
    python src/export_to_xlsx.py \
        --json_path "$output_dir/merged_data.json" \
        --xlsx_path "$output_dir/$experiment_name.xlsx"
else
    echo "Not all answers have been generated. Launching new job."
    sbatch start-generate-judgements.sh
fi