#!/bin/bash
#SBATCH --partition="dev_gpu_h100"
#SBATCH --job-name="JudgeAnswers"
#SBATCH --output=%j-%x.out
#SBATCH --error=%j-%x.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --signal=USR1@30    # Send SIGUSR1 signal 30 seconds before the job ends to allow for graceful shutdown

set -e
set -u

: ' 
Usage: sbatch start-generate-judgements.sh [judge_model] [answer_file1] [answer_file2]
Arguments:
    judge_model             (optional) Name / path of model used to judge answers
                                Example models:
                                - meta-llama/Llama-3.3-70B-Instruct
                                - meta-llama/Llama-3.1-8B-Instruct
                                - Qwen/Qwen2.5-72B-Instruct
                                - mistralai/Mixtral-8x7B-Instruct-v0.1
                                - mistralai/Mistral-Large-Instruct-2411
    answer_file1          (optional) Name of the first answer file to compare
                                Example: gpt-4.1-answers.json
    answer_file2          (optional) Name of the second answer file to compare
                                Example: gemini-1.5-flash-answers.json

Notes: 
    The optional command line arguments will overwrite the values specified in the script.
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

if [[ $# -ge 1 ]]; then
    judge_model=$1
else
    judge_model="meta-llama/Llama-3.3-70B-Instruct" # meta-llama/Llama-3.3-70B-Instruct, microsoft/phi-4
    echo "No judge model provided as argument, using script default: $judge_model"
fi

if [[ $# -ge 2 ]]; then
    amswer_file1=$2
else
    answer_file1="gpt-4.1-answers_basic.json"
    echo "No answer file 1 provided as argument, using script default: $answer_file1"
fi

if [[ $# -ge 3 ]]; then
    answer_file2=$3
else
    answer_file2="Qwen2.5-3B-Instruct-answers.json"
    echo "No answer file 2 provided as argument, using script default: $answer_file2"
fi

data_path="data/generated_answers/"
judgments_path="data/judgements/"

model_1=$(echo "$answer_file1" | sed -E 's/-answers(_[a-z]+)?\.json$/\1/')
model_2=$(echo "$answer_file2" | sed -E 's/-answers(_[a-z]+)?\.json$/\1/')


# Output directory is the combination of the models and the judge model
experiment_name="$(basename "${judge_model}")---$model_1-vs-$model_2"
output_dir="${judgments_path}/${model_1}/vs_${model_2}/$(basename "${judge_model}")"
mkdir -p "$output_dir"

### Compare answers from the worse model to the answers from the better model
srun python src/compare_model_answers_batched.py \
    --judge_model_name_or_path "$judge_model" \
    --data_1_path "${data_path}${answer_file1}" \
    --data_2_path "${data_path}${answer_file2}" \
    --output_path "$output_dir" \
    --start_index "auto" \
    --step_size 142 # optional, default is 142 now as progress is automatically saved when time is running out
    # --question_switching # uncomment to switch questions between e.g. AAE and SAE style (depends if questions in files differ)
    # --introductionary_beginning # uncomment to add an introductionary beginning to the prompt e.g. "Hi there, I am kind of stuck on this question..."
    # --prompt_switching # TODO


# This checks whether all answers have been generated
# If script returns 0, we will continue with merging the results
# data_1_path is just used to compare how many answers there should be
if srun python src/check_if_all_data_processed.py --data_1_path "${data_path}${answer_file1}" --input_dir "$output_dir/"; then
    echo "All answers have been generated. Proceeding to merge results."
    srun python src/merge_json_files.py \
        --merge_path "$output_dir"
    
    echo "Converting merged JSON to XLSX..."
    srun python src/export_to_xlsx.py \
        --json_path "$output_dir/merged_data.json" \
        --xlsx_path "$output_dir/$experiment_name.xlsx"
else
    echo "Not all answers have been generated. Launching new job."
    sbatch start-generate-judgements.sh "$judge_model" "$answer_file1" "$answer_file2"
fi