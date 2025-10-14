#!/bin/bash
#SBATCH --partition="gpu_h100"
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
#SBATCH --signal=SIGUSR1@180    # Send SIGUSR1 signal 3 minutes before the job ends to allow for graceful shutdown

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
start_time=$(date +%s)
remaining_time=$(squeue -h -j $SLURM_JOB_ID -O TimeLeft | \
awk -F '[-:]' '{
    if (NF == 1) { print $1 }
    else if (NF == 2) { print ($1 * 60) + $2 }
    else if (NF == 3) { print ($1 * 3600) + ($2 * 60) + $3 }
    else if (NF == 4) { print ($1 * 86400) + ($2 * 3600) + ($3 * 60) + $4 }
}')
end_time=$((start_time + remaining_time))

export PYTHONPATH=${PWD}/src/

module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-11.4
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate qwen-new

# if [ ! -d "venv" ]; then
#     echo "Creating virtual environment..."
#     python -m venv venv
#     source venv/bin/activate
#     pip install -r requirements.txt
# else
#     source venv/bin/activate
# fi

# sets "judge_model", "answer_file1" and "answer_file2" env variables
# check if python has thrown error
if ! python3 src/slurm_helper_scripts/get_next_task.py > /tmp/tmp_env.sh; then
    echo "Failed to get next task. Exiting."
    exit 1
else
    cat /tmp/tmp_env.sh
fi

# source the generated env file to get the variables
source /tmp/tmp_env.sh
echo Read in variables:
echo "  judge_model: $judge_model"
echo "  answer_file1: $answer_file1"
echo "  answer_file2: $answer_file2"

data_path="data/generated_answers/"
judgments_path="data/judgements/"

model_1=$(echo "$answer_file1" | sed -E 's/-answers(_[a-z]+)?\.json$/\1/')
model_2=$(echo "$answer_file2" | sed -E 's/-answers(_[a-z]+)?\.json$/\1/')

echo "Model 1: $model_1"
echo "Model 2: $model_2"

# Output directory is the combination of the models and the judge model
experiment_name="$(basename "${judge_model}")---$model_1-vs-$model_2"
output_dir="${judgments_path}/${model_1}/vs_${model_2}/$(basename "${judge_model}")"
mkdir -p "$output_dir"

echo "Experiment name: $experiment_name"
echo "Output directory: $output_dir"

### Compare answers from the worse model to the answers from the better model
python -u src/compare_model_answers_batched.py \
    --judge_model_name_or_path "$judge_model" \
    --data_1_path "${data_path}${answer_file1}" \
    --data_2_path "${data_path}${answer_file2}" \
    --output_path "$output_dir" \
    --start_index "auto" \
    --end_time "$end_time" \
    --data_fraction 1 # process all data, for testing you can set e.g. 0.1 to only process 10% of the data
    # --question_switching # uncomment to switch questions between e.g. AAE and SAE style (depends if questions in files differ)
    # --introductionary_beginning # uncomment to add an introductionary beginning to the prompt e.g. "Hi there, I am kind of stuck on this question..."
    # --prompt_switching # TODO

echo ""

# This checks whether all answers have been generated
# If script returns 0, we will continue with merging the results
# data_1_path is just used to compare how many answers there should be
if python src/slurm_helper_scripts/check_if_all_data_processed.py --data_1_path "${data_path}${answer_file1}" --input_dir "$output_dir/"; then
    python src/slurm_helper_scripts/set_task_finished.py "$judge_model" "$model_2"
    echo "All answers have been generated. Proceeding to merge results."
    python src/merge_json_files.py \
        --merge_path "$output_dir"
    
    echo "Converting merged JSON to XLSX..."
    python src/export_to_xlsx.py \
        --json_path "$output_dir/merged_data.json" \
        --xlsx_path "$output_dir/$experiment_name.xlsx"

    echo Starting next job to generate more judgements.
else
    echo "Not all answers have been generated. Launching new job."
fi

conda deactivate
module unload devel/miniforge
sbatch start-generate-judgements.sh