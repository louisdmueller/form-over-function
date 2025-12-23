#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --job-name=JudgeAnswers-AAVE
#SBATCH --output=outputs/slurm/%x/%j.out
#SBATCH --error=outputs/slurm/%x/%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user="cluster-notifications.7fo8i@simplelogin.com"
set -e

echo Time is `date +"%H:%M %d-%m-%y"`

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

# Output directory is the name of the job
output_dir="data/${SLURM_JOB_NAME}/Qwen2.5-72B-Instruct"
# Choose an output directory
data_1_path="data/Qwen2-0.5B-Instruct-answers.json"
mkdir -p "$output_dir"

# judge_model_name="meta-llama/Llama-3.3-70B-Instruct"
judge_model_name="Qwen/Qwen2.5-72B-Instruct"
# judge_model_name="mistralai/Mistral-7B-Instruct-v0.2"
# judge_model_name="RandomAnswer"

### Compare answers using the main.py script
cd src
python -u main.py \
    --config_path "../config.yml" \
    --tasks_file "../tasks_files/tasks_gpt-oss-120b.json"
cd ..

# This checks whether all answers have been generated
# If script returns 0, we will continue with merging the results
if python src/utils/slurm_helper_scripts/check_if_all_data_processed.py --data_1_path "$data_1_path" --input_dir "$output_dir/"; then
    echo "All answers have been generated. Judgements processing complete."
else
    echo "Not all answers have been generated. Launching new job."
    sbatch start-generate-judgements_aae.sh
fi