#!/bin/bash
#SBATCH --partition="dev_gpu_h100"
#SBATCH --job-name="vllm-judgements"
#SBATCH --output=outputs/slurm/job-%x/%j.out
#SBATCH --error=outputs/slurm/job-%x/%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL

set -e # exit on error
set -u # treat unset variables as an error

# export HF_HOME=/pfs/work9/workspace/scratch/hd_dg324-models

echo "Time is $(date +"%H:%M %d-%m-%y") ($(date +%s))"
echo "Endtime is $(date -d "@${SLURM_JOB_END_TIME}" '+%H:%M %d-%m-%y') (${SLURM_JOB_END_TIME})"


module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-11.4
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate vLLM-test-clone

python -u src/main.py --multi_tasks_mode


# check if tasks file all finished
if python src/utils/tasks/check_all_metatasks_finished.py ; then
    echo "All tasks finished, nothing more to do."
else
    conda deactivate
    module unload devel/miniforge
    sbatch scripts/slurm/start-generate-judgements.sh
fi