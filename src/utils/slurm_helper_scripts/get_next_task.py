import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from src.utils.slurm_helper_scripts.check_if_all_data_processed import (
    check_if_all_data_processed,
)


def load_available_answer_files() -> List[str]:
    """Load list of available answer files from the generated_answers directory."""
    answers_dir = "data/generated_answers/"
    if not os.path.exists(answers_dir):
        raise FileNotFoundError(f"Directory {answers_dir} not found")
    
    file_list = os.listdir(answers_dir)
    # Remove .json extension to get just the base names
    return [os.path.splitext(f)[0] for f in file_list if f.endswith('.json')]

def load_tasks_config() -> Dict:
    """Load tasks configuration from tasks.json."""
    with open("tasks.json", "r") as f:
        return json.load(f)

def validate_base_data_exists(base_data: str, base_data_variant: str, available_files: List[str]) -> None:
    """Validate that the base data file exists."""
    base_data_file_name = generate_base_data_file_name(base_data, base_data_variant)
    if base_data_file_name not in available_files:
        raise FileNotFoundError(f"Base data file {base_data_file_name}.json not found in data/generated_answers/")

def generate_base_data_file_name(base_data, base_data_variant):
    if not base_data.endswith("-answers"):
        base_data += "-answers"
    base_data_file_name = f"{base_data}_{base_data_variant}"
    return base_data_file_name

def normalize_compare_against(compare_against: str) -> str:
    """Ensure compare_against ends with '-answers' suffix."""
    if not compare_against.endswith("-answers"):
        return f"{compare_against}-answers"
    return compare_against

def find_next_available_task(tasks: Dict) -> Tuple[str, Dict]:
    """Find the next task that is not finished, errored, or missing base data."""
    completed_statuses = ["FINISHED", "ERROR", "MISSING_BASE_DATA"]
    
    for model_name, model_tasks in tasks["tasks"].items():
        for task in model_tasks:
            if task["status"] not in completed_statuses:
                return model_name, task
    
    raise ValueError("No available tasks found")

def mark_task(tasks: Dict, model_name: str, task: Dict, message: str, index: Optional[int] = None) -> None:
    """Mark a task with a specific message and save to tasks.json."""
    print(f"Base data for {task['compare_against']} not found, marking task as {message}", file=sys.stderr)

    # Update task in-place at the same index
    task_idx = tasks["tasks"][model_name].index(task)
    task["status"] = message
    if index is not None:
        task["cur_idx"] = index
    tasks["tasks"][model_name][task_idx] = task

    with open("tasks.json", "w") as f:
        json.dump(tasks, f, indent=4)

def get_next_valid_task(tasks: Dict, available_files: List[str], base_data: str, base_data_variant: str) -> Tuple[str, Dict]:
    """Get the next task that has all required base data available."""

    base_data_dir_name = generate_base_data_dir_name(base_data, base_data_variant)
    judgements_path = "data/judgements/"
    while True:
        model_name, task = find_next_available_task(tasks)
        
        # Check if the required comparison data exists
        if normalize_compare_against(task["compare_against"]) not in available_files:
            mark_task(tasks, model_name, task, "MISSING_BASE_DATA")
            print(f"Available files: {available_files}", file=sys.stderr)
            print(f"Required file {normalize_compare_against(task['compare_against'])}.json not found.", file=sys.stderr)
            continue

        # check whether there are already judgements for this task
        comparison_data_dir = f"vs_{task['compare_against']}"
        judgement_dir = os.path.join(
            judgements_path, 
            base_data_dir_name, 
            comparison_data_dir,
            model_name.split("/")[-1]
        )
        if os.path.exists(judgement_dir) and os.listdir(judgement_dir):
            data_1_path = os.path.join("data/generated_answers", generate_base_data_file_name(base_data, base_data_variant) + ".json")
            all_data_processed, last_index = check_if_all_data_processed(judgement_dir, data_1_path)
            if all_data_processed:
                print(f"Judgements already exist in {judgement_dir}, marking task as FINISHED", file=sys.stderr)
                mark_task(tasks, model_name, task, "FINISHED")
                continue
            else:
                print(f"Judgements in {judgement_dir} are incomplete, proceeding with task", file=sys.stderr)
                mark_task(tasks, model_name, task, "IN_PROGRESS", index=last_index)

        return model_name, task

def generate_base_data_dir_name(base_data, base_data_variant):
    if not base_data_variant:
        return base_data
    return f"{base_data}_{base_data_variant}"
    
def create_json_file_name(base_data, base_data_variant):
    base_data = base_data + "-answers" if not base_data.endswith("-answers") else base_data
    if not base_data_variant:
        return f"{base_data}.json"
    return f"{base_data}_{base_data_variant}.json"

def main():
    """Main function to get the next available task and prepare variables."""
    # Load configuration and available files
    available_files = load_available_answer_files()
    tasks = load_tasks_config()
    
    base_data = tasks["base_data"]
    base_data_variant = tasks["base_data_variant"]
    
    validate_base_data_exists(base_data, base_data_variant, available_files)

    model_name, task = get_next_valid_task(tasks, available_files, base_data, base_data_variant)

    # Prepare output variables
    answer_file1 = create_json_file_name(base_data, base_data_variant)
    answer_file2 = create_json_file_name(task["compare_against"], "")

    return {
        "judge_model": model_name,
        "answer_file1": answer_file1,
        "answer_file2": answer_file2,
    }

# Execute main logic
if __name__ == "__main__":
    vars = main()
    
    # set env variables
    for key, value in vars.items():
        print("export {}='{}'".format(key, value))
else:
    # For when this module is imported
    vars = main()