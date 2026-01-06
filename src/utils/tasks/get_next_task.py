import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def check_if_all_data_processed(input_dir, data_1_path):
    """
    Check if all data has been processed by comparing the last index in the newest JSON file
    with the expected length from the reference file.

    Args:
        input_dir (str, optional): Input directory path. If None, uses parsed args.
        data_1_path (str, optional): Path to reference data file. If None, uses parsed args.

    Returns:
        bool: True if all data is processed, False otherwise.
        int: The last index found in the newest JSON file.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    newest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(input_dir, f)))

    logger.info(f"Newest file found: {os.path.join(input_dir, newest_file)}")

    with open(os.path.join(input_dir, newest_file), "r") as f:
        data = json.load(f)

    indices = [int(indice) for indice in data.keys() if indice != "metadata"]
    max_index = max(indices)

    with open(data_1_path, "r") as f:
        length_reference_file = len(f.readlines()) - 1  # index starts at 0

    if max_index == length_reference_file:
        logger.info(
            f"Data processed completely. Last index in judge file: {max_index}, expected: {length_reference_file}"
        )
        return True, max_index
    else:
        logger.info(
            f"Data not processed completely. Last index in judge file: {max_index}, expected: {length_reference_file}"
        )
        return False, max_index


def load_available_answer_files() -> List[str]:
    """Load list of available answer files from the generated_answers directory."""
    answers_dir = "data/generated_answers/"
    if not os.path.exists(answers_dir):
        raise FileNotFoundError(f"Directory {answers_dir} not found")

    file_list = os.listdir(answers_dir)
    # Remove .json extension to get just the base names
    return [os.path.splitext(f)[0] for f in file_list if f.endswith(".json")]


def load_tasks_config(tasks_file: str) -> Dict:
    """Load tasks configuration from tasks.json."""
    with open(tasks_file, "r") as f:
        return json.load(f)


def validate_base_data_exists(
    base_data: str, base_data_variant: str, available_files: List[str]
) -> None:
    """Validate that the base data file exists."""
    base_data_file_name = generate_base_data_file_name(base_data, base_data_variant)
    if base_data_file_name not in available_files:
        raise FileNotFoundError(
            f"Base data file {base_data_file_name}.json not found in data/generated_answers/"
        )


def get_base_data(base_data: str, base_data_variant: str) -> str:
    """Get the full base data file name."""
    available_answer_files = load_available_answer_files()
    validate_base_data_exists(base_data, base_data_variant, available_answer_files)
    with open(
        generate_base_data_file_name(base_data, base_data_variant) + ".json", "r"
    ) as f:
        return json.load(f)


def generate_base_data_file_name(base_data, base_data_variant):
    if not base_data.endswith("-answers"):
        base_data += "-answers"
    return f"{base_data}_{base_data_variant}" if base_data_variant else base_data


def normalize_compare_against(compare_against: str) -> str:
    """Ensure compare_against ends with '-answers' suffix."""
    if not compare_against.endswith("-answers"):
        return f"{compare_against}-answers"
    return compare_against


def find_next_available_task(tasks: Dict) -> Tuple[str, Dict]:
    """Find the next task that is not finished, errored, or missing base data."""
    completed_statuses = ["FINISHED", "ERROR", "MISSING_BASE_DATA"]
    model_name = tasks["judge_model_name"]

    for task in tasks["tasks"]:
        if task["status"] not in completed_statuses:
            return model_name, task

    raise ValueError("No available tasks found")


def mark_task(
    tasks: Dict, model_name: str, task: Dict, message: str, index: Optional[int] = None
) -> None:
    """Mark a task with a specific message and save to tasks.json."""
    logger.info(
        f"Base data for {task['compare_against']} not found, marking task as {message}"
    )

    # Update task in-place at the same index
    task_idx = tasks["tasks"].index(task)
    task["status"] = message
    if index is not None:
        task["cur_idx"] = index
    tasks["tasks"][task_idx] = task

    with open("tasks.json", "w") as f:
        json.dump(tasks, f, indent=4)


def get_next_valid_task(
    tasks: Dict, base_data: str, base_data_variant: str
) -> Tuple[str, Dict]:
    """Get the next task that has all required base data available."""
    available_files = load_available_answer_files()
    base_data_dir_name = generate_base_data_dir_name(base_data, base_data_variant)
    judgements_path = "data/judgements/"
    while True:
        model_name, task = find_next_available_task(tasks)

        # Check if the required comparison data exists
        if normalize_compare_against(task["compare_against"]) not in available_files:
            mark_task(tasks, model_name, task, "MISSING_BASE_DATA")
            logger.info(f"Available files: {available_files}")
            logger.info(
                f"Required file {normalize_compare_against(task['compare_against'])}.json not found."
            )
            continue

        # check whether there are already judgements for this task
        comparison_data_dir = f"vs_{task['compare_against']}"
        judgement_dir = os.path.join(
            judgements_path,
            base_data_dir_name,
            comparison_data_dir,
            model_name.split("/")[-1],
        )
        if os.path.exists(judgement_dir) and os.listdir(judgement_dir):
            data_1_path = os.path.join(
                "data/generated_answers",
                generate_base_data_file_name(base_data, base_data_variant) + ".json",
            )
            all_data_processed, last_index = check_if_all_data_processed(
                judgement_dir, data_1_path
            )
            if all_data_processed:
                logger.info(
                    f"Judgements already exist in {judgement_dir}, marking task as FINISHED"
                )
                mark_task(tasks, model_name, task, "FINISHED")
                continue
            else:
                logger.info(
                    f"Judgements in {judgement_dir} are incomplete, proceeding with task"
                )
                mark_task(tasks, model_name, task, "IN_PROGRESS", index=last_index)

        return model_name, task


def generate_base_data_dir_name(base_data, base_data_variant):
    if not base_data_variant:
        return base_data
    return f"{base_data}_{base_data_variant}"


def create_json_file_name(base_data, base_data_variant):
    base_data = (
        base_data + "-answers" if not base_data.endswith("-answers") else base_data
    )
    if not base_data_variant:
        return f"{base_data}.json"
    return f"{base_data}_{base_data_variant}.json"


def main(tasks_file) -> Dict[str, str]:
    """Main function to get the next available task and prepare variables."""
    available_files = load_available_answer_files()
    tasks = load_tasks_config(tasks_file)

    base_data = tasks["base_data"]
    base_data_variant = tasks["base_data_variant"]

    validate_base_data_exists(base_data, base_data_variant, available_files)

    model_name, task = get_next_valid_task(tasks, base_data, base_data_variant)

    # Prepare output variables
    answer_file1 = create_json_file_name(base_data, base_data_variant)
    answer_file2 = create_json_file_name(task["compare_against"], "")

    return {
        "judge_model": model_name,
        "answer_file1": answer_file1,
        "answer_file2": answer_file2,
    }


if __name__ == "__main__":
    # Load configuration and available files
    assert len(sys.argv) == 2, "Usage: python get_next_task.py <tasks_file>"
    tasks_file = sys.argv[1]
    assert os.path.exists(tasks_file), f"Tasks file {tasks_file} does not exist"
    vars = main(tasks_file)

    # set env variables
    for key, value in vars.items():
        print("export {}='{}'".format(key, value))
