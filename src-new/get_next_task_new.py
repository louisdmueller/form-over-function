import json
import os


def load_tasks_file(tasks_file_path: str) -> dict:
    with open(tasks_file_path, "r") as file:
        tasks_data = json.load(file)
    return tasks_data


def get_next_not_finished_task(tasks) -> dict:
    for task in tasks["tasks"]:
        if task["status"] == "NOT_FINISHED" and set(task["done"]) != set(
            tasks["base_data_variants"]
        ):
            return task
    return {}


def get_next_not_finished_task_with_base_data_variant(tasks) -> dict:
    task = get_next_not_finished_task(tasks)
    if task:
        for base_data_variant in tasks["base_data_variants"]:
            if base_data_variant not in task["done"]:
                task = {**task, "base_data_variant": base_data_variant}
                return task
    return {}


def get_next_task_path(tasks: dict, data_dir = "data/generated_answers") -> str:
    next_task = get_next_not_finished_task(tasks)
    if next_task:
        compare_against = next_task["compare_against"]
        if "-answers" not in compare_against:
            compare_against += "-answers"
        if not compare_against.endswith(".json"):
            compare_against += ".json"
        task_path = os.path.join(data_dir, compare_against)
        return task_path
    return ""

def mark_variant_as_done(task: dict, variant: str, tasks_filepath: str) -> dict:
    tasks = load_tasks_file(tasks_filepath)
    task.pop("base_data_variant", None)  # Remove base_data_variant key if exists
    for t in tasks["tasks"]:
        if t == task:
            t["done"].append(variant)
            if set(t["done"]) == set(tasks["base_data_variants"]):
                t["status"] = "COMPLETED"
            break
    with open(tasks_filepath, "w") as file:
        json.dump(tasks, file, indent=4)
    return load_tasks_file(tasks_filepath)

if __name__ == "__main__":
    tasks_file_path = "tasks.json"
    tasks = load_tasks_file(tasks_file_path)
    next_not_finished_task = get_next_not_finished_task_with_base_data_variant(tasks)
    print("Next not finished task:", next_not_finished_task)

    mark_variant_as_done(
        next_not_finished_task, next_not_finished_task["base_data_variant"], tasks_file_path
    )
    print("Marked variant as done for the first task.")