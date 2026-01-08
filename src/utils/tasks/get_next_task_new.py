import json
import os
from typing import Any
import logging

logger = logging.getLogger(__name__)

def load_tasks_file(tasks_file_path: str) -> dict:
    with open(tasks_file_path, "r") as file:
        tasks_data = json.load(file)
    return tasks_data


def get_next_not_finished_task(tasks) -> dict:
    for task in tasks["tasks"]:
        task.setdefault("submitted", [])
        if task["status"] == "NOT_FINISHED" and set(task["done"]) != set(
            tasks["base_data_variants"]
        ):
            return task
    return {}


def is_all_subtasks_finished(tasks) -> bool:
    if get_next_not_finished_task(tasks):
        return False
    return True


def get_next_not_finished_task_with_base_data_variant(tasks) -> dict | None:
    task = get_next_not_finished_task(tasks)
    if task:
        submitted = set(task.get("submitted", []))
        for base_data_variant in tasks["base_data_variants"]:
            if base_data_variant not in task["done"] and base_data_variant not in submitted:
                task = {**task, "base_data_variant": base_data_variant}
                return task
    return None


def get_next_task_path(tasks: dict, data_dir="data/generated_answers") -> str:
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
    target_compare = task.get("compare_against")
    for t in tasks["tasks"]:
        if t.get("compare_against") != target_compare:
            continue
        submitted = set(t.get("submitted", []))
        if variant in submitted:
            submitted.discard(variant)
            t["submitted"] = list(submitted)
        if variant not in t["done"]:
            t["done"].append(variant)
        if set(t["done"]) == set(tasks["base_data_variants"]):
            t["status"] = "COMPLETED"
        break
    with open(tasks_filepath, "w") as file:
        json.dump(tasks, file, indent=4)
    return load_tasks_file(tasks_filepath)


def mark_variant_as_submitted(task: dict, variant: str, tasks_filepath: str) -> dict:
    tasks = load_tasks_file(tasks_filepath)
    target_compare = task.get("compare_against")
    for t in tasks["tasks"]:
        t.setdefault("submitted", [])
        if t.get("compare_against") != target_compare:
            continue
        if variant not in t["submitted"] and variant not in t["done"]:
            t["submitted"].append(variant)
        break
    with open(tasks_filepath, "w") as file:
        json.dump(tasks, file, indent=4)
    return load_tasks_file(tasks_filepath)


def get_next_not_finished_meta_task(tasks: dict[str, Any]) -> dict:
    """
    Gets the next meta level task that is not yet finished.
    """
    for task_id, task_info in tasks.items():
        if task_id.startswith("_"):
            continue
        if task_info["status"] == "NOT_FINISHED":
            return task_info
    return {}


def get_next_meta_task_filepath(meta_tasks_file_path: str) -> str:
    """
    This is utilized in the context of tasks_files.json, which defines the individual tasks files to be processed.
    """
    tasks = load_tasks_file(meta_tasks_file_path)
    next_task = get_next_not_finished_meta_task(tasks)
    if next_task:
        return next_task["path"]
    return ""


def is_all_meta_tasks_finished(meta_tasks_file_path: str) -> bool:
    tasks = load_tasks_file(meta_tasks_file_path)
    if get_next_not_finished_meta_task(tasks):
        return False
    return True


def mark_meta_task_as_finished(meta_tasks_file_path: str, meta_task_path: str) -> dict:
    tasks = load_tasks_file(meta_tasks_file_path)
    for task_id, task_info in tasks.items():
        if task_info["path"] == meta_task_path:
            task_info["status"] = "FINISHED"
            break
    with open(meta_tasks_file_path, "w") as file:
        json.dump(tasks, file, indent=4)
    return load_tasks_file(meta_tasks_file_path)


if __name__ == "__main__":
    tasks_file_path = "tasks.json"
    tasks = load_tasks_file(tasks_file_path)
    next_not_finished_task = get_next_not_finished_task_with_base_data_variant(tasks)
    logger.info(f"Next not finished task: {next_not_finished_task}")

    mark_variant_as_done(
        next_not_finished_task,
        next_not_finished_task["base_data_variant"],
        tasks_file_path,
    )
    logger.info("Marked variant as done for the first task.")
