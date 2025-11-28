"""
This script goes through the meta-tasks file, finds the standard tasks and then goes through these individually to count how many tasks there are remaining to be done.
"""

from utils_new import load_config, parse_args
import json
import os
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parse_args()
    meta_tasks = load_config(args.meta_tasks_file)

    tbd = 0
    done = 0
    for meta_task_id, meta_task_info in meta_tasks.items():
        if meta_task_id.startswith("_"):
            continue  # skip comments

        standard_tasks_filepath = meta_task_info["path"]
        if not os.path.exists(standard_tasks_filepath):
            logger.info(f"Standard tasks file does not exist: {standard_tasks_filepath}. Skipping.")
            continue

        with open(standard_tasks_filepath, "r") as f:
            standard_tasks = json.load(f)

        for task in standard_tasks["tasks"]:
            done += len(task["done"])
            tbd += len(standard_tasks["base_data_variants"]) - len(task["done"])

    logger.info(f"Amount of tasks already done: {done}/{done + tbd} ({(done / (done + tbd)) * 100:.2f}%)")

