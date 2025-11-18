"""
Reads in all tasks defined in meta_tasks.json and creates empty standard tasks files for each meta task.
The user then has to fill in the sampling parameters in each standard tasks file.
"""

import json
import os

with open("tasks_files/meta_tasks.json", "r") as f:
    meta_tasks = json.load(f)

with open("tasks_files/standard_tasks_template.json", "r") as f:
    standard_tasks_template = json.load(f)

for meta_task_id, meta_task_info in meta_tasks.items():
    if meta_task_id.startswith("_"):
        continue  # skip comments
    if meta_task_info["status"] == "FINISHED":
        continue  # skip finished meta tasks
    if os.path.exists(meta_task_info["path"]):
        print(f"Standard tasks file already exists: {meta_task_info['path']}. Skipping creation.")
        continue
    standard_tasks_filepath = meta_task_info["path"]
    with open(standard_tasks_filepath, "w") as f:
        json.dump(standard_tasks_template, f, indent=4)
    print(f"Created standard tasks file: {standard_tasks_filepath}")