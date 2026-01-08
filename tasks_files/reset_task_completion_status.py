"""
This script goes through all tasks and resets their completion status to NOT_FINISHED.
"""

import json
import os


if __name__ == "__main__":
    ans = input("Are you sure you want to reset the completion status of all tasks to NOT_FINISHED? (y/n): ")
    if ans.lower() == 'n':
        exit("Operation cancelled by user.")
    for file in os.listdir("tasks_files"):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join("tasks_files", file)
        with open(file_path, 'r') as f:
            task_data = json.load(f)
        for task in task_data.get("tasks", []):
            task["status"] = "NOT_FINISHED"
            task["done"] = []
        with open(file_path, 'w') as f:
            json.dump(task_data, f, indent=4)
    print("All task completion statuses have been reset to NOT_FINISHED.")