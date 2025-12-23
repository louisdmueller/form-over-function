"""
This script prompts the user for a parameter name and a new value, infers the correct
datatype (bool, int, float, null, list, dict), and updates that parameter for all tasks
in the JSON files within the 'tasks_files' directory.
"""

import json
import os

def parse_value(val: str):
    """Infer type: bool, int, float, null, list, dict, or fallback to string."""
    lowered = val.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if lowered == "null" or lowered == "none":
        return None

    try:
        return int(val)
    except ValueError:
        pass

    try:
        return float(val)
    except ValueError:
        pass

    # JSON structures (list, dict, etc.)
    try:
        return json.loads(val)
    except Exception:
        pass

    # Fallback to string
    return val


if __name__ == "__main__":
    param_name = input("Enter the parameter name to change: ")
    raw_value = input(f"Enter the new value for '{param_name}': ")
    new_value = parse_value(raw_value)

    for file in os.listdir("tasks_files"):
        if not file.endswith(".json"):
            continue

        if file == "standard_tasks_template.json":
            continue
        if file == "meta_tasks.json":
            continue

        file_path = os.path.join("tasks_files", file)

        with open(file_path, "r") as f:
            task_data = json.load(f)

        if "." in param_name:
            parts = param_name.split(".")
            current = task_data
            for idx, key in enumerate(parts):
                if idx == len(parts) - 1:
                    current[key] = new_value
                else:
                    if key not in current or not isinstance(current[key], dict):
                        current[key] = {}
                    current = current[key]
        else:
            print(
                f"Failed to update tasks in {file}: parameter name must contain at least one dot for nested parameters."
            )
            continue

        with open(file_path, "w") as f:
            json.dump(task_data, f, indent=4)

    print(f"All tasks have been updated: set '{param_name}' to {new_value!r}.")

