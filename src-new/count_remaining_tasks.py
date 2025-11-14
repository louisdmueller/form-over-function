"""
This script goes through the meta-tasks file, finds the standard tasks and then goes through these individually to count how many tasks there are remaining to be done.
"""

from get_next_task_new import get_next_meta_task_filepath
from utils_new import load_config, parse_args
import json
import os
import requests

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
            print(f"Standard tasks file does not exist: {standard_tasks_filepath}. Skipping.")
            continue

        with open(standard_tasks_filepath, "r") as f:
            standard_tasks = json.load(f)

        for task in standard_tasks["tasks"]:
            done += len(task["done"])
            tbd += len(standard_tasks["base_data_variants"]) - len(task["done"])

    print(f"Amount of tasks already done: {done}/{done + tbd} ({(done / (done + tbd)) * 100:.2f}%)")

    # Write to homeassistant variable "input_text.done"
    """
    curl -X POST -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d '{"entity_id": "light.bettlampe"}' \
        http://homeassistant.local:8123/api/services/homeassistant/toggle > /dev/null 2>&1
    """

    ha_token = load_config("config.yml")["ha"]
    url = "https://vdqqtl5abn46wlveloj9adcvnpv7ngfw.ui.nabu.casa/api/services/counter/set_value"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ha_token}"}
    data = {
        "entity_id": "counter.done",
        "value": done
    }
    response = requests.post(url, headers=headers, json=data)
    
    data = {
        "entity_id": "counter.percent_done",
        "value": done / (done + tbd) * 100
    }
    response = requests.post(url, headers=headers, json=data)

    current_model = get_next_meta_task_filepath(args.meta_tasks_file)
    current_model = os.path.basename(current_model).replace(".json", "")
    current_model = current_model.replace("tasks_", "")
    print(f"Current model: {current_model}")
    url = "https://vdqqtl5abn46wlveloj9adcvnpv7ngfw.ui.nabu.casa/api/services/input_text/set_value"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ha_token}"}
    data = {
        "entity_id": "input_text.current_model",
        "value": current_model
    }
    response = requests.post(url, headers=headers, json=data)