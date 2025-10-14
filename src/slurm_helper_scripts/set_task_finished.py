import json
import sys

def set_task_finished(model_name: str, compare_against: str) -> None:
    """Mark a specific task as FINISHED in tasks.json"""
    
    with open("tasks.json", "r") as f:
        tasks = json.load(f)
    
    # Find and update the specific task
    task_found = False
    if model_name in tasks["tasks"]:
        for task in tasks["tasks"][model_name]:
            if task["compare_against"] == compare_against:
                task["status"] = "FINISHED"
                task_found = True
                print(f"Task marked as FINISHED: {model_name} comparing against {compare_against}")
                break
    
    if not task_found:
        print(f"Task not found: {model_name} comparing against {compare_against}")
        print(f"Available models: {list(tasks['tasks'].keys())}")
        if model_name in tasks["tasks"]:
            print(f"Available compare_against values for {model_name}: {[task['compare_against'] for task in tasks['tasks'][model_name]]}")
        return
    
    # Save the updated tasks
    with open("tasks.json", "w") as f:
        json.dump(tasks, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python set_task_finished.py <model_name> <compare_against>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    compare_against = sys.argv[2]
    
    set_task_finished(model_name, compare_against)