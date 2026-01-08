"""
This script is a helper script for organizing meta level and standard tasks.
Meta level tasks: defined in meta_tasks.json. Contains all the paths to the standard tasks files.
Standard tasks: defined in individual tasks files. Contains the actual file-model combinations with parameters to be processed.

The script goes into meta_tasks.json and gets the first meta level task that is not yet finished.
It then loads its corresponding standard tasks file and checks if all its subtasks are finished.
If all subtasks are finished, it marks the meta level task as "FINISHED" in meta_tasks.json.
It then submits a new SLURM job array for the next meta level task that is not yet finished.
If no meta level tasks are left, it exits.
"""
from utils.tasks.get_next_task_new import (
    get_next_meta_task_filepath,
    is_all_meta_tasks_finished,
    is_all_subtasks_finished,
    load_tasks_file,
    mark_meta_task_as_finished,
)
from utils.utils import parse_args

if __name__ == "__main__":
    args = parse_args()
    meta_tasks_filepath = args.meta_tasks_file

    # go in to standard-tasks file of the next meta level task
    next_meta_task_filepath = get_next_meta_task_filepath(meta_tasks_filepath)
    standard_tasks = load_tasks_file(next_meta_task_filepath)
    # check if all subtasks are finished
    if is_all_subtasks_finished(standard_tasks):
        # if so, mark meta task as finished
        mark_meta_task_as_finished(meta_tasks_filepath, next_meta_task_filepath)

    check = is_all_meta_tasks_finished(meta_tasks_filepath)
    exit(0 if check else 1)

