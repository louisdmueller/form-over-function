"""
This script lets a judge model evaluate answers from two different models
in batches.
"""

from dataclasses import dataclass
import json
import os
from typing import List, Tuple


from tqdm import tqdm
from model_new import get_model
from utils_new import (
    get_judgements_path,
    parse_args,
    load_config,
    get_prompt,
    prepare_question_with_intro,
    TimeBasedTimeoutHandler,
    get_file_path,
    read_data_file,
)

from create_overview_xlsx import create_excel_overview

# from get_next_task import get_next_valid_task, mark_task
from get_next_task_new import (
    get_next_not_finished_task_with_base_data_variant, 
    get_next_meta_task_filepath, 
    mark_variant_as_done
)
import logging
import csv

logger = logging.getLogger("generate_judgements_logger")
logger.setLevel(logging.DEBUG)

# Add a stream handler so debug messages appear on stdout
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.debug("Logger initialized.")

def load_data_lists(data_1_path: str, data_2_path: str) -> Tuple[list, list]:
    """
    Load the data lists from the given file paths and ensure they are valid.

    Args:
        data_1_path (str): Path to the first data file.
        data_2_path (str): Path to the second data file.

    Returns:
        tuple: A tuple containing two lists of data.
    """
    if data_1_path == data_2_path:
        raise ValueError(
            "The paths for data_1 and data_2 are the same. "
            "Please provide different paths."
        )
    data_1 = read_data_file(data_1_path)
    data_2 = read_data_file(data_2_path)

    if len(data_1) != len(data_2):
        raise ValueError(
            "The data lists have different lengths: "
            f"{len(data_1)} and {len(data_2)}. "
            "Please provide lists with the same length."
        )
    return data_1, data_2


@dataclass
class JudgementInput:
    input_text: str
    answer_dict: dict
    question: str
    question_style: str
    idx: int
    answer_position: str


def create_position_bias_mitigation_dict(
    answer_model_1: str, answer_model_2: str, name_model_1: str, name_model_2: str
) -> dict:
    """Create the answer dictionary with both orderings to mitigate position bias.

    Args:
        answer_model_1 (str): Answer from model 1.
        answer_model_2 (str): Answer from model 2.
        name_model_1 (str): Name of model 1.
        name_model_2 (str): Name of model 2.

    Returns:
        dict: A dictionary containing both orderings of answers.
    """
    return {
        "model1-first": {
            "answer1": {"text": answer_model_1, "label": name_model_1},
            "answer2": {"text": answer_model_2, "label": name_model_2},
            "tie": {"text": None, "label": "TIE"},
        },
        "model2-first": {
            "answer1": {"text": answer_model_2, "label": name_model_2},
            "answer2": {"text": answer_model_1, "label": name_model_1},
            "tie": {"text": None, "label": "TIE"},
        },
    }


def prepare_judgement_inputs(
    data_1: list,
    data_2: list,
    data_1_model_name: str,
    data_2_model_name: str,
    prompt_template: str,
    question_style_switching: bool,
    introductory_beginning: bool,
) -> List[JudgementInput]:
    """Prepare all inputs for model evaluation."""
    judgement_inputs = []

    for idx, (item_1, item_2) in tqdm(
        enumerate(zip(data_1, data_2)),
        desc="Creating batched inputs",
        unit="batch",
    ):
        question_1, question_2 = item_1["question"], item_2["question"]
        answer_1 = item_1["answers"]["answer1"]["answer"]
        answer_2 = item_2["answers"]["answer1"]["answer"]
        # name_1, name_2 = item_1["model_name"], item_2["model_name"]
        name_1, name_2 = data_1_model_name, data_2_model_name

        questions_to_use = (
            [(question_1, name_1), (question_2, name_2)]
            if question_1 != question_2 and question_style_switching
            else [(question_1, name_1)]
        )

        if introductory_beginning:
            question_1 = prepare_question_with_intro(
                question_1, item_1["answers"]["answer1"]["style"]
            )
            question_2 = prepare_question_with_intro(
                question_2, item_2["answers"]["answer1"]["style"]
            )

        answer_dict = create_position_bias_mitigation_dict(
            answer_1, answer_2, name_1, name_2
        )

        for question, model_name in questions_to_use:
            for answer_position in ["model1-first", "model2-first"]:
                input_text = prompt_template.format(
                    question=question,
                    answer1=answer_dict[answer_position]["answer1"]["text"],
                    answer2=answer_dict[answer_position]["answer2"]["text"],
                )

                judgement_inputs.append(
                    JudgementInput(
                        input_text=input_text,
                        answer_dict=answer_dict,
                        question=question,
                        question_style=model_name,
                        idx=idx,
                        answer_position=answer_position,
                    )
                )

    return judgement_inputs


def load_tasks_file(tasks_file: str) -> dict:
    with open(tasks_file, "r") as f:
        data = json.load(f)
    return data


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    if args.multi_tasks_mode:
        # Instead of providing a single task file we can also provide a file containing all the 
        # individual task files to be processed
        logger.info(f"Multi tasks mode enabled. Loading next meta task from: {args.meta_tasks_file}")
        task_filepath = get_next_meta_task_filepath(args.meta_tasks_file)
        tasks = load_tasks_file(task_filepath)
    else: 
        task_filepath = args.tasks_file
        tasks = load_tasks_file(task_filepath)
    logger.info(f"Loaded standard tasks from: {task_filepath}")

    judgement_files_directory = config["judgement_files_directory"]
    excel_output_directory = config["excel_output_directory"]

    timeout_handler = TimeBasedTimeoutHandler(threshold=400, logger=logger)

    prompt, prompt_template = get_prompt(
        tasks["prompting_parameters"]["prompt_file"],
        tasks["prompting_parameters"]["prompt_key"],
    )
    judge_model = get_model(tasks["judge_model_name"], config, tasks)

    while not timeout_handler.is_timeout_imminent():
        task = get_next_not_finished_task_with_base_data_variant(tasks)
        if task is None:
            logger.info("All tasks are finished. Aborting.")
            break
        
        # I call it base data since this data does not change throughout the tasks
        # Its always compared against, doesn't matter in which form (aae, basic, errors)
        base_data_model = tasks["base_data_model"]
        # Get the base data variant for the current task
        base_data_variant = task["base_data_variant"]
        base_data_filepath = get_file_path(base_data_model, base_data_variant)
        base_data = read_data_file(base_data_filepath)

        # This data is the data that is compared against
        # Since we only permutate the base data, we always load the unpermutated variant '""' here
        comp_data_model = task["compare_against"]
        comp_data_path = get_file_path(comp_data_model, "")
        comp_data = read_data_file(comp_data_path)

        logger.info(
            f"Processing task: base_data_model={base_data_model}, base_data_variant={base_data_variant}, compare_against={comp_data_model}"
        )

        judgement_inputs = prepare_judgement_inputs(
            base_data,
            comp_data,
            base_data_model,
            comp_data_model,
            prompt_template,
            tasks["prompting_parameters"].get("question_style_switching", False),
            tasks["prompting_parameters"].get("introductory_beginning", False),
        )
        input_texts = [item.input_text for item in judgement_inputs]
        system_prompts = [prompt] * len(input_texts)

        judgements = judge_model.generate(system_prompts, input_texts)
        extracted_answers = judge_model.get_response_data(judgements)

        # Process and save results
        file_content = {}
        file_content["metadata"] = create_judgement_metadata(tasks, base_data_filepath, comp_data_path)
        for i, judgement_input in enumerate(judgement_inputs):
            if judgement_input.idx not in file_content:
                file_content[judgement_input.idx] = []

            answer_preferences = extract_answer_labels(extracted_answers, i, judgement_input)

            file_content[judgement_input.idx].append(
                create_judgement_record(extracted_answers, i, judgement_input, answer_preferences)
            )
        output_filename = "judgements.json"
        output_dir = get_judgements_path(
            base_path=judgement_files_directory,
            base_model=tasks["base_data_model"],
            base_model_variant=base_data_variant,
            comp_model=task["compare_against"],
            judge_model=tasks["judge_model_name"],
        )
        output_file_path = os.path.join(output_dir, output_filename)
        with open(output_file_path, "w") as f:
            logger.info(f"Saving results to {output_file_path}")
            f.write(json.dumps(file_content, indent=4))

        tasks = mark_variant_as_done(task, base_data_variant, task_filepath)
    
    create_excel_overview(judgement_files_directory=judgement_files_directory, excel_output_directory=excel_output_directory)

    log_job_info(tasks, config, timeout_handler)


def log_job_info(tasks: dict, config: dict, timeout_handler: TimeBasedTimeoutHandler) -> None:
    # append job information to a csv file
    execution_time = timeout_handler.get_elapsed_time()
    # TODO: ressources used, files written, models used, ...
    job_info = {
        "job_id": os.getenv("SLURM_JOB_ID", "local_run"),
        "execution_time": execution_time,
        "base_data_model": tasks["base_data_model"],
        "judge_model": tasks["judge_model_name"],
    }

    csv_file_path = config["run_info_csv_path"]
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode="a", newline="") as csvfile:
        fieldnames = list(job_info.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(job_info)

def extract_answer_labels(extracted_answers, i, judgement_input):
    answer_preferences = []
    for answer in extracted_answers["extracted_answers"][i]:
        if (
                    answer
                    in judgement_input.answer_dict[judgement_input.answer_position]
                ):
            answer_preferences.append(
                        judgement_input.answer_dict[judgement_input.answer_position][
                            answer
                        ]["label"]
                    )
        else:
            answer_preferences.append("Unknown")
    return answer_preferences

def create_judgement_record(extracted_answers, i, judgement_input, answer_preferences):
    return {
                    "prompt_style": judgement_input.question_style,
                    "answer_order": judgement_input.answer_position,
                    "question": judgement_input.question,
                    "answer1": judgement_input.answer_dict[
                        judgement_input.answer_position
                    ]["answer1"],
                    "answer2": judgement_input.answer_dict[
                        judgement_input.answer_position
                    ]["answer2"],
                    "result": extracted_answers["output"][i],
                    "extracted_answers": answer_preferences,
                }

def create_judgement_metadata(tasks, base_data_filepath, comp_data_path):
    return {
            "judge_model": tasks["judge_model_name"],
            "data_1_path": base_data_filepath,
            "data_2_path": comp_data_path,
            "prompt_name": tasks["prompting_parameters"]["prompt_key"],
            "_comment": None,
            "model_parameters": tasks["model_parameters"],
            "sampling_parameters": tasks["sampling_parameters"],
            "vllm_parameters": tasks.get("vllm_parameters", {}),
            "prompting_parameters": tasks.get("prompting_parameters", {}),
        }



if __name__ == "__main__":
    main()
