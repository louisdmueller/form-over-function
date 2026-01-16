"""
This script lets a judge model evaluate answers from two different models
in batches.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from evaluation.create_overview_xlsx import create_excel_overview
from model import get_model
from utils.logging import log_job_info, setup_logger
from utils.tasks.get_next_task_new import (
    get_next_meta_task_filepath,
    get_next_not_finished_task_with_base_data_variant,
    mark_variant_as_done,
    mark_variant_as_submitted,
)
from utils.utils import (
    TimeBasedTimeoutHandler,
    get_file_path,
    get_judgements_path,
    get_prompt,
    load_config,
    parse_args,
    prepare_question_with_intro,
    read_jsonl_file,
)

logger = setup_logger(__name__, log_level=logging.DEBUG)

BATCH_QUEUE_DIR = Path("outputs/debug") / "batch_queue"
PROCESSED_QUEUE_DIR = BATCH_QUEUE_DIR / "processed"


def load_data_lists(data_1_path: str, data_2_path: str) -> tuple[list, list]:
    """
    Load the data lists from the given file paths and ensure they are valid.

    Args:
        data_1_path (str): Path to the first JSONL data file.
        data_2_path (str): Path to the second JSONL data file.

    Returns:
        tuple: A tuple containing two lists of data.
    """
    if data_1_path == data_2_path:
        raise ValueError(
            "The paths for data_1 and data_2 are the same. "
            "Please provide different paths."
        )
    data_1 = read_jsonl_file(data_1_path)
    data_2 = read_jsonl_file(data_2_path)

    if len(data_1) != len(data_2):
        raise ValueError(
            "The data files have different lengths: "
            f"{len(data_1)} and {len(data_2)}. "
            "Please provide files with the same length."
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
) -> dict[str, dict[str, dict[str, Optional[str]]]]:
    """Create the answer dictionary with both orderings to mitigate position bias. We 
    can later use this to present the answers in different orders to the judge model.

    Args:
        answer_model_1 (str): The answer provided by model 1.
        answer_model_2 (str): The answer provided by model 2.
        name_model_1 (str): The name of model 1.
        name_model_2 (str): The name of model 2.

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
    data_1_style: str,
    data_2_style: str,
    prompt_template: str,
    question_style_switching: bool,
    introductory_beginning: bool,
) -> list[JudgementInput]:
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


        if introductory_beginning:
            question_1 = prepare_question_with_intro(
                question_1, data_1_style
            )
            question_2 = prepare_question_with_intro(
                question_2, data_2_style
            )

        questions_to_use = (
            [(question_1, name_1), (question_2, name_2)]
            if question_1 != question_2 and question_style_switching
            else [(question_1, name_1)]
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

    batch_mode = (os.getenv("ANTHROPIC_MODE") or os.getenv("GEMINI_MODE") or "").lower()

    if args.multi_tasks_mode:
        # Instead of providing a single task file we can also provide a file containing all the
        # individual task files to be processed
        logger.info(
            f"Multi tasks mode enabled. Loading next meta task from: {args.meta_tasks_file}"
        )
        task_filepath = get_next_meta_task_filepath(args.meta_tasks_file)
        tasks = load_tasks_file(task_filepath)
    else:
        task_filepath = args.tasks_file
        tasks = load_tasks_file(task_filepath)
    logger.info(f"Loaded standard tasks from: {task_filepath}")

    judgement_files_directory = config["judgement_files_directory"]
    excel_output_directory = config["excel_output_directory"]

    timeout_handler = TimeBasedTimeoutHandler(threshold=400)
    produced_outputs = False

    if batch_mode == "retrieve":
        produced_outputs |= process_batch_queue(config)
        if produced_outputs:
            create_excel_overview(
                judgement_files_directory=judgement_files_directory,
                excel_output_directory=excel_output_directory,
            )
        log_job_info(tasks, config, timeout_handler.get_elapsed_time())
        return

    prompt, prompt_template = get_prompt(
        tasks["prompting_parameters"]["prompt_file"],
        tasks["prompting_parameters"]["prompt_key"],
    )

    while not timeout_handler.is_timeout_imminent():
        task = get_next_not_finished_task_with_base_data_variant(tasks)
        if task is None:
            logger.info("All tasks are finished. Aborting.")
            break

        base_data_model = tasks["base_data_model"]
        base_data_variant = task["base_data_variant"]
        base_data_filepath = get_file_path(base_data_model, base_data_variant)
        base_data = read_jsonl_file(base_data_filepath)

        comp_data_model = task["compare_against"]
        comp_data_variant = ""
        comp_data_path = get_file_path(comp_data_model, comp_data_variant)
        comp_data = read_jsonl_file(comp_data_path)

        logger.info(
            f"Processing task: base_data_model={base_data_model}, base_data_variant={base_data_variant}, compare_against={comp_data_model}"
        )

        judgement_inputs = prepare_judgement_inputs(
            comp_data,
            base_data,
            comp_data_model,
            base_data_model,
            comp_data_variant,
            base_data_variant,
            prompt_template,
            tasks["prompting_parameters"].get("question_style_switching", False),
            tasks["prompting_parameters"].get("introductory_beginning", False),
        )
        input_texts = [item.input_text for item in judgement_inputs]
        system_prompts = [prompt] * len(input_texts)

        job_id = f"{base_data_model}-{base_data_variant}-vs-{comp_data_model}-{int(timeout_handler.get_elapsed_time())}"
        judge_model = get_model(tasks["judge_model_name"], config, tasks, job_id=job_id)

        if getattr(judge_model, "mode", "") == "submit":
            persist_queue_entry(
                job_id=job_id,
                task=task,
                tasks_snapshot=tasks,
                tasks_filepath=task_filepath,
                base_data_model=base_data_model,
                base_data_variant=base_data_variant,
                comp_data_model=comp_data_model,
                comp_data_path=comp_data_path,
                base_data_filepath=base_data_filepath,
                judgement_inputs=judgement_inputs,
                judge_model_name=tasks["judge_model_name"],
            )
            judge_model.submit_batch(system_prompts, input_texts)
            tasks = mark_variant_as_submitted(task, base_data_variant, task_filepath)
            continue

        judgements = judge_model.generate(system_prompts, input_texts)
        model_response = judge_model.get_response_data(judgements)

        persist_and_write_judgements(
            tasks,
            base_data_filepath,
            comp_data_path,
            judgement_inputs,
            model_response,
            judgement_files_directory,
            base_data_variant,
            comp_data_model,
        )
        produced_outputs = True

        tasks = mark_variant_as_done(task, base_data_variant, task_filepath)

    if produced_outputs:
        create_excel_overview(
            judgement_files_directory=judgement_files_directory,
            excel_output_directory=excel_output_directory,
        )

    log_job_info(tasks, config, timeout_handler.get_elapsed_time())


def map_judge_decisions_to_labels(judge_decisions: list[list[str]], idx: int, model_name_to_answer_map: dict[str, dict[str, Optional[str]]]) -> list[str]:
    """
    Convert the judge model's extracted decision answers into the corresponding model names.
    
    Args:
        judge_decisions (list[list[str]]): Each inner list contains the judge model's decisions for a specific input.
        idx (int): The index of the current judgement input.
        model_name_to_answer_map (dict): A mapping from model names to their corresponding answer details.
        
    Returns:
        list[str]: A list of model names corresponding to the judge model's decisions.
    """
    answer_preferences = []
    for answer in judge_decisions[idx]:
        if answer in model_name_to_answer_map.keys():
            answer_preferences.append(
                model_name_to_answer_map[answer][
                    "label"
                ]
            )
        else:
            answer_preferences.append("Unknown")
    return answer_preferences


def create_judgement_record(model_response: dict[str, list[list[str]]], idx: int, judgement_input: JudgementInput, decision_labels: list[str]) -> dict:
    return {
        "prompt_style": judgement_input.question_style,
        "answer_order": judgement_input.answer_position,
        "question": judgement_input.question,
        "answer1": judgement_input.answer_dict[judgement_input.answer_position][
            "answer1"
        ],
        "answer2": judgement_input.answer_dict[judgement_input.answer_position][
            "answer2"
        ],
        "result": model_response["output"][idx],
        "extracted_decisions": decision_labels,
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


def persist_queue_entry(
    *,
    job_id: str,
    task: dict,
    tasks_snapshot: dict,
    tasks_filepath: str,
    base_data_model: str,
    base_data_variant: str,
    comp_data_model: str,
    comp_data_path: str,
    base_data_filepath: str,
    judgement_inputs: list[JudgementInput],
    judge_model_name: str,
) -> None:
    BATCH_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "task": task,
        "tasks_snapshot": tasks_snapshot,
        "tasks_filepath": tasks_filepath,
        "base_data_model": base_data_model,
        "base_data_variant": base_data_variant,
        "comp_data_model": comp_data_model,
        "comp_data_path": comp_data_path,
        "base_data_filepath": base_data_filepath,
        "judge_model_name": judge_model_name,
        "judgement_inputs": [ji.__dict__ for ji in judgement_inputs],
    }
    queue_file = BATCH_QUEUE_DIR / f"{job_id}.json"
    with queue_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Queued batch job %s to %s", job_id, queue_file)


def persist_and_write_judgements(
    tasks: dict,
    base_data_filepath: str,
    comp_data_path: str,
    judgement_inputs: list[JudgementInput],
    model_response: dict[str, list[list[str]]],
    judgement_files_directory: str,
    base_data_variant: str,
    comp_data_model: str,
) -> None:
    file_content = {}
    file_content["metadata"] = create_judgement_metadata(
        tasks, base_data_filepath, comp_data_path
    )
    for idx, judgement_input in enumerate(judgement_inputs):
        if judgement_input.idx not in file_content:
            file_content[judgement_input.idx] = []

        decision_labels = map_judge_decisions_to_labels(
            model_response["extracted_decisions"], idx, judgement_input.answer_dict[judgement_input.answer_position]
        )

        file_content[judgement_input.idx].append(
            create_judgement_record(
                model_response, idx, judgement_input, decision_labels
            )
        )
    output_filename = "judgements.json"
    output_dir = get_judgements_path(
        base_path=judgement_files_directory,
        base_model=tasks["base_data_model"],
        base_model_variant=base_data_variant,
        comp_model=comp_data_model,
        judge_model=tasks["judge_model_name"],
    )
    output_file_path = os.path.join(output_dir, output_filename)
    with open(output_file_path, "w") as f:
        logger.info(f"Saving results to {output_file_path}")
        f.write(json.dumps(file_content, indent=4))


def process_batch_queue(config: dict) -> bool:
    if not BATCH_QUEUE_DIR.exists():
        logger.info("No batch queue directory found at %s", BATCH_QUEUE_DIR)
        return False

    queue_files = sorted(BATCH_QUEUE_DIR.glob("*.json"))
    if not queue_files:
        logger.info("No queued batch jobs found in %s", BATCH_QUEUE_DIR)
        return False

    PROCESSED_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    produced_outputs = False

    for queue_file in queue_files:
        with queue_file.open("r", encoding="utf-8") as f:
            entry = json.load(f)

        job_id = entry["job_id"]
        tasks_snapshot = entry["tasks_snapshot"]
        task = entry["task"]
        tasks_filepath = entry["tasks_filepath"]
        base_data_variant = entry["base_data_variant"]
        comp_data_model = entry["comp_data_model"]

        judge_model = get_model(
            entry["judge_model_name"], config, tasks_snapshot, job_id=job_id
        )
        outputs = judge_model.retrieve_batch()
        model_response = judge_model.get_response_data(outputs)

        judgement_inputs = [JudgementInput(**item) for item in entry["judgement_inputs"]]

        persist_and_write_judgements(
            tasks_snapshot,
            entry["base_data_filepath"],
            entry["comp_data_path"],
            judgement_inputs,
            model_response,
            config["judgement_files_directory"],
            base_data_variant,
            comp_data_model,
        )

        mark_variant_as_done(task, base_data_variant, tasks_filepath)
        produced_outputs = True

        processed_path = PROCESSED_QUEUE_DIR / queue_file.name
        queue_file.rename(processed_path)
        logger.info("Processed batch job %s and archived to %s", job_id, processed_path)

    return produced_outputs


if __name__ == "__main__":
    main()
