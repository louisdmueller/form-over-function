import argparse
import json
import logging
import os
import time
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(
        description="Argument parser for our research project."
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Path to the configuration YAML file.",
    )

    parser.add_argument(
        "--tasks_file",
        type=str,
        default="tasks.json",
        help="Path to the tasks JSON file.",
    )

    parser.add_argument(
        "--multi_tasks_mode", action="store_true", help="Enable multi-tasks mode."
    )

    parser.add_argument(
        "--meta_tasks_file",
        type=str,
        default="tasks_files/meta_tasks.json",
        help="Path to the meta tasks JSON file.",
    )

    args = parser.parse_args()
    return args


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_prompt(prompt_file: str, prompt_key: str) -> tuple[str, str]:
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    return prompts[prompt_key]["system"], prompts[prompt_key]["template"]


def load_available_answer_files() -> set:
    import os

    answer_files = set()
    base_dir = "data/generated_answers/"

    for file_name in os.listdir(base_dir):
        if file_name.endswith("-answers.json"):
            model_name = file_name.replace("-answers.json", "")
            answer_files.add(model_name)

    return answer_files


def get_full_model_variant(data_name: str, data_variant: str) -> str:
    if not data_variant:
        return data_name
    return f"{data_name}_{data_variant}"


def get_file_path(data_name: str, data_variant: str = "") -> str:
    data_name = (
        data_name + "-answers" if not data_name.endswith("-answers") else data_name
    )
    full_model_variant = get_full_model_variant(data_name, data_variant)
    return f"data/generated_answers/{full_model_variant}.json"


def read_data_file(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def prepare_question_with_intro(
    question: str,
    style: str,
) -> str:
    """
    Add introductory text to question based on style.
    Args:
        question (str): The original question.
        style (str): The style of the question, e.g., 'aave' or 'sae'.

    """
    logger = logging.getLogger()
    if style in ["aave", "aae"]:
        logger.debug("Using AAVE/AAE style for question.")
        return (
            "Hey, I'm stuck on this question and was wonderin' if you could help me out. So, the question go: "
            + str(question)
        )
    elif style in ["sae", ""]:
        logger.debug("Using SAE/Default style for question.")
        return (
            "Hi there, I'm a bit stuck on a question and was wondering if you could help me out. Here's the question: "
            + str(question)
        )
    elif style == "simple":
        logger.debug("Using Simple style for question.")
        return (
            "Hi, I'm stuck on a question and need your help. Here it is: "
            + str(question)
        )
    else:
        raise Exception(f"Unknown style '{style}' for question. No introductory text added.")


def get_judgements_path(
    base_path, base_model, base_model_variant, comp_model, judge_model
):
    # e.g. data/judgements/gpt-4_aae
    base_model_dir = (
        base_model + f"_{base_model_variant}" if base_model_variant else base_model
    )
    path = os.path.join(base_path, base_model_dir)

    # e.g. data/judgements/gpt-4_aae/vs_gemini-1.5-flash
    comp_model_dir = "vs_" + comp_model
    path = os.path.join(path, comp_model_dir)

    # e.g. data/judgements/gpt-4_aae/vs_gemini-1.5-flash/gps-oss-120b
    if "/" in judge_model:
        judge_model = judge_model.split("/")[-1]
    path = os.path.join(path, judge_model)

    os.makedirs(path, exist_ok=True)
    return path


def remove_organization_from_hf_model_name(model_name: str) -> str:
    """
    Remove the organization from the (huggingface) model name.
    This is necessary because the slash used in the model name is also
    used in the path and this can cause issues when saving the output file.

    Example:
        "meta-llama/Llama-3.3-70B-Instruct" will be replaced with
        "Llama-3.3-70B-Instruct".
    """
    return model_name.split("/")[-1] if "/" in model_name else model_name


class TimeBasedTimeoutHandler:
    """
    A handler to gracefully shut down the script when the remaining time is below a certain threshold.
    This is useful to ensure that the script can finish processing/saving before the job is terminated.
    """

    # TODO: passing the logger as argument is a bit clunky, refactor this
    def __init__(self, threshold: int = 300):
        """
        Initialize the TimeBasedTimeoutHandler.

        Args:
            end_time (int): The end time as a unix timestamp.
            threshold (int): The threshold in seconds to consider the timeout imminent. Default is 300 seconds (5 minutes).
        """
        self.logger = logging.getLogger()
        self.logger.info("Initializing TimeBasedTimeoutHandler.")

        job_start_time = os.getenv("SLURM_JOB_START_TIME")
        if job_start_time is None:
            self.logger.warning("SLURM_JOB_START_TIME environment variable not set. ")
            job_start_time = int(time.time())
        else:
            job_start_time = int(job_start_time)

        job_end_time = os.getenv("SLURM_JOB_END_TIME")
        if job_end_time is None:
            self.logger.warning(
                "SLURM_JOB_END_TIME environment variable not set. "
                "Timeout handling may not work as expected."
            )
            job_end_time = int(time.time()) + 1800  # Default to 30 minutes from now
        else:
            job_end_time = int(job_end_time)
        self.logger.info(f"Job end time from SLURM: {time.ctime(job_end_time)}")

        self.start_time = job_start_time
        self.end_time = job_end_time
        self.threshold = threshold

    def is_timeout_imminent(self) -> bool:
        """
        Check if the timeout is imminent based on the remaining time.

        Returns:
            bool: True if the timeout is imminent, False otherwise.
        """
        current_time = int(time.time())
        remaining_time = self.end_time - current_time
        self.logger.info(f"Remaining time: {remaining_time} seconds")
        timeout_imminent = remaining_time <= self.threshold
        if timeout_imminent:
            self.logger.warning(
                f"Timeout imminent! Only {remaining_time} seconds remaining."
            )
        return timeout_imminent

    def get_elapsed_time(self) -> int:
        """
        Get the elapsed time since the start of the job.

        Returns:
            int: The elapsed time in seconds.
        """
        current_time = int(time.time())
        elapsed_time = current_time - self.start_time
        return elapsed_time
