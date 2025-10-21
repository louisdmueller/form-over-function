import argparse
import json
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
        "--job_end_time",
        type=int,
        # required=True,
        default=int(time.time()) + 1800,
        help="Unix timestamp indicating when the job will end.",
    )

    parser.add_argument(
        "--tasks_file",
        type=str,
        default="tasks.json",
        help="Path to the tasks JSON file.",
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

def get_file_path(data_name: str, data_variant: str = "") -> str:
    data_name = (
        data_name + "-answers" if not data_name.endswith("-answers") else data_name
    )
    if not data_variant:
        return f"data/generated_answers/{data_name}.json"
    return f"data/generated_answers/{data_name}_{data_variant}.json"

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
    if style == "aave":
        return (
            "Hey, I'm stuck on this question and was wonderin' if you could help me out. So, the question go: "
            + str(question)
        )
    elif style == "sae":
        return (
            "Hi there, I'm a bit stuck on a question and was wondering if you could help me out. Here's the question: "
            + str(question)
        )
    else:
        print(f"Unknown style '{style}' for question. No introductory text added.")
        return question
    
def get_judgements_path(base_model, base_model_variant, comp_model, judge_model):
    # e.g. data/judgements/gpt-4_aae
    base_model_dir = base_model + f"_{base_model_variant}" if base_model_variant else base_model
    path = os.path.join("data", "judgements", base_model_dir)

    # e.g. data/judgements/gpt-4_aae/vs_gemini-1.5-flash
    comp_model_dir = "vs_" + comp_model
    path = os.path.join(path, comp_model_dir)

    # e.g. data/judgements/gpt-4_aae/vs_gemini-1.5-flash/gps-oss-120b
    path = os.path.join(path, judge_model)

    os.makedirs(path, exist_ok=True)
    return path

class TimeBasedTimeoutHandler():
    """
    A handler to gracefully shut down the script when the remaining time is below a certain threshold.
    This is useful to ensure that the script can finish processing/saving before the job is terminated.
    """

    def __init__(self, end_time: int, threshold: int = 300):
        """
        Initialize the TimeBasedTimeoutHandler.

        Args:
            end_time (int): The end time as a unix timestamp.
            threshold (int): The threshold in seconds to consider the timeout imminent. Default is 300 seconds (5 minutes).
        """
        print("Initializing TimeBasedTimeoutHandler.")
        self.end_time = end_time
        self.threshold = threshold

    def is_timeout_imminent(self) -> bool:
        """
        Check if the timeout is imminent based on the remaining time.

        Returns:
            bool: True if the timeout is imminent, False otherwise.
        """
        current_time = int(time.time())
        remaining_time = self.end_time - current_time
        print(f"Remaining time: {remaining_time} seconds")
        return remaining_time <= self.threshold
    
