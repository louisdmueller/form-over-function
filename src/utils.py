import argparse
import ctypes
import json
from multiprocessing import Value
import os
import random
import string
import time
from typing import Any, Dict, List

import pandas as pd
import yaml
from nltk import download, edit_distance
from nltk.tokenize import word_tokenize

import signal

download("punkt_tab", quiet=True)


def get_df_from_file(file_path: str) -> pd.DataFrame:
    """
    Read a Jsonl file and return a DataFrame.
    """
    df = pd.read_json(file_path, lines=True)
    return df


def read_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a Jsonl file and return a list of dicts.
    """
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def write_file(file_path: str, data: List[Dict[str, Any]]) -> None:
    """
    Write a list of dicts to a Jsonl file.
    """
    with open(file_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def parse_args() -> argparse.Namespace:
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(
        description="Argument parser for our research project."
    )

    parser.add_argument(
        "--judge_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Path to the judge model.",
    )

    parser.add_argument(
        "--prompt_model_name_or_path",
        type=str,
        default="gpt-4.1",
        help="Path to the prompt model.",
    )

    parser.add_argument(
        "--answer_generation_model_name_or_path",
        type=str,
        default="gemini-1.5-flash",
        help="Path to the answer generation model.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/chen-et-al/raw_data.json",
        help="Path to the raw data.",
    )

    parser.add_argument(
        "--data_1_path",
        type=str,
        default="data/chen-et-al/raw_data-new_format.json",
        help="Path to the raw data.",
    )

    parser.add_argument(
        "--data_2_path",
        type=str,
        default="data/chen-et-al/raw_data-new_format.json",
        help="Path to the raw data.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Path to the config file.",
    )

    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to the output file."
    )

    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="Comment to add to the output file metadata.",
    )

    # provide start and end index to process a subset of the data
    # useful for debugging and testing
    parser.add_argument(
        "--start_index",
        # type=float,
        default=0.0,
        help="Index to start processing the data from. Can be an integer or a float (e.g., 0.1 for 10% of the data).",
    )

    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of the data to process, between 0 and 1. Default is 1 (all data). For debugging and testing.",
    )

    parser.add_argument(
        "--prompt_name",
        type=str,
        default="directly_answer_question_without_cot",
        help="Name of the prompt to use from the prompts.json file.",
    )

    parser.add_argument(
        "--question_style_switching",
        action="store_true",
        help="If set, the question style will be switched between the two models.",
    )

    parser.add_argument(
        "--introductionary_beginning",
        action="store_true",
        help="If set, an introductionary beginning will be added to the prompts.",
    )

    parser.add_argument(
        "--merge_path",
        type=str,
        default=None,
        help="Path to the directory of the individual JSON files to merge.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the directory containing the judge files.",
    )

    parser.add_argument(
        "--end_time",
        type=int,
        help="End time as unix timestamp.",
    )

    args = parser.parse_args()
    return args


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def random_id(length=8):
    chars = string.ascii_letters + string.digits
    return "".join(random.choices(chars, k=length))


def get_start_index_by_newest_file(
    data_directory: str,
) -> int:
    """
    Get the start index based on the newest file in the data directory.
    The newest file is determined by the last modified time.

    Args:
        data_directory (str): The directory containing the judgement data files.

    Returns:
        int: The start index to continue processing from.
    """
    files = [
        f
        for f in os.listdir(data_directory)
        if f.endswith(".json") and f.startswith("results-")
    ]
    if not files:
        print("No JSON files found in the directory.")
        return 0

    newest_file = max(
        files, key=lambda f: os.path.getmtime(os.path.join(data_directory, f))
    )

    with open(os.path.join(data_directory, newest_file), "r") as f:
        data = json.load(f)

    indices = [int(indice) for indice in data.keys() if indice != "metadata"]
    if not indices:
        raise ValueError(
            "No indices found in the newest file. "
            f"Only {data.keys()} were found, but expected indices."
        )
    new_start_index = max(indices) + 1

    return new_start_index


def create_comparison_csv(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a CSV file to compare the translated answers to the original ones.
    """
    comparison_df = pd.DataFrame(
        {
            "Prompt": df["question"],
            "Original Answer": df["answers"].apply(lambda a: a["answer1"]["answer"]),
            "Translated Answer (AAE)": df["answers"].apply(
                lambda a: a.get("answer1_permutated", "N/A")
            ),
        }
    )
    comparison_df = _add_length_column(comparison_df)
    comparison_df = _add_edit_distance_column(comparison_df)
    comparison_df = _add_type_token_ratio_column(comparison_df)

    comparison_df.to_csv(output_path, index=False)


def _add_length_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a column that indicates the normalized length difference between the AAE answers and the original answers.
    """
    df["Original->AAE character difference"] = df.apply(
        lambda x: len(x["Translated Answer (AAE)"])
        - len(x["Original Answer"]) / len(x["Original Answer"]),
        axis=1,
    )
    return df


def _add_edit_distance_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a column that indicates the normalized minimal edit distance from the original answers to the AAE answers.
    """
    df["Original->AAE Edit distance"] = df.apply(
        lambda x: edit_distance(
            x["Original Answer"],
            x["Translated Answer (AAE)"],
        )
        / len(x["Original Answer"]),
        axis=1,
    )
    return df


def _add_type_token_ratio_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a type-token ratio column to the DataFrame.
    The type-token ratio is the number of unique words divided by the total number of words.
    """
    df["Type-Token Ratio Original"] = df.apply(
        lambda x: len(set(word_tokenize(x["Original Answer"])))
        / len(word_tokenize(x["Original Answer"])),
        axis=1,
    )
    df["Type-Token Ratio AAE"] = df.apply(
        lambda x: len(set(word_tokenize(x["Translated Answer (AAE)"])))
        / len(word_tokenize(x["Translated Answer (AAE)"])),
        axis=1,
    )
    return df


def sanitize_output_path(output_path: str, model_name: str) -> str:
    """
    If the model name contains a slash and is part of the output path,
    replace the model name in the output path with a sanitized version
    that has the slash removed.

    Args:
        output_path (str): The original output path.
        model_name (str): The model name that may contain a slash.

    Returns:
        str: The sanitized output path.
    """
    if "/" in model_name and model_name in output_path:
        sanitized_model_name = remove_organization_from_hf_model_name(model_name)
        output_path = output_path.replace(model_name, sanitized_model_name)
    return output_path


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


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name by replacing slashes with underscores.
    This is useful for creating file names that include the model name.

    Args:
        model_name (str): The original model name.

    Returns:
        str: The sanitized model name.
    """
    return model_name.replace("/", "_") if "/" in model_name else model_name


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


class SlurmTimeoutHandler:
    """
    A handler to gracefully shut down the script when a SIGUSR1 signal is received.
    This is useful to ensure that the script can finish processing/saving before the job is terminated.
    """

    def __init__(self):
        print("Initializing SlurmTimeoutHandler.")
        self.timeout_imminent = Value(ctypes.c_bool, False)
        self._setup_signals()

    def _setup_signals(self):
        """
        Set up the signal handler for SIGUSR1.
        """
        signal.signal(signal.SIGUSR1, self._handle_timeout_signal)

    def _handle_timeout_signal(self, signum, frame):
        """Handle the SIGUSR1 signal by setting the timeout_imminent flag to True."""
        print(f"Received signal {signum} in PID {os.getpid()}")
        self.timeout_imminent.value = True

    def is_timeout_imminent(self) -> bool:
        """
        Check if the timeout is imminent.
        """
        print(f"Timeout imminent: {self.timeout_imminent.value}")
        return self.timeout_imminent.value

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