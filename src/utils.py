import argparse
import json
import os
import random
import string
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from nltk import download, edit_distance
from nltk.tokenize import word_tokenize

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
        "--output_path",
        type=str,
        default=None,
        help="Path to the output file."
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
        "--step_size",
        type=float,
        default=None,
        help="Step size to process the data in chunks. If not provided, the entire data will be processed. Can either be an integer or a float (e.g., 0.1 for 10% of the data).",
    )

    parser.add_argument(
        "--prompt_name",
        type=str,
        default="directly_answer_question_without_cot",
        help="Name of the prompt to use from the prompts.json file.",
    )

    parser.add_argument(
        "--aae",
        action="store_true",
        help="If set, the answers will be translated from SAE to AAE.",
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

    args = parser.parse_args()
    return args


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def random_id(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

def get_start_end_indices(start_index, step_size, data_length) -> Tuple[int, int]:
    """
    Calculate the start and end indices based on the provided start index and step size.
    If the start index is a float, it is treated as a percentage of the data length.
    The end index is calculated as start_index + step_size.
    """
    if start_index < 0 or step_size <= 0:
        raise ValueError("Start index must be non-negative and step size must be positive.")
    
    if isinstance(start_index, float):
        start_index = int(start_index * data_length)
    else:
        start_index = int(start_index)

    if step_size.is_integer():
        step_size = int(step_size)
    elif isinstance(step_size, float):
        if step_size < 0 or step_size > 1:
            raise ValueError("Step size as a float must be between 0 and 1.")
        step_size = int(step_size * data_length)

    end_index = start_index + step_size
    if end_index > data_length:
        end_index = data_length

    return start_index, end_index

def get_start_end_by_newest_file(
    data_file_path: str,
    step_size: float,
    length: int,
) -> Tuple[int, int]:
    """
    Get the start and end indices based on the newest file in the data directory.
    The newest file is determined by the last modified time.
    """
    data_directory = os.path.dirname(data_file_path)
    files = [f for f in os.listdir(data_directory) if f.endswith('.json') and f.startswith('results-')]
    if not files:
        print("No JSON files found in the directory.")
        if step_size is not None:
            start_index, end_index = get_start_end_indices(0, step_size, length)
            return start_index, end_index
        else:
            print("No step size provided, returning default indices (0, 64).")
            return 0, 64

    newest_file = max(
        files,
        key=lambda f: os.path.getmtime(os.path.join(data_directory, f))
    )
    
    with open(os.path.join(data_directory, newest_file), 'r') as f:
        data = json.load(f)
    
    indices = [int(indice) for indice in data.keys() if indice != "metadata"]
    if not indices:
        raise ValueError(
            "No indices found in the newest file. "
            f"Only {data.keys()} were found, but expected indices."
        )
    new_start_index = max(indices) + 1
    if step_size is not None:
        new_start_index, new_end_index = get_start_end_indices(
            new_start_index, step_size, length
        )
    else:
        new_end_index = new_start_index + len(indices)

    if new_start_index == new_end_index:
        raise ValueError(
            "The new start index is equal to the new end index. "
            "The data was probably already processed completely."
        )
    
    return new_start_index, new_end_index


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
        lambda x: len(x["Translated Answer (AAE)"]) - len(x["Original Answer"]) / len(x["Original Answer"]),
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
        ) / len(x["Original Answer"]),
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

def remove_slash_in_model_name(args: argparse.Namespace) -> None:
    """
    Remove the organization from the (huggingface) model name.
    This is necessary because the slash used in the model name is also 
    used in the path and this can cause issues when saving the output file.

    Example:    
        "meta-llama/Llama-3.3-70B-Instruct" will be replaced with 
        "Llama-3.3-70B-Instruct".
    """
    if (
        "/" in args.answer_generation_model_name_or_path
        and args.answer_generation_model_name_or_path in args.output_path
    ):
        model_name = args.answer_generation_model_name_or_path.split("/")[-1]
        args.output_path = args.output_path.replace(
            args.answer_generation_model_name_or_path, model_name
        )
