import argparse
import random
import string

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
        type=float,
        default=0.0,
        help="Index to start processing the data from. Can be an integer or a float (e.g., 0.1 for 10% of the data).",
    )
    
    parser.add_argument(
        "--end_index",
        type=float,
        default=None,
        help="Index to stop processing the data at. Can be an integer or a float (e.g., 0.5 for 50% of the data). If None, process all data.",
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

    args = parser.parse_args()
    return args


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def random_id(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

def get_start_end_indices(start_index, end_index, data_length) -> tuple:
    # If end_index is not provided, it it set to None, since
    # the data length is not initialized yet.
    # We need to manually set it to the length of the dataframe
    if end_index is None or end_index > data_length:
        end_index = data_length
    if start_index < 0 or end_index < 0 or start_index >= data_length or end_index > data_length:
        raise ValueError("Invalid start or end index.")
    
    if start_index.is_integer():
        start_index = int(start_index)
    else:
        start_index = int(start_index * data_length)
    if end_index.is_integer():
        end_index = int(end_index)
    else:
        end_index = int(end_index * data_length)

    return start_index, end_index
    


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
