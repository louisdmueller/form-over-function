import argparse

import pandas as pd
import yaml
from nltk import edit_distance


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
        default="",
        help="Path to the judge model.",
    )

    parser.add_argument(
        "--prompt_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the prompt model.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/chen-et-al/raw.json",
        help="Path to the data directory.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Path to the config file.",
    )

    args = parser.parse_args()
    return args


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_comparison_csv(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create a CSV file with the comparison results.
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
    comparison_df = add_length_column(comparison_df)
    comparison_df = add_edit_distance_column(comparison_df)

    comparison_df.to_csv(output_path, index=False)


def add_length_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a length column to the DataFrame.
    """
    df["Original->AAE character difference"] = df.apply(
        lambda x: len(x["Translated Answer (AAE)"]) - len(x["Original Answer"]),
        axis=1,
    )
    return df


def add_edit_distance_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a edit distance column to the DataFrame.
    """
    df["Original->AAE Edit distance"] = df.apply(
        lambda x: edit_distance(
            x["Original Answer"],
            x["Translated Answer (AAE)"],
        ),
        axis=1,
    )
    return df
