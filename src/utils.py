import pandas as pd
import argparse

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
    parser = argparse.ArgumentParser(description="Argument parser for our research project.")

    parser.add_argument(
        "--judge_model_name_or_path",
        type=str,
        default="",
        help="Path to the judge model.",
    )

    parser.add_argument(
        "--prompt_model_name_or_path",
        type=str,
        default="",
        help="Path to the prompt model.",
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/chen-et-al/raw.json",
        help="Path to the data directory.",
    )

    args = parser.parse_args()
    return args