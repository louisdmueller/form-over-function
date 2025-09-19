"""
This script takes all "merged_data.json" files in the "data/judgements" directory,
computes ASR for each file, and creates an overview Excel file with the results.
The rows are the individual judgment models, the columns are the models and the
items are the ASR values the V1, V2 and Vties values.
Model columns are sorted by the strength of the model.

The output file is saved as "overview_{experiment_name}.xlsx" in the "data/judgements" directory.
"""

from collections import defaultdict
import os
from typing import Dict, List, Tuple
import pandas as pd

from analyze_results import (
    load_json_file,
    run_analysis_on_judgements,
)


JUDGEMENTS_DIR = "data/judgements"


def find_folder_pairs(
    experiments: List[str] = ["aae", "basic"]
) -> Dict[str, List[Tuple[str, str]]]:
    """Find pairs of folders for comparison, one with experiment suffix, one without."""
    pairs = {experiment: [] for experiment in experiments}

    # Get base folders (without experiment suffixes)
    base_folders = [
        f for f in os.listdir(JUDGEMENTS_DIR) if not any(s in f for s in experiments)
    ]

    for base_folder in base_folders:
        for experiment in experiments:
            style_folder = f"{base_folder}_{experiment}"
            experiment_path = os.path.join(JUDGEMENTS_DIR, style_folder)
            base_path = os.path.join(JUDGEMENTS_DIR, base_folder)

            if not os.path.exists(experiment_path):
                continue

            # Find matching subfolders and judge folders
            for subfolder in os.listdir(experiment_path):
                experiment_subfolder_path = os.path.join(experiment_path, subfolder)
                base_subfolder_path = os.path.join(base_path, subfolder)

                if not os.path.exists(base_subfolder_path):
                    continue

                for judge_folder in os.listdir(experiment_subfolder_path):
                    experiment_judge_path = os.path.join(
                        experiment_subfolder_path, judge_folder
                    )
                    base_judge_path = os.path.join(base_subfolder_path, judge_folder)

                    if os.path.exists(base_judge_path):
                        pairs[experiment].append(
                            (experiment_judge_path, base_judge_path)
                        )

    return pairs


def apply_formatting_to_worksheet(writer: pd.ExcelWriter, sheet_name: str) -> None:
    """Apply formatting to the given worksheet."""
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    wrap_format = workbook.add_format(  # type: ignore
        {"text_wrap": True, "align": "center", "valign": "vcenter"}
    )

    max_col = worksheet.dim_colmax + 1
    for col_num in range(max_col + 1):
        worksheet.set_column(col_num, col_num, 20, wrap_format)


def process_pair_results(
    pairs: List[Tuple[str, str]], metric: str
) -> Dict[str, Dict[str, str]]:
    """Process analysis results for all folder pairs for a specific metric."""
    results = defaultdict(dict)

    for experiment_folder, base_folder in pairs:
        experiment_file = os.path.join(experiment_folder, "merged_data.json")
        base_file = os.path.join(base_folder, "merged_data.json")

        if not (os.path.exists(experiment_file) and os.path.exists(base_file)):
            print(f"Skipping pair - missing files: {experiment_folder}, {base_folder}")
            continue

        # Run analysis and extract results
        analysis_results = run_analysis_on_judgements(
            load_json_file(base_file),
            load_json_file(experiment_file),
            output_directory="data/analysis_results",
        )

        judge_model_name = os.path.basename(experiment_folder)
        worse_model_name = os.path.basename(os.path.dirname(base_folder))

        if judge_model_name != worse_model_name:
            results[judge_model_name][worse_model_name] = (
                f"{metric.upper()}: {analysis_results[metric]:.2f}\n"
                f"V1: {analysis_results['v1']} V2: {analysis_results['v2']} "
                f"Vties: {analysis_results['ties']}"
            )

    return results


def write_results_to_excel(
    style_pairs: List[Tuple[str, str]], output_filename: str
) -> None:
    """Write the analysis results to an Excel file with multiple metrics."""
    metrics = ["asr", "aasr", "fr", "cr"]

    with pd.ExcelWriter(f"{output_filename}.xlsx", engine="xlsxwriter") as writer:
        for metric in metrics:
            metric_results = defaultdict(dict)
            style_results = process_pair_results(style_pairs, metric)
            for judge, comparisons in style_results.items():
                metric_results[judge].update(comparisons)

            # Create and format DataFrame
            df = pd.DataFrame.from_dict(metric_results, orient="index")
            df = df[sorted(df.columns)] if not df.empty else df

            df.to_excel(
                writer,
                sheet_name=f"{metric.upper()}-Overview",
                index_label="Judge Model",
            )
            apply_formatting_to_worksheet(writer, f"{metric.upper()}-Overview")


def main() -> None:
    """Main function to execute the script."""
    os.makedirs("data/analysis_results", exist_ok=True)
    folder_pairs = find_folder_pairs(experiments=["aae", "basic"])
    for experiment, style_pairs in folder_pairs.items():
        if style_pairs:
            write_results_to_excel(
                style_pairs,
                output_filename=os.path.join(
                    "data/analysis_results", f"overview_{experiment}"
                ),
            )


if __name__ == "__main__":
    main()
