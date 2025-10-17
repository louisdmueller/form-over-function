from collections import defaultdict
import json
import os
import statistics
import argparse
from typing import Dict, Optional, Tuple

from numpy import extract
from pandas import DataFrame

from analyze_reasonings import analyze_reasonings_topic_model
import pandas as pd

from utils import remove_organization_from_hf_model_name


perturbations = ["default", "basic", "aae"]


def load_json_file(filepath: str) -> dict:
    """Load JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_names_from_data(data: dict) -> Tuple[str, str]:
    """Extract model names from the data, specifically from the first questions answers labels."""
    for key, questions in data.items():
        if key != "metadata" and questions:
            first_question = questions[0]
            model1 = first_question["answer1"]["label"]
            model2 = first_question["answer2"]["label"]
            better_model_name = model1 if model1 == "gpt-4.1" else model2
            worse_model_name = model2 if better_model_name == model1 else model1
            return (better_model_name, worse_model_name)
    return "Model A", "Model B"


def map_vote_to_value(
    vote: str, better_model_name: str, worse_model_name: str
) -> Optional[float]:
    """Map vote to numeric value
    1.0 for better_model_name,
    0.0 for worse_model_name,
    0.5 for tie
    """
    mapping = {better_model_name: 1.0, worse_model_name: 0.0, "TIE": 0.5}
    return mapping.get(vote)


def aggregate_voting_data(
    data: dict, better_model_name: str, worse_model_name: str
) -> Dict[str, Dict[str, float]]:
    """Extract and process question results for the given model pair."""
    question_results = {}

    for question_group_key in data.keys():
        if question_group_key == "metadata":
            continue

        question_group = data[question_group_key]

        for question_data in question_group:
            question_text = question_data["question"]
            extracted_answers = question_data["extracted_answers"]
            # TODO In the basic experiment answer files
            # (for example gpt-4.1-answers_basic.json), the model_name is gpt-4o-mini instead of gpt-4.1.
            # This is not intended and leads to a KeyError and unexpected results. Therefore, I am adding this hacky fix here.
            # It should be fixed in the files.
            extracted_answers = [
                better_model_name if ans == "gpt-4o-mini" else ans
                for ans in extracted_answers
            ]
            question_id = question_group_key

            if question_id not in question_results:
                question_results[question_id] = {
                    "question_text": question_text,
                    "all_votes": [],
                }

            for vote in extracted_answers:
                vote_value = map_vote_to_value(
                    vote, better_model_name, worse_model_name
                )
                if vote_value is not None:
                    question_results[question_id]["all_votes"].append(vote_value)

    final_results = {}
    for question_id, votes_data in question_results.items():
        all_votes = votes_data["all_votes"]
        if all_votes:
            better_model_avg = statistics.mean(all_votes)
            final_results[question_id] = {
                "question_text": votes_data["question_text"],
                f"{better_model_name}_avg": better_model_avg,
                f"{worse_model_name}_avg": 1.0 - better_model_avg,
                "winner": (
                    better_model_name
                    if better_model_avg > 0.5
                    else worse_model_name if better_model_avg < 0.5 else "tie"
                ),
                "total_votes": len(all_votes),
            }
    return final_results


def create_vote_counts_table(
    data: dict, better_model_name: str, worse_model_name: str
) -> DataFrame:
    """Create a DataFrame summarizing total votes for each answer order."""
    model1_name = data["0"][0]["answer1"]["label"]
    model2_name = data["0"][0]["answer2"]["label"]
    aggregated_data = defaultdict(
        lambda: {
            better_model_name: 0,
            worse_model_name: 0,
            "TIE": 0,
            "Unknown": 0,
            "total": 0,
        }
    )
    for question_id, entries in data.items():
        if question_id == "metadata":
            continue
        for entry in entries:
            answer_order = (
                entry["answer_order"]
                .replace("model1", model1_name)
                .replace("model2", model2_name)
            )
            answer_order = answer_order
            for answer in entry["extracted_answers"]:
                # TODO In the basic experiment answer files
                # (for example gpt-4.1-answers_basic.json), the model_name is gpt-4o-mini instead of gpt-4.1.
                # This is not intended and leads to a KeyError. Therefore, I am adding this hacky fix here.
                if answer == "gpt-4o-mini":
                    answer = better_model_name
                aggregated_data[answer_order][answer] += 1
                aggregated_data[answer_order]["total"] += 1

    rows = []
    for answer_order, counts in aggregated_data.items():
        rows.append(
            {
                "answer_order": answer_order,
                " ": "",
                better_model_name: counts[better_model_name],
                worse_model_name: counts[worse_model_name],
                "TIE": counts["TIE"],
                "": "",
                "Unknown": counts["Unknown"],
                "total": counts["total"],
            }
        )
    df = DataFrame(rows)
    df = df.sort_values(by=["answer_order"]).reset_index(drop=True)
    return df


def count_judgement_transitions_between_files(
    file1_results: Dict,
    file2_results: Dict,
    better_model_name: str,
    worse_model_name: str,
):
    """
    This function counts the following averaged outcomes:
    - How many times better_model_name wins in file1
    - How many times worse_model_name wins in file1
    - How often the tie occurs in file1
    - How often worse_model_name wins in file2, after better_model_name wins in file1
    - How often worse_model_name wins in file2, after worse_model_name wins in file1
    - How often better_model_name wins in file2, after worse_model_name wins in file1
    - How often better_model_name wins in file2, after better_model_name wins in file1
    - How often the outcome in file 1 changes in file 2
    """
    better_model_wins_file1 = 0
    worse_model_wins_file1 = 0
    ties_file1 = 0

    better_model_wins_file2_after_better = 0
    worse_model_wins_file2_after_better = 0
    better_model_wins_file2_after_worse = 0
    worse_model_wins_file2_after_worse = 0
    flips = 0

    for question_id in file1_results.keys():
        if question_id not in file2_results:
            continue

        file1_winner = file1_results[question_id]["winner"]
        file2_winner = file2_results[question_id]["winner"]

        if file1_winner == better_model_name:
            better_model_wins_file1 += 1
            if file2_winner == worse_model_name:
                worse_model_wins_file2_after_better += 1
                flips += 1
            elif file2_winner == "TIE":
                flips += 1
            else:
                better_model_wins_file2_after_better += 1
        elif file1_winner == worse_model_name:
            worse_model_wins_file1 += 1
            if file2_winner == better_model_name:
                better_model_wins_file2_after_worse += 1
                flips += 1
            elif file2_winner == "TIE":
                flips += 1
            else:
                worse_model_wins_file2_after_worse += 1
        else:
            ties_file1 += 1
            if file2_winner != "TIE":
                flips += 1

    return {
        "better_model_wins_file1": better_model_wins_file1,
        "worse_model_wins_file1": worse_model_wins_file1,
        "ties_file1": ties_file1,
        "better_model_wins_file2_after_better": better_model_wins_file2_after_better,
        "worse_model_wins_file2_after_better": worse_model_wins_file2_after_better,
        "better_model_wins_file2_after_worse": better_model_wins_file2_after_worse,
        "worse_model_wins_file2_after_worse": worse_model_wins_file2_after_worse,
        "flips": flips,
    }


def calculate_asr(outcomes: Dict[str, int]) -> float:
    """Calculate Attack Success Rate: how often better_model_name wins flip to worse_model_name wins."""
    asr = (
        outcomes["worse_model_wins_file2_after_better"]
        / outcomes["better_model_wins_file1"]
        if outcomes["better_model_wins_file1"] > 0
        else 0.0
    )
    return asr


def calculate_aasr(outcomes: Dict[str, int]) -> float:
    """Calculate Anti Attack Success Rate: how often worse_model_name wins after better_model_name wins."""
    aasr = (
        outcomes["better_model_wins_file2_after_worse"]
        / outcomes["worse_model_wins_file1"]
        if outcomes["worse_model_wins_file1"] > 0
        else 0.0
    )
    return aasr


def calculate_fr(outcomes: Dict[str, int]) -> float:
    """Calculate Flip Rate: how often the outcome in file 1 changes in file 2."""
    fr = (
        outcomes["flips"]
        / (
            outcomes["better_model_wins_file1"]
            + outcomes["worse_model_wins_file1"]
            + outcomes["ties_file1"]
        )
        if (
            outcomes["better_model_wins_file1"]
            + outcomes["worse_model_wins_file1"]
            + outcomes["ties_file1"]
        )
        > 0
        else 0.0
    )

    return fr


def calculate_cr(outcomes: Dict[str, int]) -> float:
    """Calculate Consistency Rate: how often the answer (worse or better) in file 1 remains the same in file 2. (Ties excluded)"""
    cr = (
        (
            outcomes["better_model_wins_file2_after_better"]
            + outcomes["worse_model_wins_file2_after_worse"]
        )
        / (outcomes["better_model_wins_file1"] + outcomes["worse_model_wins_file1"])
        if (outcomes["better_model_wins_file1"] + outcomes["worse_model_wins_file1"])
        > 0
        else 0.0
    )

    return cr


def extract_perturbation_style(file_data: dict) -> str:
    """Extract perturbation style from metadata in the file data."""
    for file_idx in [1, 2]:
        file_path = file_data.get("metadata", {}).get(f"data_{file_idx}_path", "")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        for perturbation in perturbations:
            if file_name.lower().endswith(perturbation):
                return perturbation
    return "default"


def extract_judge_model(file_data: dict) -> str:
    """Extract judge model from metadata in the file data."""
    return file_data.get("metadata", {}).get("judge_model", "unknown_judge_model")


def run_analysis_on_judgements(
    file1_data: dict,
    file2_data: dict,
    output_directory: str,
    better_model_name: Optional[str] = None,
    worse_model_name: Optional[str] = None,
):
    """Analyze two result files and calculate ASR."""

    if not better_model_name or not worse_model_name:
        better_model_name, worse_model_name = extract_model_names_from_data(file1_data)

    file1_results = aggregate_voting_data(
        file1_data, better_model_name, worse_model_name
    )
    file2_results = aggregate_voting_data(
        file2_data, better_model_name, worse_model_name
    )

    judge_model_name = remove_organization_from_hf_model_name(
        extract_judge_model(file1_data)
    )

    output_directory = os.path.join(
        output_directory,
        (
            better_model_name
            + "_vs_"
            + remove_organization_from_hf_model_name(worse_model_name)
        ),
        judge_model_name,
    )
    for style_perturbation in perturbations:
        os.makedirs(os.path.join(output_directory, style_perturbation), exist_ok=True)

    create_vote_counts_table(file1_data, better_model_name, worse_model_name).to_excel(
        os.path.join(
            output_directory,
            perturbation_style_1 := extract_perturbation_style(file1_data),
            f"total_votes_file.xlsx",
        ),
        index=False,
    )
    create_vote_counts_table(file2_data, better_model_name, worse_model_name).to_excel(
        os.path.join(
            output_directory,
            perturbation_style_2 := extract_perturbation_style(file2_data),
            f"total_votes_file.xlsx",
        ),
        index=False,
    )

    outcomes = count_judgement_transitions_between_files(
        file1_results, file2_results, better_model_name, worse_model_name
    )

    result = {
        "asr": calculate_asr(outcomes),
        "aasr": calculate_aasr(outcomes),
        "fr": calculate_fr(outcomes),
        "cr": calculate_cr(outcomes),
        "v1": outcomes["better_model_wins_file1"],
        "v2": outcomes["worse_model_wins_file1"],
        "ties": outcomes["ties_file1"],
    }
    pd.DataFrame([result]).to_excel(
        os.path.join(
            output_directory,
            f"metrics_{perturbation_style_1}_vs_{perturbation_style_2}.xlsx",
        ),
        index=False,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ASR (Attack Success Rate) between two JSON result files for any model pair"
    )
    parser.add_argument(
        "--file1",
        default="/home/hd/hd_hd/hd_go226/projects/research-project/data/judgements/GPT4.1-vs-gemini-1.5-flash/Llama-3.3-70B-Instruct/merged_data.json",
        type=str,
        help="Path to the first JSON results file",
    )
    parser.add_argument(
        "--file2",
        default="/home/hd/hd_hd/hd_go226/projects/research-project/data/judgements/Llama-3.3-70B-Instruct---gpt-4.1_basic-vs-gemini-1.5-flash/merged_data.json",
        type=str,
        help="Path to the second JSON results file",
    )
    parser.add_argument(
        "--better_model_name",
        type=str,
        default=None,
        help="Name of the first model (auto-detected if not provided)",
    )
    parser.add_argument(
        "--worse_model_name",
        type=str,
        default=None,
        help="Name of the second model (auto-detected if not provided)",
    )

    parser.add_argument(
        "--analyze_reasonings",
        action="store_true",
        help="If set, analyze reasonings using topic modeling",
        default=False,
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        default="data/analysis_results",
        help="Directory to save the output files",
    )

    args = parser.parse_args()

    # Output into the same directory as file1 or file2
    file1_data = load_json_file(args.file1)
    file2_data = load_json_file(args.file2)

    if args.analyze_reasonings:
        analyze_reasonings_topic_model(
            file2_data,
            args.better_model_name,
            args.worse_model_name,
            args.output_directory,
        )
    run_analysis_on_judgements(
        file1_data,
        file2_data,
        args.better_model_name,
        args.worse_model_name,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
