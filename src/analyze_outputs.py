"""
This script analyzes the answers of a judge model.
It does that by counting combinations of prompt style, order and answers.
"""

import json
from collections import defaultdict
import os
import pandas as pd


def map_answer_to_float(answer: str) -> float:
    if answer == "AAE Answer":
        return 1.0
    elif answer == "SAE Answer":
        return 0.0
    else:
        return 0.5


def get_average_vote(vote_sum: float, total_votes: int) -> str:
    result = vote_sum / total_votes
    if result < 0.5:
        return "SAE Answer"
    elif result > 0.5:
        return "AAE Answer"
    else:
        return "TIE"


def get_experiment_and_control_votes_length(
    votes_control: list[str], votes_experiment: list[str]
) -> dict[str, int]:
    """
    Compute the total number of votes:
        1. control_sae: the number of SAE votes in the control group
        2. experiment_sae: the number of SAE votes in the experiment group when the control also voted SAE
        3. control_aae: the number of AAE votes in the control group
        4. experiment_aae: the number of AAE votes in the experiment group when the control also voted AAE
    """
    control_sae = 0
    experiment_sae = 0
    experiment_aae = 0

    for vote_ctrl, vote_exp in zip(votes_control, votes_experiment):
        if vote_ctrl == "SAE Answer" or vote_ctrl == "TIE":
            control_sae += 1
            if vote_exp == "SAE Answer":
                experiment_sae += 1
            elif vote_exp == "AAE Answer":
                experiment_aae += 1
    return {
        "control_sae": control_sae,
        "experiment_sae": experiment_sae,
        "experiment_aae": experiment_aae,
    }


def compute_asr(
    bias_type: str, votes_control: list[str], votes_experiment: list[str]
) -> float:
    """
    Compute the ASR (Answer Style Ratio) for a given bias type.

    """
    votes_length = get_experiment_and_control_votes_length(
        votes_control, votes_experiment
    )

    if bias_type == "Alignment Bias":
        return votes_length["experiment_aae"] / votes_length["control_sae"]
    elif bias_type == "SAE Bias":
        return votes_length["experiment_sae"] / votes_length["control_sae"]
    else:
        raise ValueError(f"Did not expect bias: {bias_type}")


print()
subdirs = [
    directory
    for directory in os.listdir("data/chen-et-al/")
    if os.path.isdir(os.path.join("data/chen-et-al", directory))
]
for directory in subdirs:
    input_directory = os.path.join("data/chen-et-al", directory)
    file_path = os.path.join(input_directory, "merged_data.json")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    aggregated_data = defaultdict(lambda: {"AAE Answer": 0, "SAE Answer": 0, "TIE": 0, "Unknown": 0, "total": 0})

    average_vote_results = []

    for question_id, entries in raw_data.items():
        vote_sum = 0.0
        if question_id == "metadata":
            continue
        for index, entry in enumerate(entries):
            key = (entry["prompt_style"], entry["answer_order"])
            for answer in entry["extracted_answers"]:
                vote_sum += map_answer_to_float(answer)
                aggregated_data[key][answer] += 1
                aggregated_data[key]["total"] += 1
            if index == 1:
                average_vote_results.append(
                    {
                        "question_id": question_id,
                        "prompt_style": "sae",
                        "vote": get_average_vote(vote_sum, 6),
                    }
                )
                vote_sum = 0.0
        average_vote_results.append(
            {
                "question_id": question_id,
                "prompt_style": "aae",
                "vote": get_average_vote(vote_sum, 6),
            }
        )

    bias_types = ["Alignment Bias", "SAE Bias"]
    asr_results = {}
    for bias_type in bias_types:
        asr_results[bias_type] = compute_asr(
            bias_type,
            [
                entry["vote"]
                for entry in average_vote_results
                if entry["prompt_style"] == "sae"
            ],
            [
                entry["vote"]
                for entry in average_vote_results
                if entry["prompt_style"] == "aae"
            ],
        )

    rows = []
    for (prompt_style, answer_order), counts in aggregated_data.items():
        rows.append({
            "prompt_style": prompt_style,
            "answer_order": answer_order,
            " ": "",
            "AAE": counts["AAE Answer"],
            "SAE": counts["SAE Answer"],
            "TIE": counts["TIE"],
            "": "",
            "Unknown": counts["Unknown"],
            "total": counts["total"]
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(by=["prompt_style", "answer_order"]).reset_index(drop=True)

    print("Results for", os.path.basename(input_directory))
    print(df)
    print()
    print("ASR Results:")
    for bias_type, asr in asr_results.items():
        print(f"{bias_type}: {asr:.2f}")
    print()
