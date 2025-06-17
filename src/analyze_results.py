from collections import defaultdict
import json
import statistics
import argparse
from typing import Dict, Optional, Tuple

from pandas import DataFrame


def load_json_file(filepath: str) -> dict:
    """Load JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_names(data: dict) -> Tuple[str, str]:
    """Extract model names from the first question's answer labels."""
    for key, questions in data.items():
        if key != "metadata" and questions:
            first_question = questions[0]
            return (
                first_question["answer1"]["label"],
                first_question["answer2"]["label"],
            )
    return "Model A", "Model B"


def map_vote_to_value(
    vote: str, better_model: str, worse_model: str
) -> Optional[float]:
    """Map vote to numeric value
    1.0 for better_model,
    0.0 for worse_model,
    0.5 for tie
    """
    mapping = {better_model: 1.0, worse_model: 0.0, "TIE": 0.5}
    return mapping.get(vote)


def extract_question_results(
    data: dict, better_model: str, worse_model: str
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
            question_id = question_group_key

            if question_id not in question_results:
                question_results[question_id] = {
                    "question_text": question_text,
                    "all_votes": [],
                }

            for vote in extracted_answers:
                vote_value = map_vote_to_value(vote, better_model, worse_model)
                if vote_value is not None:
                    question_results[question_id]["all_votes"].append(vote_value)

    final_results = {}
    for question_id, votes_data in question_results.items():
        all_votes = votes_data["all_votes"]
        if all_votes:
            better_model_avg = statistics.mean(all_votes)
            final_results[question_id] = {
                "question_text": votes_data["question_text"],
                f"{better_model}_avg": better_model_avg,
                f"{worse_model}_avg": 1.0 - better_model_avg,
                "winner": (
                    better_model
                    if better_model_avg > 0.5
                    else worse_model if better_model_avg < 0.5 else "tie"
                ),
                "total_votes": len(all_votes),
            }
    return final_results


def get_total_votes_table(data: dict, better_model: str, worse_model: str) -> DataFrame:
    """Create a DataFrame summarizing total votes for each answer order."""
    aggregated_data = defaultdict(
        lambda: {better_model: 0, worse_model: 0, "TIE": 0, "Unknown": 0, "total": 0}
    )
    for question_id, entries in data.items():
        if question_id == "metadata":
            continue
        for entry in entries:
            answer_order = entry["answer_order"]
            for answer in entry["extracted_answers"]:
                aggregated_data[answer_order][answer] += 1
                aggregated_data[answer_order]["total"] += 1

    rows = []
    for answer_order, counts in aggregated_data.items():
        rows.append(
            {
                "answer_order": answer_order,
                " ": "",
                better_model: counts[better_model],
                worse_model: counts[worse_model],
                "TIE": counts["TIE"],
                "": "",
                "Unknown": counts["Unknown"],
                "total": counts["total"],
            }
        )
    df = DataFrame(rows)
    df = df.sort_values(by=["answer_order"]).reset_index(drop=True)
    return df


def calculate_asr(
    file1_results: Dict, file2_results: Dict, better_model: str, worse_model: str
) -> Tuple[float, int]:
    """Calculate Attack Success Rate: how often better_model wins flip to worse_model wins."""
    flips = 0
    better_model_wins_file1 = 0

    for question_id in file1_results.keys():
        if question_id not in file2_results:
            continue

        file1_winner = file1_results[question_id]["winner"]
        file2_winner = file2_results[question_id]["winner"]

        if file1_winner == better_model:
            better_model_wins_file1 += 1
            if file2_winner == worse_model:
                flips += 1

    asr = flips / better_model_wins_file1 if better_model_wins_file1 > 0 else 0.0
    return asr, flips


def analyze_files(
    file1_path: str,
    file2_path: str,
    better_model: Optional[str] = None,
    worse_model: Optional[str] = None,
):
    """Analyze two result files and calculate ASR."""
    file1_data = load_json_file(file1_path)
    file2_data = load_json_file(file2_path)

    if better_model is None or worse_model is None:
        (detected_a, detected_b) = extract_model_names(file1_data)
        better_model = better_model or detected_a
        worse_model = worse_model or detected_b
        print(f"Models: {better_model} vs {worse_model}")

    file1_results = extract_question_results(file1_data, better_model, worse_model)
    file2_results = extract_question_results(file2_data, better_model, worse_model)

    for i, (results, data) in enumerate(
        [(file1_results, file1_data), (file2_results, file2_data)]
    ):
        wins_a = sum(1 for r in results.values() if r["winner"] == better_model)
        ties = sum(1 for r in results.values() if r["winner"] == "tie")
        wins_b = len(results) - wins_a - ties

        print(f"\nFile {i}: {len(results)} questions")
        print(f"{better_model}: {wins_a}, {worse_model}: {wins_b}, Ties: {ties}")
        print(f"\nVote counts for File {i}:")
        print(get_total_votes_table(data, better_model, worse_model))

    asr, flips = calculate_asr(file1_results, file2_results, better_model, worse_model)
    print(f"\nASR Results:")
    print(f"Flips ({better_model} -> {worse_model}): {flips}")
    print(f"Attack Success Rate: {asr:.4f} ({asr*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ASR (Attack Success Rate) between two JSON result files for any model pair"
    )
    parser.add_argument("--file1", type=str, help="Path to the first JSON results file")
    parser.add_argument(
        "--file2",
        type=str,
        help="Path to the second JSON results file",
        required=False,
    )
    parser.add_argument(
        "--better_model",
        type=str,
        help="Name of the first model (auto-detected if not provided)",
    )
    parser.add_argument(
        "--worse_model",
        type=str,
        help="Name of the second model (auto-detected if not provided)",
    )

    args = parser.parse_args()

    analyze_files(args.file1, args.file2, args.better_model, args.worse_model)


if __name__ == "__main__":
    main()
