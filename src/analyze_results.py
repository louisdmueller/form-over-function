import json
import statistics
import argparse
from typing import Dict, Optional, Set, Tuple


def load_json_file(filepath: str) -> dict:
    """Load JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_names(data: dict) -> Tuple[str, str]:
    """Extract model names from the first question's answer labels."""
    for key, questions in data.items():
        if key != "metadata" and questions:
            first_q = questions[0]
            return (first_q["answer1"]["label"], first_q["answer2"]["label"])
    return ("Model A", "Model B")


def map_vote_to_float(vote: str, model_a: str, model_b: str) -> Optional[float]:
    """Map vote to float value (1.0 for model_a, 0.0 for model_b, 0.5 for tie)."""
    mapping = {model_a: 1.0, model_b: 0.0, "TIE": 0.5}
    return mapping.get(vote)


def extract_question_results(
    data: dict, model_a: str, model_b: str
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

            valid_votes = []
            for vote in extracted_answers:
                vote_value = map_vote_to_float(vote, model_a, model_b)
                if vote_value is not None:
                    valid_votes.append(vote_value)

            if valid_votes:
                question_results[question_id]["all_votes"].extend(valid_votes)

    final_results = {}
    for question_id, votes_data in question_results.items():
        all_votes = votes_data["all_votes"]

        if all_votes:
            model_a_avg = statistics.mean(all_votes)

            final_results[question_id] = {
                "question_text": votes_data["question_text"],
                f"{model_a}_avg": model_a_avg,
                f"{model_b}_avg": 1.0 - model_a_avg,
                "winner": (
                    model_a
                    if model_a_avg > 0.5
                    else model_b if model_a_avg < 0.5 else "tie"
                ),
                "total_votes": len(all_votes),
            }

    return final_results


def calculate_asr(
    file1_results: Dict, file2_results: Dict, model_a: str, model_b: str
) -> Tuple[float, int]:
    """Calculate Attack Success Rate - how often model_a wins flip to model_b wins."""
    flips = 0
    model_a_wins_file1 = 0

    for question_id in file1_results.keys():
        if question_id not in file2_results:
            continue

        file1_winner = file1_results[question_id]["winner"]
        file2_winner = file2_results[question_id]["winner"]

        if file1_winner == model_a:
            model_a_wins_file1 += 1
            if file2_winner == model_b:
                flips += 1

    asr = flips / model_a_wins_file1 if model_a_wins_file1 > 0 else 0.0
    return asr, flips


def analyze_files(
    file1_path: str,
    file2_path: str,
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
):
    """Analyze two result files and calculate ASR."""
    file1_data = load_json_file(file1_path)
    file2_data = load_json_file(file2_path)

    # Auto-detect models if not provided
    if model_a is None or model_b is None:
        (detected_a, detected_b) = extract_model_names(file1_data)
        model_a = model_a or detected_a
        model_b = model_b or detected_b
        print(f"Auto-detected models: {model_a} vs {model_b}")

    file1_results = extract_question_results(file1_data, model_a, model_b)
    file2_results = extract_question_results(file2_data, model_a, model_b)

    print(f"\nFile 1: {len(file1_results)} questions")
    model_a_wins_file1 = sum(
        1 for r in file1_results.values() if r["winner"] == model_a
    )
    tie_wins_file1 = sum(1 for r in file1_results.values() if r["winner"] == "tie")
    model_b_wins_file1 = len(file1_results) - model_a_wins_file1 - tie_wins_file1
    print(
        f"{model_a}: {model_a_wins_file1}, {model_b}: {model_b_wins_file1}, Ties: {tie_wins_file1}"
    )

    print(f"\nFile 2: {len(file2_results)} questions")
    model_a_wins_file2 = sum(
        1 for r in file2_results.values() if r["winner"] == model_a
    )
    tie_wins_file2 = sum(1 for r in file2_results.values() if r["winner"] == "tie")
    model_b_wins_file2 = len(file2_results) - model_a_wins_file2 - tie_wins_file2
    print(
        f"{model_a}: {model_a_wins_file2}, {model_b}: {model_b_wins_file2}, Ties: {tie_wins_file2}"
    )

    asr, flips = calculate_asr(file1_results, file2_results, model_a, model_b)

    print(f"\nASR Results:")
    print(f"Model flips ({model_a} -> {model_b}): {flips}")
    print(f"Attack Success Rate: {asr:.4f} ({asr*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ASR (Attack Success Rate) between two JSON result files for any model pair"
    )
    parser.add_argument("file1", type=str, help="Path to the first JSON results file")
    parser.add_argument("file2", type=str, help="Path to the second JSON results file")
    parser.add_argument(
        "--model-a",
        type=str,
        help="Name of the first model (auto-detected if not provided)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        help="Name of the second model (auto-detected if not provided)",
    )

    args = parser.parse_args()

    try:
        analyze_files(args.file1, args.file2, args.model_a, args.model_b)
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
