"""
This file takes merged_data.json and converts it into a single CSV file.
The CSV file will have the following columns:
id  question	question_style	answer1	answer_style1	answer2	answer_style2	model_answer1	extracted_answer1	model_answer2	extracted_answer2	model_answer3	extracted_answer3
"""

import json
import csv
from pathlib import Path

def export_merged_json_to_csv(
    json_path,
    csv_path,
    columns=[
        "id", "question", "question_style",
        "answer1", "answer_style1",
        "answer2", "answer_style2",
        "model_answer1", "extracted_answer1",
        "model_answer2", "extracted_answer2",
        "model_answer3", "extracted_answer3"
    ]
):
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove metadata if present
    data = {k: v for k, v in data.items() if not k == "metadata"}

    rows = []
    for id_, entries in data.items():
        # Each id_ has a list of 2 entries (model1-first, model2-first)
        for entry in entries:
            row = {
                "id": id_,
                "question": entry.get("question", ""),
                "question_style": entry.get("prompt_style", ""),
                "answer1": entry.get("answer1", {}).get("text", ""),
                "answer_style1": entry.get("answer1", {}).get("label", ""),
                "answer2": entry.get("answer2", {}).get("text", ""),
                "answer_style2": entry.get("answer2", {}).get("label", ""),
                "model_answer1": "",
                "extracted_answer1": "",
                "model_answer2": "",
                "extracted_answer2": "",
                "model_answer3": "",
                "extracted_answer3": "",
            }
            # Try to extract model answers and extracted answers from result/extracted_answers
            # The result field is a list of strings (usually 1 element)
            if "result" in entry and entry["result"]:
                row["model_answer1"] = entry["result"][0]
            # if "result" in entry and entry["result"]:
                row["model_answer2"] = entry["result"][1]
            # if "result" in entry and entry["result"]:
                row["model_answer3"] = entry["result"][2]
            
            if "extracted_answers" in entry and entry["extracted_answers"]:
                # If extracted_answers is a list, fill as many as possible
                for i, ans in enumerate(entry["extracted_answers"]):
                    if i < 3:
                        row[f"extracted_answer{i+1}"] = ans
            rows.append(row)

    # Write CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    # Example usage
    json_path = Path(__file__).parent.parent / "data/GPT4.1-vs-Mistral-7B-Instruct/merged_data.json"
    csv_path = Path(__file__).parent.parent / "data/GPT4.1-vs-Mistral-7B-Instruct/merged_data.csv"
    export_merged_json_to_csv(json_path, csv_path)