"""
This file takes merged_data.json and converts it into a single XLSX file.
The XLSX file will have the following columns:
id  question	question_style	answer1	answer_style1	answer2	answer_style2	model_answer1	extracted_answer1	model_answer2	extracted_answer2	model_answer3	extracted_answer3
"""

import json
from pathlib import Path
import xlsxwriter
import argparse

def export_merged_json_to_xlsx(
    json_path,
    xlsx_path,
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
            if "result" in entry and entry["result"]:
                if len(entry["result"]) > 0:
                    row["model_answer1"] = entry["result"][0]
                if len(entry["result"]) > 1:
                    row["model_answer2"] = entry["result"][1]
                if len(entry["result"]) > 2:
                    row["model_answer3"] = entry["result"][2]
            if "extracted_answers" in entry and entry["extracted_answers"]:
                for i, ans in enumerate(entry["extracted_answers"]):
                    if i < 3:
                        row[f"extracted_answer{i+1}"] = ans
            rows.append(row)

    # Write XLSX using xlsxwriter
    workbook = xlsxwriter.Workbook(str(xlsx_path))
    worksheet = workbook.add_worksheet()

    worksheet.set_row(0, 30)  # Set header row height

    # Set column width for model_answer1, model_answer2, model_answer3
    for col_idx, col_name in enumerate(columns):
        if col_name in ["question", "question_style", "answer_style1", "answer_style2"]:
            worksheet.set_column(col_idx, col_idx, 20)
        if col_name in ["answer1", "answer2"]:
            worksheet.set_column(col_idx, col_idx, 30)
        if col_name in ["model_answer1", "model_answer2", "model_answer3"]:
            worksheet.set_column(col_idx, col_idx, 100)

    wrap_format = workbook.add_format({'text_wrap': True}) 
    worksheet.freeze_panes(1, 7) # Freeze header row and first 7 columns

    # Write header
    for col_idx, col_name in enumerate(columns):
        worksheet.write(0, col_idx, col_name, wrap_format)

    # Write data rows
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, col_name in enumerate(columns):
            worksheet.write(row_idx, col_idx, row.get(col_name, ""), wrap_format)

    workbook.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert merged_data.json to XLSX.")
    parser.add_argument("json_path", type=Path, help="Path to the merged_data.json file")
    parser.add_argument("xlsx_path", type=Path, help="Path to the output XLSX file")
    args = parser.parse_args()

    export_merged_json_to_xlsx(args.json_path, args.xlsx_path)