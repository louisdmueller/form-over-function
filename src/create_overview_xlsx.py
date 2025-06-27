"""
This script takes all "merged_data.json" files in the "data/judgements" directory,
computes ASR for each file, and creates an overview Excel file with the results.
The rows are the individual judgment models, the columns are the models and the
items are the ASR values and the V1 Values.
Model columns are sorted by the strength of the model.

The output file is saved as "overview.xlsx" in the "data/judgements" directory.
"""

from collections import defaultdict
import os
import json
import pandas as pd
import xlsxwriter
from analyze_results import analyze_files

BASE_DIR = "data/judgements"

# first we need to find the two merged_data.json from the same model comparison subfolder
# we do this by finding pairs of folders where the folder names are the same except that one has
# _aae in it
pairs = []
subfolders = [f.path for f in os.scandir(BASE_DIR) if f.is_dir()]
for folder in subfolders:
    if "_aae" in folder:
        aae_folder_name = folder
        base_folder_name = folder.replace("_aae", "")

        # directory structure is like this: data/judgements/ModelA_aae-vs-ModelB/JudgeModelX
        #                                   data/judgements/ModelA-vs-ModelB/JudgeModelX
        # So we need to check if the aae comparison was done for the same judge models
        for aae_model_judge_directory in os.listdir(aae_folder_name):
            aae_judge_model_path = os.path.join(
                aae_folder_name, aae_model_judge_directory
            )
            base_judge_model_path = os.path.join(
                base_folder_name, aae_model_judge_directory
            )
            if os.path.exists(aae_judge_model_path) and os.path.exists(
                base_judge_model_path
            ):
                pairs.append((aae_judge_model_path, base_judge_model_path))


# now that we have the pairs, we can analyze the files
results = defaultdict(dict)
for pair in pairs:
    aae_folder, base_folder = pair
    aae_file = os.path.join(aae_folder, "merged_data.json")
    base_file = os.path.join(base_folder, "merged_data.json")

    if not os.path.exists(aae_file) or not os.path.exists(base_file):
        print(f"Skipping pair {pair} because one of the files does not exist.")
        continue

    def get_model_name(folder_path):
        """
        Extracts the model name from the folder path.
        The folder name is expected to be in the format 'ModelA_aae-vs-ModelB' or 'ModelA-vs-ModelB'.
        """
        if "_aae" in folder_path:
            return folder_path.split("-vs-")[0].split("/")[-1].replace("_aae", "")
        else:
            return folder_path.split("-vs-")[1].split("/")[0]

    aae_model_name = get_model_name(aae_folder)
    base_model_name = get_model_name(base_folder)

    # print(aae_model_name)
    # print(base_model_name)
    # analyze the files and get the ASR values
    asr, flips, v1 = analyze_files(
        base_file, aae_file
    )  # , aae_model_name, base_model_name)

    judge_model_name = os.path.basename(aae_folder)
    data = {
        "Judge Model": judge_model_name,
        "Base Model": base_model_name,
        "AAE Model": aae_model_name,
        "ASR": asr,
        "Flips": flips,
        "V1": v1,
    }

    # results[judge_model_name][base_model_name] = {
    #     "ASR": asr,
    #     "Flips": flips,
    # }
    # results[judge_model_name][aae_model_name] = {
    #     "ASR": asr,
    #     "Flips": flips,
    # }
    if judge_model_name != base_model_name:
        results[judge_model_name][base_model_name] = f"ASR: {asr:.4f} | V1: {v1}"
    # results[judge_model_name][aae_model_name] = asr

# Create a DataFrame from the results
df = pd.DataFrame.from_dict(results, orient="index")

# sort alphabetically and hope it works
df = df[sorted(df.columns)]

print(df)

df.to_excel(
    "overview.xlsx",
    sheet_name="ASR Overview",
    index_label="Judge Model",
)
