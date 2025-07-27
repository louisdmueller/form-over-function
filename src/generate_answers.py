"""
This script generates two files with answers to the prompts in the data file.
 - The first file contains the answers in Standard American English (SAE),
 - The second file (if AAE translation is requested) contains the answers translated to African American English (AAE)
"""

import json
import os
from tqdm import tqdm
from model import get_model
from utils import load_config, parse_args, random_id, remove_slash_in_model_name
from aae_translation import add_aae_to_df

args = parse_args()
config = load_config(args.config_path)

remove_slash_in_model_name(args)

if os.path.exists(args.output_path):
    print(
        f"Output file {args.output_path} already exists. Exiting to avoid overwriting."
    )
    exit(1)

if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

answer_generation_model = get_model(
    model_name_or_path=args.answer_generation_model_name_or_path,
    config=config,
)
if args.aae:
    prompt_gen_model = get_model(
        model_name_or_path=args.prompt_model_name_or_path,
        config=config,
    )

with open(args.data_path, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

desc = "Generating SAE answers" if not args.aae else "Generating SAE and AAE answers"

# batch generate answers for all prompts
prompts = [entry["prompt"] for entry in data]
num_generations = 2
do_sample = False
temperatures = [entry.get("temperature", None) if do_sample else None for entry in data]

responses_batch = answer_generation_model.generate(
    system_prompts=[""] * len(prompts),
    input_texts=prompts,
    max_output_tokens=max([len(p) for p in prompts]) + 50,
    num_generations=num_generations,
    do_sample=do_sample,
    temperature=None,
    **config,
)

generated_data_list = []
for idx, entry in enumerate(data):
    generated_data = {key: entry[key] for key in entry.keys()}
    generated_data["model_name"] = args.answer_generation_model_name_or_path

    responses = responses_batch[idx]
    if isinstance(responses, list) and len(responses) == 1:
        responses = responses[0]

    generated_data["answers"] = {}
    for i in range(num_generations):
        generated_data["answers"][f"answer{i + 1}"] = {
            "answer": responses[i],
            "answer_id": random_id(8),
        }

    generated_data["metadata"] = {
        "generation_model_name": args.answer_generation_model_name_or_path,
        "temperature": temperatures[idx],
        "do_sample": do_sample,
    }
    generated_data_list.append(generated_data)

    with open(args.output_path, "a") as f:
        f.write(json.dumps(generated_data) + "\n")

if args.aae:
    import pandas as pd

    df = pd.DataFrame(generated_data_list)
    df = add_aae_to_df(df, prompt_gen_model)

    for idx, generated_data in enumerate(generated_data_list):
        generated_data["question"] = df["question_aae"].iloc[idx]
        for i in range(num_generations):
            generated_data["answers"][f"answer{i + 1}"]["answer"] = df["answers"].iloc[
                idx
            ][f"answer{i + 1}_aae"]
            # answer_id stays the same

        generated_data["metadata"] = {
            "translation_model_name": args.prompt_model_name_or_path,
            "generation_model_name": args.answer_generation_model_name_or_path,
        }

        with open(args.output_path.replace(".json", "_aae.json"), "a") as f_aae:
            f_aae.write(json.dumps(generated_data) + "\n")
