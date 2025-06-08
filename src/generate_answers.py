"""
This script generates two files with answers to the prompts in the data file.
 - The first file contains the answers in Standard American English (SAE),
 - The second file (if AAE translation is requested) contains the answers translated to African American English (AAE)
"""

import json
import os
from tqdm import tqdm

from model import get_model
from utils import load_config, parse_args, random_id
from aae_translation import add_aae_to_df

args = parse_args()
config = load_config(args.config_path)

if (
    "/" in args.answer_generation_model_name_or_path
    and args.answer_generation_model_name_or_path in args.output_path
):
    # If the model name contains a slash, it is a Hugging Face model
    # but slash is also used in the path
    # Since the name is often used in the output file name,
    # we replace the slash with an underscore to avoid issues
    model_name = args.answer_generation_model_name_or_path.split("/")[-1]
    args.output_path = args.output_path.replace(
        args.answer_generation_model_name_or_path, model_name
    )

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
for entry in tqdm(data, desc=desc):
    generated_data = {}
    # copy original data to new file
    for key in [key for key in entry.keys()]:
        generated_data[key] = entry[key]
    generated_data["model_name"] = args.answer_generation_model_name_or_path

    prompt = entry["prompt"]
    num_generations=2
    do_sample = False
    temperature = entry.get("temperature", None) if do_sample else None
    responses = answer_generation_model.generate(
        system_prompts=[""],
        input_texts=[prompt],
        max_output_tokens=len(prompt) + 50,
        num_generations=num_generations,
        do_sample=do_sample,
        temperature=temperature,
    )

    if isinstance(responses, list) and len(responses) == 1:
        # .generate() returns a list of a batches of strings,
        # so we need to extract the first batch
        responses = responses[0]

    generated_data["answers"] = {}
    for i in range(num_generations):
        generated_data["answers"][f"answer{i + 1}"] = {
            "answer": responses[i],
            "answer_id": random_id(8),
        }

    generated_data["metadata"] = {
        "generation_model_name": args.answer_generation_model_name_or_path,
        "temperature": temperature,
        "do_sample": do_sample,
    }

    # append to output file in order to not lose data in case of an error
    with open(args.output_path, "a") as f:
        f.write(json.dumps(generated_data) + "\n")

    if args.aae:
        # If AAE translation is requested, add AAE translations to the answers
        import pandas as pd

        df = pd.DataFrame([generated_data])
        df = add_aae_to_df(df, prompt_gen_model)

        generated_data["question"] = df["question_aae"].iloc[0]

        for i in range(num_generations):
            generated_data["answers"][f"answer{i + 1}"]["answer"] = df["answers"].iloc[0][f"answer{i + 1}_aae"]
            generated_data["answers"][f"answer{i + 1}"]["answer_id"] = generated_data["answers"][f"answer{i + 1}"]["answer_id"]

        generated_data["metadata"] = {
            "translation_model_name": args.prompt_model_name_or_path,
            "generation_model_name": args.answer_generation_model_name_or_path,
        }

        # Save in a different file to make project structure cleaner
        with open(args.output_path.replace(".json", "_aae.json"), "a") as f_aae:
            f_aae.write(json.dumps(generated_data) + "\n")
