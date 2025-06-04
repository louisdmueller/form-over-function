import json
import os
from tqdm import tqdm

from model import get_model
from utils import (
    load_config,
    parse_args,
    random_id
)

args = parse_args()
config = load_config(args.config_path)

if os.path.exists(args.output_path):
    print(f"Output file {args.output_path} already exists. Exiting to avoid overwriting.")
    exit(1)

model = get_model(
    model_name_or_path=args.answer_generation_model_name_or_path,
    config=config,
)

# generate answers without AAE
with open(args.data_path, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

for entry in tqdm(data, desc="Generating SAE answers"):
    generated_data = {}
    # copy original data to new file
    for key in [key for key in entry.keys() if key not in ["answers"]]:
        generated_data[key] = entry[key]
    generated_data["original_answers"] = entry["answers"]

    prompt = entry["prompt"]
    text = model.query_model(
        message=prompt,
        num_generations=2,
    )

    generated_data["answers_gemini"] = {
        "answer1": {
            "answer": text[0],
            "answer_id": random_id(),
        },
        "answer2": {
            "answer": text[1],
            "answer_id": random_id(),
        },
    }
    
    # append to output file in order to not lose data in case of an error
    with open(args.output_path, "a") as f:
        f.write(json.dumps(generated_data) + "\n")
