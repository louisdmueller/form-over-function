import json
import os

from tqdm import tqdm
from model import get_model

from utils import (
    load_config,
    parse_args,
)

model_name = "gemini-1.5-flash"

args = parse_args()
config = load_config(args.config_path)

model = get_model(
    model_name_or_path=args.answer_generation_model_name_or_path,
    config=config,
)

# generate answers without AAE
with open("data/chen-et-al/data_with_aae_gpt4-1.json", "r") as f:
    data = [json.loads(line) for line in f.readlines()]

for entry in tqdm(data):
    generated_data = {}
    # copy original data 
    generated_data["answers_gpt"] = entry["answers"]
    for key in [key for key in entry.keys() if key not in ["answers"]]:
        generated_data[key] = entry[key]

    prompt = entry["prompt"]
    text = model.generate_answers(
        message=prompt,
        num_generations=2,
    )

    generated_data["answers_gemini"] = {
        "answer1": {
            "answer": text[0],
            "answer_id": "NA",
        },
        "answer2": {
            "answer": text[1],
            "answer_id": "NA",
        },
    }
    
    with open(
        os.path.join(
            os.path.dirname(args.data_path),
            f"data_with_aae_{args.prompt_model_name_or_path}.json",
        ),
        "a",
    ) as f:
        f.write(json.dumps(generated_data) + "\n")
