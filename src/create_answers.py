import json
import os
from model import get_model, GeminiModel

from google import genai

from huggingface_hub import login

from aae_translation import add_aae_to_df
from utils import (
    get_df_from_file,
    load_config,
    parse_args,
)

model_name = "gemini-1.5-flash"

args = parse_args()
config = load_config(args.config_path)
login(token=config["huggingface_hub_token"])

gemini = GeminiModel(
    model_name_or_path=model_name,
    api_key=config["gemini_key"],
)

# generate answers without AAE
with open("data/chen-et-al/data_with_aae_gpt4-1.json", "r") as f:
    data = [json.loads(line) for line in f.readlines()]

for entry in data:
    generated_data = {}
    prompt = entry["prompt"]
    text = gemini.prompt(
        message=prompt,
        num_generations=2,
    )
    generated_data["question_id"] = entry["question_id"]
    generated_data["question"] = entry["question"]
    generated_data["prompt"] = prompt
    generated_data["temperature"] = entry["temperature"]
    generated_data["model_id"] = entry["model_id"]
    generated_data["level"] = entry["level"]
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
    generated_data["answers_gpt"] = entry["answers"]
    with open(
        os.path.join(
            os.path.dirname(args.data_path),
            f"data_with_aae_{args.prompt_model_name_or_path}.json",
        ),
        "a",
    ) as f:
        f.write(json.dumps(generated_data) + "\n")

# data_df = get_df_from_file(args.data_path)
# data_directory = os.path.dirname(args.data_path)
# if not os.path.exists(f"{data_directory}/data_with_aae_{model_name}.json"):
#     data_df = add_aae_to_df(data_df, prompt_gen_model)
#     data_df.to_json(
#         f"{data_directory}/data_with_aae_{model_name}.json",
#         lines=True,
#         orient="records",
#     )
# else:
#     data_df = get_df_from_file(
#         os.path.join(data_directory, f"data_with_aae_{model_name}.json")
#     )