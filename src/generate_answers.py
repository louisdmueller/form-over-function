"""
This script generates a file with answers to the prompts in the data file
"""

import json
import os

from model import get_model
from utils.utils import (
    load_config,
    parse_args,
    random_id,
    read_jsonl_file,
    sanitize_output_path,
)


def main():
    """Main function to generate answers."""
    args = parse_args()
    config = load_config(args.config_path)

    data = read_jsonl_file(args.data_path)

    args.output_path = sanitize_output_path(
        args.output_path, args.answer_generation_model_name_or_path
    )

    if os.path.exists(args.output_path):
        print(
            f"Output file {args.output_path} already exists. Exiting to avoid overwriting."
        )
        exit(1)

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        
    # batch generate answers for all prompts
    prompts = [entry["prompt"] for entry in data]
    num_generations = 2
    do_sample = False
    temperatures = [
        entry.get("temperature", None) if do_sample else None for entry in data
    ]

    answer_generation_model = get_model(
        model_name_or_path=args.answer_generation_model_name_or_path,
        config=config,
    )

    responses_batch = answer_generation_model.generate(
        system_prompts=[""] * len(prompts),
        input_texts=prompts,
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


if __name__ == "__main__":
    main()
