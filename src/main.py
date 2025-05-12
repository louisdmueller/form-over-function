import json
import math
import os

import torch
import transformers
from huggingface_hub import login

from aae_translation import setup_openai_client, translate_df
from utils import (
    create_comparison_csv,
    get_df_from_file,
    load_config,
    parse_args,
)


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)

    # access to llama models is restricted
    login(token=config["huggingface_hub_token"])
    setup_openai_client(config["openai_key"])

    data_df = get_df_from_file(args.data_path)
    data_directory = os.path.dirname(args.data_path)
    if not os.path.exists(f"{data_directory}/translated.json"):
        translate_df(data_df)
        data_df.to_json(
            f"{data_directory}/translated.json",
            lines=True,
            orient="records",
        )
    else:
        data_df = get_df_from_file(os.path.join(data_directory, "translated.json"))

    with open(os.path.join(data_directory, "prompts.json"), "r") as f:
        prompts = json.load(f)
    prompt = prompts["cot_and_then_answer_question"]

    # To manually verify the translations, create a CSV
    # with the originals and the permuted answers
    create_comparison_csv(
        data_df,
        f"{data_directory}/comparison.csv",
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.prompt_model_name_or_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.prompt_model_name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token  # for correct padding

    system_prompt = prompt["system"]

    with torch.no_grad():
        for idx, data in data_df.iterrows():
            question = data["question"]
            answer1 = data["answers"]["answer1"]["answer"]
            answer2 = data["answers"]["answer1_permutated"]

            input_text = prompt["template"].format(
                question=question, answer1=answer1, answer2=answer2
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": input_text,
                },
            ]

            formatted_input = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                tokenize=False,
                add_generation_prompt=True,
                max_length=tokenizer.model_max_length,
            )
            inputs = tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            inputs = inputs.to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=6,
                num_return_sequences=6,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

            sequence_scores = output.sequences_scores  # es.tolist()

            results = []
            for i, (sequence, log_score) in enumerate(
                zip(output.sequences, sequence_scores)
            ):
                decoded_text = tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                decoded_text = decoded_text.split("assistant\n")[1].strip()

                prob_score = math.exp(log_score)
                results.append((decoded_text, prob_score))

            for i, (text, score) in enumerate(results):
                print(f"Generated Text {i + 1}: {text}")
                print(f"Sequence Score: {score:.4f}")

            break


if __name__ == "__main__":
    main()
