import json
import os
from model import get_model

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
    openai_client = setup_openai_client(config["openai_key"])
    
    judge_model = get_model(
        model_name_or_path=args.judge_model_name_or_path,
        openai_key=config["openai_key"],
    )
    
    prompt_generation_model = get_model(
        model_name_or_path=args.prompt_model_name_or_path,
        openai_key=config["openai_key"],
    )

    data_df = get_df_from_file(args.data_path)
    data_directory = os.path.dirname(args.data_path)
    if not os.path.exists(f"{data_directory}/translated.json"):
        translate_df(data_df, prompt_generation_model)
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

    system_prompt = prompt["system"]

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

        results = judge_model.prompt(messages, num_generations=6, max_output_tokens=512)

        # for i, (text, score) in enumerate(results):
        #     print(f"Generated Text {i + 1}: {text}")
        #     print(f"Sequence Score: {score:.4f}")

        for i, text in enumerate(results):
            print(f"Generated Text {i + 1}: {text}")


        break


if __name__ == "__main__":
    main()
