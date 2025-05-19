import json
import os
from tqdm import tqdm

from model import get_model

from huggingface_hub import login

from aae_translation import add_aae_to_df
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
    
    judge_model = get_model(
        model_name_or_path=args.judge_model_name_or_path,
        openai_key=config["openai_key"],
    )
    
    prompt_gen_model = get_model(
        model_name_or_path=args.prompt_model_name_or_path,
        openai_key=config["openai_key"],
    )

    data_df = get_df_from_file(args.data_path)
    data_directory = os.path.dirname(args.data_path)
    if not os.path.exists(f"{data_directory}/data_with_aae_gpt4-1.json"):
        data_df = add_aae_to_df(data_df, prompt_gen_model)
        data_df.to_json(
            f"{data_directory}/data_with_aae_gpt4-1.json",
            lines=True,
            orient="records",
        )
    else:
        data_df = get_df_from_file(os.path.join(data_directory, "data_with_aae_gpt4-1.json"))

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

    for idx, data in tqdm(data_df.iterrows(), total=len(data_df), desc="Generating results"):
        question = data["question"]
        answer1 = data["answers"]["answer1"]["answer"]
        answer2 = data["answers"]["answer1_aae"]

        for i in range(2):
            # make judge model generate its answer for both permutations
            if i == 0:
                print("Answers in original order")
                input_text = prompt["template"].format(
                    question=question, answer1=answer1, answer2=answer2
                )
            else:
                print("Answers in switched order")
                input_text = prompt["template"].format(
                    question=question, answer1=answer2, answer2=answer1
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
