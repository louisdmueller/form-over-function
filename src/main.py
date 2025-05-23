import json
import os
from tqdm import tqdm
from datetime import datetime

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
        data_df = get_df_from_file(
            os.path.join(data_directory, "data_with_aae_gpt4-1.json")
        )

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

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for idx, data in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Generating results"
    ):
        question_sae = data["question"]
        question_aae = data["question_aae"]
        answer_sae = data["answers"]["answer1"]["answer"]
        answer_aae = data["answers"]["answer1_aae"]

        for prompt_style in ["sae", "aae"]:
            question = question_sae if prompt_style == "sae" else question_aae

            for answer_position in ["sae-first", "aae-first"]:
                # make judge model generate its answer for both permutations
                answer_dict = {
                    "sae-first": {
                        "answer1": {"text": answer_sae, "label": "SAE Answer"},
                        "answer2": {"text": answer_aae, "label": "AAE Answer"},
                        "tie": {"text": None, "label": "TIE"},
                    },
                    "aae-first": {
                        "answer1": {"text": answer_aae, "label": "AAE Answer"},
                        "answer2": {"text": answer_sae, "label": "SAE Answer"},
                        "tie": {"text": None, "label": "TIE"},
                    },
                }
                input_text = prompt["template"].format(
                    question=question,
                    answer1=answer_dict[answer_position]["answer1"],
                    answer2=answer_dict[answer_position]["answer2"],
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": input_text,
                    },
                ]

                results = judge_model.prompt(
                    messages, num_generations=3, max_output_tokens=512
                )

                answer_preferences = [
                    answer_dict[answer_position][answer]["label"]
                    for answer in results["extracted_answers"]
                ]

                # for i, (text, score) in enumerate(results):
                #     print(f"Generated Text {i + 1}: {text}")
                #     print(f"Sequence Score: {score:.4f}")

                with open(f"{data_directory}/results-{current_time}.json", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "question_nr": idx,
                                "prompt_style": prompt_style,
                                "answer_order": answer_position,
                                "question": question,
                                "answer1": answer_dict[answer_position]["answer1"],
                                "answer2": answer_dict[answer_position]["answer2"],
                                "result": results["output"],
                                "extracted_answers": answer_preferences,
                            },
                            indent=4,
                        )
                    )
                    f.write("\n")

                for i, text in enumerate(results):
                    print(f"Generated Text {idx} {i}/{len(results)}: {text}")


if __name__ == "__main__":
    main()
