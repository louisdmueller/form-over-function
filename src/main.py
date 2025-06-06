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

    judge_model = get_model(
        model_name_or_path=args.judge_model_name_or_path,
        config=config,
    )

    prompt_gen_model = get_model(
        model_name_or_path=args.prompt_model_name_or_path,
        config=config,
    )

    data_df = get_df_from_file(args.data_path)
    data_directory = os.path.dirname(args.data_path)
    if not os.path.exists(f"{data_directory}/data_with_aae_{args.prompt_model_name_or_path}.json"):
        data_df = add_aae_to_df(data_df, prompt_gen_model)
        data_df.to_json(
            f"{data_directory}/data_with_aae_{args.prompt_model_name_or_path}.json",
            lines=True,
            orient="records",
        )
    else:
        data_df = get_df_from_file(
            os.path.join(data_directory, f"data_with_aae_{args.prompt_model_name_or_path}.json")
        )

    with open(os.path.join(data_directory, "prompts.json"), "r") as f:
        prompts = json.load(f)
    prompt = prompts[args.prompt_name]

    if not os.path.exists(f"{data_directory}/comparison.csv"):
        # To manually verify the translations, create a CSV
        # with the originals and the permuted answers
        create_comparison_csv(
            data_df,
            f"{data_directory}/comparison.csv",
        )

    system_prompt = prompt["system"]

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_content = {}
    file_content["metadata"] = {
        "judge_model": args.judge_model_name_or_path,
        "prompt_model": args.prompt_model_name_or_path,
        "data_source": args.data_path,
        "prompt_name": args.prompt_name,
    }

    # If end_index is not provided, it it set to None, since
    # the data length is not initialized yet.
    # We need to manually set it to the length of the dataframe
    if args.end_index is None:
        args.end_index = len(data_df)

    for idx, data in tqdm(
        data_df.iloc[args.start_index : args.end_index].iterrows(),
        total=args.end_index - args.start_index,
        desc="Generating results",
    ):
        question_sae = data["question"]
        question_aae = data["question_aae"]
        answer_sae = data["answers"]["answer1"]["answer"]
        answer_aae = data["answers"]["answer1_aae"]

        # key of each entry is the index of the question in the dataframe
        # and the value is a list of dictionaries with the results
        # for each permutation of the answers and prompt style
        file_content[idx] = []

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

                # only included for debugging purposes
                try:
                    answer_preferences = []
                    for answer in results["extracted_answers"]:
                        if answer in answer_dict[answer_position]:
                            answer_preferences.append(
                                answer_dict[answer_position][answer]["label"]
                            )
                        else:
                            answer_preferences.append("Unknown")
                except KeyError:
                    print(
                        f"KeyError: {answer_position} or extracted_answers not found in results for idx {idx}"
                    )
                    print(f"Results: {json.dumps(results, indent=4)}")
                    print(f"Answer Dict: {json.dumps(answer_dict, indent=4)}")

                # for i, (text, score) in enumerate(results):
                #     print(f"Generated Text {i + 1}: {text}")
                #     print(f"Sequence Score: {score:.4f}")

                file_content[idx].append(
                    {
                        "prompt_style": prompt_style,
                        "answer_order": answer_position,
                        "question": question,
                        "answer1": answer_dict[answer_position]["answer1"],
                        "answer2": answer_dict[answer_position]["answer2"],
                        "result": results["output"],
                        "extracted_answers": answer_preferences,
                    }
                )

                for i, text in enumerate(results):
                    print(f"Generated Text {idx} {i+1}/{len(results)}: {text}")

        # Writing after every question to avoid losing results in case of an error or timeout
        with open(f"{data_directory}/results-{current_time}.json", "w") as f:
            f.write(json.dumps(file_content, indent=4))


if __name__ == "__main__":
    main()
