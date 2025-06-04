import json
import os
from tqdm import tqdm
from datetime import datetime

from model import get_model

from utils import (
    get_df_from_file,
    load_config,
    parse_args,
)


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)

    # args.judge_model_name_or_path = "RandomAnswer"

    judge_model = get_model(
        model_name_or_path=args.judge_model_name_or_path,
        config=config,
    )

    data_df = get_df_from_file(args.data_path)

    data_directory = os.path.dirname(args.data_path)
    with open(os.path.join(data_directory, "prompts.json"), "r") as f:
        prompts = json.load(f)
    prompt = prompts[args.prompt_name]

    system_prompt = prompt["system"]

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_content = {}
    file_content["metadata"] = {
        "judge_model": args.judge_model_name_or_path,
        "data_source": args.data_path,
        "prompt_name": args.prompt_name,
        "comment": args.comment,
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
        question = data["question"]
        answer_model_1 = data["model_1"]["answer1"]["answer"]
        name_model_1 = data["model_1"]["model_name"]
        answer_model_2 = data["model_2"]["answer1"]["answer"]
        name_model_2 = data["model_2"]["model_name"]

        # key of each entry is the index of the question in the dataframe
        # and the value is a list of dictionaries with the results
        # for each permutation of the answers and prompt style
        file_content[idx] = []

        ## currently not doing question switching
        # question_weak_model = data["question_aae"] # currently not doing question switching
        # for prompt_style in ["sae", "aae"]:
            # question = question_sae if prompt_style == "sae" else question_aae

        for answer_position in ["model1-first", "model2-first"]:
            # make judge model generate its answer for both permutations
            answer_dict = {
                "model1-first": {
                    "answer1": {"text": answer_model_1, "label": name_model_1},
                    "answer2": {"text": answer_model_2, "label": name_model_2},
                    "tie": {"text": None, "label": "TIE"},
                },
                "model2-first": {
                    "answer1": {"text": answer_model_2, "label": name_model_2},
                    "answer2": {"text": answer_model_1, "label": name_model_1},
                    "tie": {"text": None, "label": "TIE"},
                },
            }
            input_text = prompt["template"].format(
                question=question,
                answer1=answer_dict[answer_position]["answer1"],
                answer2=answer_dict[answer_position]["answer2"],
            )

            results = judge_model.prompt(
                system_prompt, input_text, num_generations=3, max_output_tokens=512
            )

            # # only included for debugging purposes
            # try:
            answer_preferences = []
            for answer in results["extracted_answers"]:
                if answer in answer_dict[answer_position]:
                    answer_preferences.append(
                        answer_dict[answer_position][answer]["label"]
                    )
                else:
                    answer_preferences.append("Unknown")
            # except KeyError:
            #     print(
            #         f"KeyError: {answer_position} or extracted_answers not found in results for idx {idx}"
            #     )
            #     print(f"Results: {json.dumps(results, indent=4)}")
            #     print(f"Answer Dict: {json.dumps(answer_dict, indent=4)}")

            file_content[idx].append(
                {
                    # "prompt_style": prompt_style,
                    "answer_order": answer_position,
                    "question": question,
                    "answer1": answer_dict[answer_position]["answer1"],
                    "answer2": answer_dict[answer_position]["answer2"],
                    "result": results["output"],
                    "extracted_answers": answer_preferences,
                }
            )

            # for i, text in enumerate(results):
            #     print(f"Generated Text {idx} {i+1}/{len(results)}: {text}")

        # Writing after every question to avoid losing results in case of an error or timeout
        with open(f"{data_directory}/results-{current_time}-{name_model_1}-{name_model_2}.json", "w") as f:
            f.write(json.dumps(file_content, indent=4))


if __name__ == "__main__":
    main()
