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

    # TODO: is it necessary to load the data as dataframes? 
    #       we are later using iloc to access the data
    #       maybe we can just use the json lines file directly?
    df_1, df_2 = load_dataframes(args.data_1_path, args.data_2_path)

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
        args.end_index = len(df_1)

    input_texts = []
    answer_dicts = []
    questions = []
    idx_list = []
    answer_positions_list = []

    for idx in tqdm(
        range(args.start_index, args.end_index),
        total=args.end_index - args.start_index,
        desc="Generating results",
    ):
        question = df_1.iloc[idx]["question"]
        answer_model_1 = df_1.iloc[idx]["answers"]["answer1"]["answer"]
        answer_model_2 = df_2.iloc[idx]["answers"]["answer1"]["answer"]
        name_model_1 = df_1.iloc[idx]["model_name"]
        name_model_2 = df_2.iloc[idx]["model_name"]

        file_content[idx] = []

        for answer_position in ["model1-first", "model2-first"]:
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
            input_texts.append(input_text)
            answer_dicts.append(answer_dict)
            questions.append(question)
            idx_list.append(idx)
            answer_positions_list.append(answer_position)

    system_prompts = [system_prompt] * len(input_texts)

    results = judge_model.prompt_batched(
        system_prompts, input_texts, num_generations=3, max_output_tokens=512
    )

    for i in range(len(input_texts)):
        idx = idx_list[i]
        answer_position = answer_positions_list[i]
        answer_dict = answer_dicts[i]
        question = questions[i]

        answer_preferences = []
        for answer in results["extracted_answers"][i]:
            if answer in answer_dict[answer_position]:
                answer_preferences.append(
                    answer_dict[answer_position][answer]["label"]
                )
            else:
                answer_preferences.append("Unknown")

        file_content[idx].append(
            {
                "answer_order": answer_position,
                "question": question,
                "answer1": answer_dict[answer_position]["answer1"],
                "answer2": answer_dict[answer_position]["answer2"],
                "result": results["output"][i],
                "extracted_answers": answer_preferences,
            }
        )

    # Writing after every question to avoid losing results in case of an error or timeout
    with open(f"{data_directory}/results-{current_time}-{name_model_1}-{name_model_2}.json", "w") as f:
        f.write(json.dumps(file_content, indent=4))

def load_dataframes(data_1_path: str, data_2_path: str) -> tuple:
    print("Is it necessary to load the data as dataframes?")
    if data_1_path == data_2_path:
        raise ValueError(
            "The paths for data_1 and data_2 are the same. "
            "Please provide different paths."
        )
    data_df_1 = get_df_from_file(data_1_path)
    data_df_2 = get_df_from_file(data_2_path)

    # with open(data_1_path, "r") as f:
    #     data_df_1 = [json.loads(line) for line in f.readlines()]
    # with open(data_2_path, "r") as f:
    #     data_df_2 = [json.loads(line) for line in f.readlines()]
    
    if len(data_df_1) != len(data_df_2):
        raise ValueError(
            "The dataframes have different lengths: "
            f"{len(data_df_1)} and {len(data_df_2)}. "
            "Please provide dataframes with the same length."
        )
    # if not data_df_1.columns.equals(data_df_2.columns):
    #     raise ValueError(
    #         "The dataframes have different columns: "
    #         f"{data_df_1.columns} and {data_df_2.columns}. "
    #         "Please provide dataframes with the same columns."
    #     )
    return data_df_1, data_df_2


if __name__ == "__main__":
    main()
