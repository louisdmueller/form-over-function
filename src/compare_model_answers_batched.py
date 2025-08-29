import json
import os
from tqdm import tqdm
from datetime import datetime

from model import get_model

from utils import (
    get_start_end_by_newest_file,
    read_file,
    load_config,
    parse_args,
    get_start_end_indices
)

def load_data_lists(data_1_path: str, data_2_path: str) -> tuple:
    if data_1_path == data_2_path:
        raise ValueError(
            "The paths for data_1 and data_2 are the same. "
            "Please provide different paths."
        )
    data_1 = read_file(data_1_path)
    data_2 = read_file(data_2_path)

    if len(data_1) != len(data_2):
        raise ValueError(
            "The data lists have different lengths: "
            f"{len(data_1)} and {len(data_2)}. "
            "Please provide lists with the same length."
        )
    return data_1, data_2


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)

    judge_model = get_model(
        model_name_or_path=args.judge_model_name_or_path,
        config=config,
    )

    data_1, data_2 = load_data_lists(args.data_1_path, args.data_2_path)

    data_directory = os.path.dirname(args.data_path)
    with open(os.path.join(data_directory, "prompts.json"), "r") as f:
        prompts = json.load(f)
    prompt = prompts[args.prompt_name]

    system_prompt = prompt["system"]

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_content = {}
    file_content["metadata"] = {
        "judge_model": args.judge_model_name_or_path,
        "data_1_path": args.data_1_path,
        "data_2_path": args.data_2_path,
        "prompt_name": args.prompt_name,
        "comment": args.comment,
    }

    name_model_1 = data_1[0]["model_name"]
    name_model_2 = data_2[0]["model_name"]
    if "/" in name_model_1:
        name_model_1 = name_model_1.replace("/", "_")
    if "/" in name_model_2:
        name_model_2 = name_model_2.replace("/", "_")

    if args.output_path is not None:
        if os.path.isdir(args.output_path):
            args.output_path = os.path.join(
                args.output_path, f"results-{current_time}-{name_model_1}-{name_model_2}.json"
            )
    # if args.output_path is not set, we save it in the same directory as the data file
    else:
        args.output_path = os.path.join(
            data_directory, f"results-{current_time}-{name_model_1}-{name_model_2}.json"
        )

    if args.start_index == "auto":
        start_idx, end_idx = get_start_end_by_newest_file(args.output_path, args.step_size, len(data_1))
        print(f"Auto-detected start index: {start_idx}, end index: {end_idx}")
    else:
        start_idx, end_idx = get_start_end_indices(args.start_index, args.step_size, len(data_1))
        print(f"Using start index: {start_idx}, end index: {end_idx}")

    input_texts = []
    answer_dicts = []
    questions = []
    question_styles = []
    idx_list = []
    answer_positions_list = []

    for idx in tqdm(
        range(start_idx, end_idx),
        total=end_idx - start_idx,
        desc="Creating batched inputs",
        unit="batch",
    ):
        question_1 = data_1[idx]["question"]
        question_2 = data_2[idx]["question"]
        answer_model_1 = data_1[idx]["answers"]["answer1"]["answer"]
        answer_model_2 = data_2[idx]["answers"]["answer1"]["answer"]
        name_model_1 = data_1[idx]["model_name"]
        name_model_2 = data_2[idx]["model_name"]

        file_content[idx] = []

        if question_1 != question_2 and args.question_style_switching:
            questions_to_use = [(question_1, name_model_1), (question_2, name_model_2)]
        else:
            questions_to_use = [(question_1, name_model_1)] 

        if args.introductionary_beginning:
            for question, model_name in questions_to_use:
                if data_1[idx]["answers"]["answer1"]["style"] == "aave":
                    question_1 = "Hey, I'm stuck on this question and was wonderin' if you could help me out. So, the question go: " + str(question_1)
                    question_2 = "Hi there, I'm a bit stuck on a question and was wondering if you could help me out. Here's the question: " + str(question_2)
                elif data_2[idx]["answers"]["answer1"]["style"] == "aave":
                    question_1 = "Hi there, I'm a bit stuck on a question and was wondering if you could help me out. Here's the question: " + str(question_1)
                    question_2 = "Hey, I'm stuck on this question and was wonderin' if you could help me out. So, the question go: " + str(question_2)

        for question, model_name in questions_to_use:
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
                    answer1=answer_dict[answer_position]["answer1"]["text"],
                    answer2=answer_dict[answer_position]["answer2"]["text"],
                )
                input_texts.append(input_text)
                answer_dicts.append(answer_dict)
                questions.append(question)
                question_styles.append(model_name)
                idx_list.append(idx)
                answer_positions_list.append(answer_position)

    system_prompts = [system_prompt] * len(input_texts)

    results = judge_model.generate(
        system_prompts,
        input_texts,
        num_generations=6,
        # num_beams=6,
        max_new_tokens=1024,
        # do_sample=True,
        # temperature=0.6,
        # top_p=0.95,
        # top_k=20,
        # min_p=0,
        **config
    )
    extracted_answers = judge_model.get_response_data(results)

    for i in range(len(input_texts)):
        idx = idx_list[i]
        answer_position = answer_positions_list[i]
        answer_dict = answer_dicts[i]
        question = questions[i]
        question_style = question_styles[i]

        answer_preferences = []
        for answer in extracted_answers["extracted_answers"][i]:
            if answer in answer_dict[answer_position]:
                answer_preferences.append(
                    answer_dict[answer_position][answer]["label"]
                )
            else:
                answer_preferences.append("Unknown")

        file_content[idx].append(
            {
                "prompt_style": question_style,
                "answer_order": answer_position,
                "question": question,
                "answer1": answer_dict[answer_position]["answer1"],
                "answer2": answer_dict[answer_position]["answer2"],
                "result": extracted_answers["output"][i],
                "extracted_answers": answer_preferences,
            }
        )

    with open(args.output_path, "w") as f:
        f.write(json.dumps(file_content, indent=4))

if __name__ == "__main__":
    main()
