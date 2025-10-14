"""
This script lets a judge model evaluate answers from two different models
in batches.
"""

from dataclasses import dataclass
from typing import List, Tuple
import json
import os
from tqdm import tqdm
from datetime import datetime

from utils import (
    SlurmTimeoutHandler,
    TimeBasedTimeoutHandler,
    get_start_index_by_newest_file,
    prepare_question_with_intro,
    sanitize_model_name,
)

from model import get_model

from utils import (
    read_file,
    load_config,
    parse_args,
)

print(f"Main PID: {os.getpid()}")


def load_data_lists(data_1_path: str, data_2_path: str) -> Tuple[list, list]:
    """
    Load the data lists from the given file paths and ensure they are valid.

    Args:
        data_1_path (str): Path to the first data file.
        data_2_path (str): Path to the second data file.

    Returns:
        tuple: A tuple containing two lists of data.
    """
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


@dataclass
class JudgementInput:
    input_text: str
    answer_dict: dict
    question: str
    question_style: str
    idx: int
    answer_position: str


def create_position_bias_mitigation_dict(
    answer_model_1: str, answer_model_2: str, name_model_1: str, name_model_2: str
) -> dict:
    """Create the answer dictionary with both orderings to mitigate position bias.

    Args:
        answer_model_1 (str): Answer from model 1.
        answer_model_2 (str): Answer from model 2.
        name_model_1 (str): Name of model 1.
        name_model_2 (str): Name of model 2.

    Returns:
        dict: A dictionary containing both orderings of answers.
    """
    return {
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


def prepare_judgement_inputs(
    data_1: list,
    data_2: list,
    prompt_template: str,
    start_idx: int,
    end_idx: int,
    question_style_switching: bool,
    introductory_beginning: bool,
) -> List[JudgementInput]:
    """Prepare all inputs for model evaluation."""
    judgement_inputs = []

    for idx in tqdm(
        range(start_idx, end_idx),
        total=end_idx - start_idx,
        desc="Creating batched inputs",
        unit="batch",
    ):
        item_1, item_2 = data_1[idx], data_2[idx]
        question_1, question_2 = item_1["question"], item_2["question"]
        answer_1 = item_1["answers"]["answer1"]["answer"]
        answer_2 = item_2["answers"]["answer1"]["answer"]
        name_1, name_2 = item_1["model_name"], item_2["model_name"]

        questions_to_use = (
            [(question_1, name_1), (question_2, name_2)]
            if question_1 != question_2 and question_style_switching
            else [(question_1, name_1)]
        )

        if introductory_beginning:
            question_1 = prepare_question_with_intro(
                question_1, item_1["answers"]["answer1"]["style"]
            )
            question_2 = prepare_question_with_intro(
                question_2, item_2["answers"]["answer1"]["style"]
            )

        answer_dict = create_position_bias_mitigation_dict(
            answer_1, answer_2, name_1, name_2
        )

        for question, model_name in questions_to_use:
            for answer_position in ["model1-first", "model2-first"]:
                input_text = prompt_template.format(
                    question=question,
                    answer1=answer_dict[answer_position]["answer1"]["text"],
                    answer2=answer_dict[answer_position]["answer2"]["text"],
                )

                judgement_inputs.append(
                    JudgementInput(
                        input_text=input_text,
                        answer_dict=answer_dict,
                        question=question,
                        question_style=model_name,
                        idx=idx,
                        answer_position=answer_position,
                    )
                )

    return judgement_inputs


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)

    judge_model = get_model(
        model_name_or_path=args.judge_model_name_or_path,
        config=config,
    )

    timeout_handler = TimeBasedTimeoutHandler(args.end_time, threshold=180)

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

    name_model_1 = sanitize_model_name(data_1[0]["model_name"])
    name_model_2 = sanitize_model_name(data_2[0]["model_name"])

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)
        output_filepath = os.path.join(
            args.output_path,
            f"results-{current_time}-{name_model_1}-{name_model_2}.json",
        )
    # if args.output_path is not set, we save it in the judgements directory
    else:
        output_filepath = os.path.join(
            data_directory,
            f"../judgements/results-{current_time}-{name_model_1}-{name_model_2}.json",
        )

    end_idx = int(len(data_1) * args.data_fraction)

    if args.start_index == "auto":
        start_idx = get_start_index_by_newest_file(os.path.dirname(output_filepath))
        print(f"Auto-detected start index: {start_idx}, end index: {end_idx}")
    else:
        start_idx = int(args.start_index)
        print(f"Using start index: {start_idx}, end index: {end_idx}")

    judgement_inputs = prepare_judgement_inputs(
        data_1,
        data_2,
        prompt["template"],
        start_idx,
        end_idx,
        args.question_style_switching,
        args.introductionary_beginning,
    )

    input_texts = [item.input_text for item in judgement_inputs]
    system_prompts = [system_prompt] * len(input_texts)

    results = judge_model.generate(
        system_prompts,
        input_texts,
        num_generations=3,
        # num_beams=3,
        max_new_tokens=512,
        # do_sample=True,
        # temperature=0.6,
        # top_p=0.95,
        # top_k=20,
        # min_p=0,
        timeout_handler=timeout_handler,
        **config,
    )
    extracted_answers = judge_model.get_response_data(results)
    n_successfully_generated = len(extracted_answers["output"])

    print(
        f"Successfully generated {n_successfully_generated} out of {len(input_texts)} results"
    )

    # Process results
    for i, judgement_input in enumerate(judgement_inputs[:n_successfully_generated]):
        if judgement_input.idx not in file_content:
            file_content[judgement_input.idx] = []

        answer_preferences = []
        for answer in extracted_answers["extracted_answers"][i]:
            if answer in judgement_input.answer_dict[judgement_input.answer_position]:
                answer_preferences.append(
                    judgement_input.answer_dict[judgement_input.answer_position][
                        answer
                    ]["label"]
                )
            else:
                answer_preferences.append("Unknown")

        file_content[judgement_input.idx].append(
            {
                "prompt_style": judgement_input.question_style,
                "answer_order": judgement_input.answer_position,
                "question": judgement_input.question,
                "answer1": judgement_input.answer_dict[judgement_input.answer_position][
                    "answer1"
                ],
                "answer2": judgement_input.answer_dict[judgement_input.answer_position][
                    "answer2"
                ],
                "result": extracted_answers["output"][i],
                "extracted_answers": answer_preferences,
            }
        )

    with open(output_filepath, "w") as f:
        print(f"Saving results to {output_filepath}")
        f.write(json.dumps(file_content, indent=4))

    if n_successfully_generated < len(system_prompts):
        print(
            f"Job was interrupted. Saved partial results: {n_successfully_generated}/{len(system_prompts)} items processed."
        )
    else:
        print(
            f"Job completed successfully. All {n_successfully_generated} items processed."
        )


if __name__ == "__main__":
    main()
