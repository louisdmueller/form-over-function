import json

import yaml

from calculate_readability_metrics import load_onestopqa
from model import get_model
from prompts import (
    aave_prompt,
    basic_english_prompt,
    complex_prompt,
    error_prompt,
    simple_prompt,
)
from utils import read_file, write_file


def rewrite_answers(
    model_name_or_path: str = "gpt-4.1",
    src_file: str = "data/generated_answers/gpt-4.1-answers.json",
    rewrite_type: str = "aave",
) -> None:

    with open("config.yml", "r") as file:
        # supply openAI key via config
        config = yaml.safe_load(file)

    model = get_model(
        model_name_or_path=model_name_or_path,
        config=config,
    )

    original_dicts = read_file(src_file)
    new_dicts = [dict(dictionary) for dictionary in original_dicts]

    texts = [
        [entry["question"] for entry in original_dicts],
        [entry["answers"]["answer1"]["answer"] for entry in original_dicts],
        [entry["answers"]["answer2"]["answer"] for entry in original_dicts],
    ]

    # rewrite to AAVE, simplified English, or introduce errors
    if rewrite_type == "aave":
        prompt = aave_prompt
        file_ending = "_aave.json"

    elif rewrite_type == "simple":
        prompt = simple_prompt
        file_ending = "_simple.json"

    elif rewrite_type == "errors":
        prompt = error_prompt
        file_ending = "_errors.json"

    else:
        raise ValueError("Rewrite type must be one of [aave/simple/errors]")

    prompts = (
        [prompt(question, answer) for (question, answer) in zip(texts[0], texts[1])]
        if rewrite_type == "errors"
        else (
            [prompt(question) for question in texts[0]],
            [prompt(answer) for answer in texts[1]],
        )
    )

    system_prompts = [""] * len(prompts[0])

    # generate either only answers (errors) or questions and answers (aave/simplified)
    responses = (
        model.generate(system_prompts=[""] * len(prompts), input_texts=prompts)
        if rewrite_type == "errors"
        else [
            model.generate(system_prompts=system_prompts, input_texts=prompts[0]),
            model.generate(system_prompts=system_prompts, input_texts=prompts[1]),
        ]
    )

    for i, question in enumerate(new_dicts):
        question["model_name"] = model_name_or_path
        question["metadata"]["rewrite_model"] = model.exact_model
        question["metadata"]["temperature"] = model.temperature
        del question["answers"]["answer2"]
        del question["level"]
        del question["prompt"]
        del question["model_id"]
        del question["model_name"]
        del question["temperature"]

        if rewrite_type == "errors":
            question["answers"]["answer1"]["answer"] = responses[i][0]
        else:
            question["question"] = responses[0][i][0]
            question["answers"]["answer1"]["answer"] = responses[1][i][0]

    write_file(src_file.replace(".json", file_ending), new_dicts)


def rewrite_onestopqa(
    model_name_or_path: str = "gpt-4.1",
    prompt_type: str = "simple",
    outfile: str = "data/readability_metrics/onestopqa_test_data/simple_prompt.json",
) -> None:
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    model = get_model(
        model_name_or_path=model_name_or_path,
        config=config,
    )

    prompt_options = {
        "simple": simple_prompt,
        "complex": complex_prompt,
        "basic": basic_english_prompt,
    }
    prompt = prompt_options.get(prompt_type)
    if not prompt:
        raise ValueError("Prompt type must be one of [simple/complex/basic]")

    articles = load_onestopqa(reference=False)
    articles_truncated = [" ".join(paragraph[:1]) for paragraph in articles]
    prompts = [prompt(article) for article in articles_truncated]
    system_prompts = [""] * len(prompts)
    responses = model.generate(
        system_prompts=system_prompts,
        input_texts=prompts,
    )
    simple_language_paragraphs = [response[0] for response in responses]
    # write output list to file
    with open(outfile, "w") as file:
        json.dump(simple_language_paragraphs, file, indent=4)
    return


if __name__ == "__main__":
    rewrite_answers(
        src_file="data/generated_answers/gpt-4.1-answers.json", rewrite_type="aave"
    )
    rewrite_answers(
        src_file="data/generated_answers/gpt-4.1-answers.json", rewrite_type="simple"
    )
    rewrite_answers(
        src_file="data/generated_answers/gpt-4.1-answers.json", rewrite_type="errors"
    )
