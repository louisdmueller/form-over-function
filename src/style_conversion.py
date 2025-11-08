import re
from typing import Callable, List, Optional

from model import Model


def add_errors(text: str, model: Model, question: Optional[str]) -> str:
    """
    Add errors to an answer using a LLM.
    """

    prompt = f"""
    ### You are a fact checker.
    ### You will be given a question -answer pair.
    ### You will do your best to identify all the facts in the given answer.
    ### You will re-write the answer with 2-3 factual errors that are not easy to identify.
    ### You should list out the errors that you want to add in the answer.
    ### You should respond in the format of ### You should not modify any content apart from the factual errors.
    ```fact
    <fact list>
    ```
    ```error
    <error list>
    ```
    ```answer
    <factual error version of answer>
    ```
    ---
    Question: {question}
    
    Answer: {text}
    
    ---
    """
    response = model.generate(
        system_prompts=[""],
        input_texts=[prompt],
        max_output_tokens=len(text) + 50,
    )

    if isinstance(response, list) and len(response) == 1:
        # generate returns a list in a list
        # so we need to extract the first element
        # because we only get one single response from the model
        response = response[0][0]

        # Extract answer section
    answer_pattern = r"```answer\s*\n(.*?)```"
    answer_match = re.search(answer_pattern, response, re.DOTALL)  # type: ignore
    answer = answer_match.group(1).strip() if answer_match else None
    if not answer:
        raise ValueError("The response from the model does not contain an answer.")
    error_text = answer.strip().strip("'")
    return error_text


def convert_data(
    data: list[dict],
    model: Model,
    columns: List[str],
    translation_function: Callable = add_errors,
) -> List[dict]:
    """
    Convert the answers in the data to AAE using a LLM.
    # TODO also handle standard english -> basic english conversion
    """
    for column in columns:
        if column == "answers":  # handle nested structure
            for entry in data:
                original_answer1 = entry["answers"]["answer1"]["answer"]
                original_answer2 = entry["answers"]["answer2"]["answer"]
                aae_answer1 = translation_function(
                    original_answer1, model, entry["question"]
                )
                aae_answer2 = translation_function(
                    original_answer2, model, entry["question"]
                )
                entry["answers"]["answer1"]["answer"] = aae_answer1
                entry["answers"]["answer2"]["answer"] = aae_answer2
                if "metadata" not in entry:
                    entry["metadata"] = {}
                entry["metadata"]["translation_model_name"] = model.model_name_or_path
        else:  # handle flat structure
            column_data = [
                translation_function(entry[column], model, entry["question"])
                for entry in data
            ]
            for i, entry in enumerate(data):
                entry[column] = column_data[i]
                if "metadata" not in entry:
                    entry["metadata"] = {}
                entry["metadata"]["translation_model_name"] = model.model_name_or_path
    return data


if __name__ == "__main__":
    pass
