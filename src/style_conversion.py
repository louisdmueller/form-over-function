import re
from typing import Callable, List, Optional

from model import Model


def convert_to_aae(text: str, model: Model, question: Optional[str]) -> str:
    """
    Translate the given text into African American English (AAE) using a LLM.
    """
    rules = (
        "1. Null copula: Verbal copula is deleted (e.g., “He a delivery man” → “He's a delivery man”).\n"
        "2. Negative concord: Negatives agree with each other (e.g., “Nobody never say nothing” → “Nobody ever says anything”).\n"
        "3. Negative inversion: Auxiliary raises, but interpretation remains the same (e.g., “Don't nobody never say nothing to them” → “Nobody ever says anything to them”).\n"
        "4. Deletion of possessive /s/: Possessive markers are omitted, but the meaning remains (e.g., “His baby mama brother friend was there” → “His baby’s mother’s brother’s friend was there”).\n"
        "5. Habitual 'be like': describing something (e.g., “This song be like fire” → “This song is amazing”).\n"
        "6. Stressed 'been': short for have been doing something (e.g., “I been working out” → “I have been working out”).\n"
        "7. Preterite 'had': Signals the preterite or past action (e.g., “We had went to the store” → “We went to the store”).\n"
        "8. Question inversion in subordinate clauses: Inverts clauses similar to Standard English (e.g., “I was wondering did his white friend call?” → “I was wondering whether his white friend called”).\n"
        "9. Reduction of negation: Contraction of auxiliary verbs and negation (e.g., “I ain't even be feeling that” → “I don't much care for that”).\n"
        "10. Quotative 'talkin’ ’bout': Used as a verb of quotation (e.g., “She talkin’ ’bout he don’t live here no more” → “She's saying he doesn’t live here anymore”).\n"
        "11. Modal 'tryna': Short form of “trying to” (e.g., “I was tryna go to the store” → “I was planning on going to the store”).\n"
        "12. Expletive 'it': Used in place of “there” (e.g., “It’s a lot of money out there” → “There's a lot of money out there”).\n"
        "13. Shortening 'ing' words to 'in’': Any word that ends with 'ing' has to be converted into a in’ format (e.g. “he is playing” → “he is playin’”)."
    )

    user_input = (
        f"You are an African American speaker fluent in both SAE (Standard American English) and AAVE (African American Vernacular English). Translate the following sentence from SAE to AAVE in a way that feels natural and preserves the original meaning and tone. Avoid overly formal language, and aim for authenticity in the AAVE translation. Apply the following 13 translation rules:\n\n{rules}\n\nYour output must also follow these guidelines:\n\n"
        "1. Only provide the translation. Do not mention or explain how the translation was done.\n"
        "2. Do not mention any of the 13 rules in your translation.\n"
        "3. Format the output exactly like this: 'The translation is: ...'\n"
        "4. Ensure the text sounds natural and realistic in AAVE.\n\n"
        "Please translate the following text: '{text}'"
    )

    # TODO: generate awaits a list of input texts and system prompts.
    #       But currently we only pass a single input text and an empty system prompt.
    #       We should modify convert_to_aae to accept a list of texts and prompts.
    #       And return a list of translations.
    response = model.generate(
        system_prompts=[""],
        input_texts=[user_input],
        max_output_tokens=len(text) + 50,
    )

    if isinstance(response, list) and len(response) == 1:
        # generate returns a list in a list
        # so we need to extract the first element
        # because we only get one single response from the model
        response = response[0][0]

    pattern = r"The translation is:\s*(.*)"
    match = re.search(pattern, str(response), re.DOTALL)
    if match:
        aae_text = match.group(1)
    else:
        raise ValueError(
            "The response from the model does not match the expected format."
        )
    aae_text = aae_text.strip().strip("'")
    return aae_text


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
    translation_function: Callable = convert_to_aae,
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
