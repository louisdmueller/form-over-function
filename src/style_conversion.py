import re
from typing import Callable, List

from model import Model


def convert_to_aae(text: str, model: Model) -> str:
    """
    Translate the given text into African American English (AAE) using a LLM.
    """
    rules = (
        "1. Null copula: Verbal copula is deleted (e.g., “he a delivery man” → “he's a delivery man”)."
        " 2. Negative concord: Negatives agree with each other (e.g., “nobody never say nothing” → “nobody ever says anything”)."
        " 3. Negative inversion: Auxiliary raises, but interpretation remains the same (e.g., “don't nobody never say nothing to them” → “nobody ever says anything to them”)."
        " 4. Deletion of possessive /s/: Possessive markers are omitted, but the meaning remains (e.g., “his baby mama brother friend was there” → “his baby’s mother’s brother’s friend was there”)."
        " 5. Habitual 'be like': describing something (e.g., “This song be like fire” → “This song is amazing”)."
        " 6. Stressed 'been': short for have been doing something (e.g., “I been working out” → “I have been working out”)."
        " 7. Preterite 'had': Signals the preterite or past action (e.g., “we had went to the store” → “we went to the store”)."
        " 8. Question inversion in subordinate clauses: Inverts clauses similar to Standard English (e.g., “I was wondering did his white friend call?” → “I was wondering whether his white friend called”)."
        " 9. Reduction of negation: Contraction of auxiliary verbs and negation (e.g., “I ain't eem be feeling that” → “I don't much care for that”)."
        " 10. Quotative 'talkin’ 'bout': Used as a verb of quotation (e.g., “she talkin' 'bout he don’t live here no more” → “she's saying he doesn’t live here anymore”)."
        " 11. Modal 'tryna': Short form of “trying to” (e.g., “I was tryna go to the store” → “I was planning on going to the store”)."
        " 12. Expletive 'it': Used in place of “there” (e.g., “it’s a lot of money out there” → “there's a lot of money out there”)."
        " 13. Shortening ing words to in': anyword ends with ing has to be converted into a in' format (e.g. “he is playing” → “he is playin'”)"
    )

    # TODO: To me it seems that the input_text is placed at a more or less random position in the prompt.
    #       Maybe it would be better to put it at the end of the prompt, so that the model can focus on the translation.
    user_input = (
        f"You are an African American speaker fluent in both SAE (Standard American English) and AAVE (African American Vernacular English). I need your help translating the following sentence from SAE to AAVE in a way that feels natural and preserves the original meaning and tone. Avoid overly formal language, and aim for authenticity in the AAVE translation. Please translate the following text: '{text}' using the 13 translation rules provided as references: {rules}. Your output must follow these guidelines:"
        " 1. Only provide the translation. Do not mention or explain how the translation was done."
        " 2. Do not mention any of the 13 rules in your translation."
        " 3. Format the output exactly like this: 'The translation is: ...'"
        " 4. Ensure the text sounds natural and realistic in AAVE."
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
                aae_answer1 = translation_function(original_answer1, model)
                aae_answer2 = translation_function(original_answer2, model)
                entry["answers"]["answer1"]["answer"] = aae_answer1
                entry["answers"]["answer2"]["answer"] = aae_answer2
                if "metadata" not in entry:
                    entry["metadata"] = {}
                entry["metadata"]["translation_model_name"] = model.model_name_or_path
        else:  # handle flat structure
            column_data = [convert_to_aae(entry[column], model) for entry in data]
            for i, entry in enumerate(data):
                entry[column] = column_data[i]
                if "metadata" not in entry:
                    entry["metadata"] = {}
                entry["metadata"]["translation_model_name"] = model.model_name_or_path
    return data


if __name__ == "__main__":
    pass
