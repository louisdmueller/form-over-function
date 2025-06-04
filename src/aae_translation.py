import re

import pandas as pd
from openai import OpenAI

from model import Model


def setup_openai_client(api_key: str) -> OpenAI:
    """
    Set up the OpenAI client with the provided API key.
    """
    openai_client = OpenAI(api_key=api_key)
    return openai_client


def replace_words(text: str, replacement_dict: dict) -> str:
    """
    Replace words in the text based on the provided replacement dictionary.
    """
    sorted_dict = dict(sorted(replacement_dict.items(), key=lambda x: -len(x[0])))

    for word, replacement in sorted_dict.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", replacement, text)
    return text


def convert_to_aae(text: str, model: Model) -> str:
    """
    Translate the given text into African American English (AAE) using GPT-4o mini.
    """
    replacement_dict = {
        "isn't": "ain't",
        "going to": "gonna",
        "because": "cuz",
        "have": "got",
        "ing": "in",
        "about": "bout",
    }

    rules = ("1. Null copula: Verbal copula is deleted (e.g., “he a delivery man” → “he's a delivery man”)."
    " 2. Negative concord: Negatives agree with each other (e.g., “nobody never say nothing” → “nobody ever says anything”)."
    " 3. Negative inversion: Auxiliary raises, but interpretation remains the same (e.g., “don't nobody never say nothing to them” → “nobody ever says anything to them”)."
    " 4. Deletion of possessive /s/: Possessive markers are omitted, but the meaning remains (e.g., “his baby mama brother friend was there” → “his baby’s mother’s brother’s friend was there”)."
    " 5. Habitual 'be like': descrbing something (e.g., “This song be like fire” → “This song is amazing”)."
    " 6. Stressed 'been': short for have been doing something (e.g., “I been working out” → “I have been working out”)."
    " 7. Preterite 'had': Signals the preterite or past action (e.g., “we had went to the store” → “we went to the store”)."
    " 8. Question inversion in subordinate clauses: Inverts clauses similar to Standard English (e.g., “I was wondering did his white friend call?” → “I was wondering whether his white friend called”)."
    " 9. Reduction of negation: Contraction of auxiliary verbs and negation (e.g., “I ain't eem be feeling that” → “I don't much care for that”)."
    " 10. Quotative 'talkin’ 'bout': Used as a verb of quotation (e.g., “she talkin' 'bout he don’t live here no more” → “she's saying he doesn’t live here anymore”)."
    " 11. Modal 'tryna': Short form of “trying to” (e.g., “I was tryna go to the store” → “I was planning on going to the store”)."
    " 12. Expletive 'it': Used in place of “there” (e.g., “it’s a lot of money out there” → “there's a lot of money out there”)."
    " 13. Shortening ing words to in': anyword ends with ing has to be converted into a in' format (e.g. “he is playing” → “he is playin'”)")

    # TODO: To me it seems that the input_text is placed at a more or less random position in the prompt.
    #       Maybe it would be better to put it at the end of the prompt, so that the model can focus on the translation.
    user_input = (f"You are an African American speaker fluent in both SAE (Standard American English) and AAVE (African American Vernacular English). I need your help translating the following sentence from SAE to AAVE in a way that feels natural and preserves the original meaning and tone. Avoid overly formal language, and aim for authenticity in the AAVE translation. Please translate the following text: '{text}' using the 13 translation rules provided as references: {rules}. Your output must follow these guidelines:"
    " 1. Only provide the translation. Do not mention or explain how the translation was done."
    " 2. Do not mention any of the 13 rules in your translation."
    " 3. Format the output exactly like this: 'The translation is: ...'"
    " 4. Ensure the text sounds natural and realistic in AAVE.")
    
    response = model.query_model(
        system_prompt="",
        message = user_input,
        num_generations=1,
        max_output_tokens=len(text) + 50,
    )

    # For some models the response is a list of strings, for others it is a single string.
    if isinstance(response, list) and len(response) == 1:
        response = response[0]

    pattern = r"The translation is:\s*(.*)"
    match = re.search(pattern, str(response), re.DOTALL)
    if match:
        aae_text = match.group(1)
    else:
        aae_text = text
    aae_text = replace_words(aae_text, replacement_dict)
    aae_text = aae_text.strip().strip("'")
    return aae_text


def add_aae_to_answers(row: pd.Series, model: Model) -> pd.Series:
    """
    Add a translation from SAE to AAE to the row of the Dataframe.
    """
    original_answer1 = row["answers"]["answer1"]["answer"]
    original_answer2 = row["answers"]["answer2"]["answer"]

    aae_answer1 = convert_to_aae(original_answer1, model)
    aae_answer2 = convert_to_aae(original_answer2, model)
    row["answers"]["answer1_aae"] = aae_answer1
    row["answers"]["answer2_aae"] = aae_answer2

    # theoretically returning the row is not strictly necessary,
    # since the data is modified in place
    return row


def add_aae_to_df(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    """
    Run the translation from SAE to AAE on the DataFrame using the OpenAI client.
    """
    df = df.apply(lambda row: add_aae_to_answers(row, model), axis=1)
    df["question"] = df["question"].apply(lambda question: "Hi there, I'm a bit stuck on a question and was wondering if you could help me out. Here's the question: " + str(question))
    df["question_aae"] = df["question"].apply(lambda question: convert_to_aae(str(question), model))
    return df


if __name__ == "__main__":
    pass
