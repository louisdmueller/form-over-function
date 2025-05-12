import re

import pandas as pd
from openai import OpenAI


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


def translate_text(text: str, openai_client) -> str:
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

    rules = "1. Null copula: Verbal copula is deleted (e.g., “he a delivery man” → “he's a delivery man”).\
        2. Negative concord: Negatives agree with each other (e.g., “nobody never say nothing” → “nobody ever says anything”).\
        3. Negative inversion: Auxiliary raises, but interpretation remains the same (e.g., “don't nobody never say nothing to them” → “nobody ever says anything to them”).\
        4. Deletion of possessive /s/: Possessive markers are omitted, but the meaning remains (e.g., “his baby mama brother friend was there” → “his baby’s mother’s brother’s friend was there”).\
        5. Habitual 'be like': descrbing something (e.g., “This song be like fire” → “This song is amazing”).\
        6. Stressed 'been': short for have been doing something (e.g., “I been working out” → “I have been working out”).\
        7. Preterite 'had': Signals the preterite or past action (e.g., “we had went to the store” → “we went to the store”).\
        8. Question inversion in subordinate clauses: Inverts clauses similar to Standard English (e.g., “I was wondering did his white friend call?” → “I was wondering whether his white friend called”).\
        9. Reduction of negation: Contraction of auxiliary verbs and negation (e.g., “I ain't eem be feeling that” → “I don't much care for that”).\
        10. Quotative 'talkin’ 'bout': Used as a verb of quotation (e.g., “she talkin' 'bout he don’t live here no more” → “she's saying he doesn’t live here anymore”).\
        11. Modal 'tryna': Short form of “trying to” (e.g., “I was tryna go to the store” → “I was planning on going to the store”).\
        12. Expletive 'it': Used in place of “there” (e.g., “it’s a lot of money out there” → “there's a lot of money out there”).\
        13. Shortening ing words to in': anyword ends with ing has to be converted into a in' format (e.g. “he is playing” → “he is playin'”)"

    user_input = f"You are an African American speaker fluent in both SAE (Standard American English) and AAVE (African American Vernacular English). I need your help translating the following sentence from SAE to AAVE in a way that feels natural and preserves the original meaning and tone. Avoid overly formal language, and aim for authenticity in the AAVE translation. Please translate the following text: '{text}' using the 13 translation rules provided as references: {rules}. Your output must follow these guidelines: \
    1. Only provide the translation. Do not mention or explain how the translation was done. \
    2. Do not mention any of the 13 rules in your translation.\
    3. Format the output exactly like this: 'The translation is: ...' \
    4. Ensure the text sounds natural and realistic in AAVE. "

    response = openai_client.responses.create(
        model="gpt-4o-mini",
        input=user_input,
        max_output_tokens=len(text) + 50,
    )

    pattern = r"The translation is:\s*(.*)"
    match = re.search(pattern, response.output_text, re.DOTALL)
    if match:
        translated_text = match.group(1)
    else:
        translated_text = text
    updated_translation = replace_words(translated_text, replacement_dict)
    return updated_translation


def add_translation(row: pd.Series, openai_client: OpenAI) -> pd.Series:
    """
    Add a translation from SAE to AAE to the row of the Dataframe.
    """
    original_answer = row["answers"]["answer1"]["answer"]
    translated_answer = translate_text(original_answer, openai_client)
    row["answers"]["answer1_permutated"] = translated_answer
    return row


def translate_df(df: pd.DataFrame, openai_client: OpenAI) -> pd.DataFrame:
    """
    Run the translation from SAE to AAE on the DataFrame using the OpenAI client.
    """
    df = df.apply(lambda row: add_translation(row, openai_client), axis=1)
    return df


if __name__ == "__main__":
    pass
