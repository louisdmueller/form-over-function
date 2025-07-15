from pathlib import Path
from typing import List

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from bertopic.vectorizers import ClassTfidfTransformer
import nltk
import re
import openai
from bertopic.representation import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

import spacy

from utils import load_config

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """Basic text preprocessing"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())

    return text


def only_keep_sentences_with_answer(text: str, answer: str) -> List[str]:
    """Remove sentences containing specific words from the text.

    Args:
        text: The input text to process.
        words_to_remove: List of words that, if found in a sentence, will cause that sentence to be removed.

    Returns:
        str that does not contain any of the specified words.
    """
    sentences = nltk.sent_tokenize(text)
    opposite_answer = "answer2" if answer == "answer1" else "answer1"
    filtered_sentences = [
        sentence
        for sentence in sentences
        if answer.lower() in sentence.lower()
        and opposite_answer.lower() not in sentence.lower()
        and len(sentence) > 15
    ]
    return list(set(filtered_sentences))


def filter_reasonings(reasonings: list[str], answer: str):
    """
    Filter reasonings to only keep the sentences that are about the answer and that have a minimal length.
    """
    filtered_reasonings = []
    filtered_reasoning = only_keep_sentences_with_answer(reasonings[1], answer)
    if len(filtered_reasoning) >= 1:
        filtered_reasonings.extend(filtered_reasoning)
    return filtered_reasonings


def load_bert_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a BERT model for embeddings

    Args:
        model_name: Name of the sentence transformer model

    Returns:
        model: the loaded model
    """
    print(f"Loading BERT model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def extract_reasoning_from_files(
    file_data: dict, better_model: str, worse_model: str
) -> dict[str, list[str]]:
    """
    Extract reasoning from a dictionary containing file data
    """
    reasonings = {model: [] for model in [better_model, worse_model]}
    for question_group_key in file_data.keys():
        if question_group_key == "metadata":
            continue

        question_group = file_data[question_group_key]

        for question_data in question_group:
            sample_reasonings = question_data["result"]
            better_answer = (
                "answer1"
                if question_data["answer1"]["label"] == better_model
                else "answer2"
            )
            worse_answer = (
                "answer1"
                if question_data["answer1"]["label"] == worse_model
                else "answer2"
            )
            sample_reasonings_better = filter_reasonings(
                sample_reasonings, better_answer
            )
            sample_reasonings_worse = filter_reasonings(sample_reasonings, worse_answer)
            reasonings[better_model].extend(sample_reasonings_better)
            reasonings[worse_model].extend(sample_reasonings_worse)
    return reasonings


def analyze_reasonings_topic_model(
    file_data: dict, better_model: str, worse_model: str, output_directory: Path
):
    """
    Analyze results using topic modeling
    """
    config = load_config("config.yml")
    client = openai.OpenAI(api_key=config["openai_key"])
    llm_representation_model = OpenAI(
        client=client,
        model="gpt-4o-mini",
        chat=True,
    )

    stop_words = nltk.corpus.stopwords.words("english")
    stop_words.extend(["answer1", "answer2"])
    # Clustering model: See [2] for more details
    cluster_model = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
    vectorizer_model = CountVectorizer(
        analyzer="word",
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=0.05,
        max_df=0.85,
    )

    embedding_model = load_bert_model()

    key_bert_inspired_representation_model = KeyBERTInspired()

    representation_model = {
        "KeyBertInspired Representation": key_bert_inspired_representation_model,
        "LLM Representation": llm_representation_model,
    }

    better_topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model,
        ctfidf_model=ctfidf_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        language="english",
    )
    worse_topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model,
        ctfidf_model=ctfidf_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        language="english",
    )

    reasonings = extract_reasoning_from_files(file_data, better_model, worse_model)

    better_topic_model.fit_transform(reasonings[better_model])
    print(f"Better model topics: {better_topic_model.get_topic_info() }")
    better_topic_model.get_topic_info().to_excel(
        output_directory / f"better_model_topics_{better_model}_vs_{worse_model}.xlsx",
        index=False,
    )

    worse_topic_model.fit_transform(reasonings[worse_model])

    print(f"Worse model topics: {worse_topic_model.get_topic_info() }")

    worse_topic_model.get_topic_info().to_excel(
        output_directory / f"worse_model_topics_{better_model}_vs_{worse_model}.xlsx",
        index=False,
    )
