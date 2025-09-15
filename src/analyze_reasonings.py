import os
from typing import List, Optional

from bertopic import BERTopic
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
        preprocess_sentence(sentence)
        for sentence in sentences
        if answer.lower() in sentence.lower()
        and opposite_answer.lower() not in sentence.lower()
        and len(sentence) > 15
    ]
    filtered_sentences = split_sentences_more(filtered_sentences)
    return list(set(filtered_sentences))


def split_sentences_more(sentences: List[str]) -> List[str]:
    """Split sentences into smaller sentences if they are too long."""
    new_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        conjunctions = [
            token.text for token in doc if token.pos_ in ["CCONJ", "SCONJ", "PUNCT"]
        ]

        if conjunctions:
            # Escape special regex characters in conjunctions
            escaped_conjunctions = [re.escape(conj) for conj in conjunctions]
            parts = re.split(
                r"\b(?:{})\b".format("|".join(escaped_conjunctions)), sentence
            )
            for part in parts:
                part = part.strip() if part else ""
                if len(part) > 15:
                    part = part.lower()
                    part = re.sub(r"[^a-zA-Z0-9\s]", "", part)
                    if part is not None:
                        part = part.strip()
                        new_sentences.append(part)
        else:
            if len(sentence) > 15:
                new_sentences.append(sentence)
    return new_sentences


def preprocess_sentence(sentence: str) -> str:
    """Remove every noise word such as "Answer2EMPLARYassistant" that contains Answer1 or Answer2,
    but don't remove Answer1 or Answer2 if they stand alone."""

    sentence = re.sub(
        r"\b\w*answer1\w+\b|\b\w+answer1\w*\b", "", sentence, flags=re.IGNORECASE
    )
    sentence = re.sub(
        r"\b\w*answer2\w+\b|\b\w+answer2\w*\b", "", sentence, flags=re.IGNORECASE
    )

    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()


def filter_reasonings(reasonings: list[str], answer: str):
    """
    Filter reasonings to only keep the sentences that are about the answer and that have a minimal length.
    """
    filtered_reasonings = []
    filtered_reasoning = only_keep_sentences_with_answer(reasonings[1], answer)
    if len(filtered_reasoning) >= 1:
        filtered_reasonings.extend(filtered_reasoning)
    print(filtered_reasonings)
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
    file_data: dict, better_model_name: str, worse_model_name: str
) -> dict[str, list[str]]:
    """
    Extract reasoning from a dictionary containing file data
    """
    reasonings = {model: [] for model in [better_model_name, worse_model_name]}
    for question_group_key in file_data.keys():
        if question_group_key == "metadata":
            continue

        question_group = file_data[question_group_key]

        for question_data in question_group:
            sample_reasonings = question_data["result"]
            better_answer = (
                "answer1"
                if question_data["answer1"]["label"] == better_model_name
                else "answer2"
            )
            worse_answer = (
                "answer1"
                if question_data["answer1"]["label"] == worse_model_name
                else "answer2"
            )
            sample_reasonings_better = filter_reasonings(
                sample_reasonings, better_answer
            )
            sample_reasonings_worse = filter_reasonings(sample_reasonings, worse_answer)
            reasonings[better_model_name].extend(sample_reasonings_better)
            reasonings[worse_model_name].extend(sample_reasonings_worse)
    print(
        f"Extracted {len(reasonings[better_model_name])} reasonings for {better_model_name}"
        f" and {len(reasonings[worse_model_name])} reasonings for {worse_model_name}"
    )
    reasonings[better_model_name] = list(set(reasonings[better_model_name]))
    reasonings[worse_model_name] = list(set(reasonings[worse_model_name]))
    print(
        f"Unique reasonings for {better_model_name}: {len(reasonings[better_model_name])}"
    )
    print(
        f"Unique reasonings for {worse_model_name}: {len(reasonings[worse_model_name])}"
    )
    return reasonings


def analyze_reasonings_topic_model(
    file_data: dict,
    output_directory: str,
    better_model_name: str,
    worse_model_name: str,
):
    """
    Analyze results using topic modeling
    """
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words.extend(["answer1", "answer2"])

    config = load_config("config.yml")
    api_key = config["openai_key"]
    client = openai.OpenAI(api_key=api_key)
    representation_model = OpenAI(client, model="gpt-4.1", chat=True)
    cluster_model_worse = HDBSCAN(
        min_cluster_size=45,
    )

    cluster_model_better = HDBSCAN(
        min_cluster_size=12,
    )
    vectorizer_model = CountVectorizer(
        analyzer="word",
        stop_words=stop_words,
    )

    embedding_model = load_bert_model()

    better_topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model_better,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        language="english",
    )
    worse_topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model_worse,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        language="english",
    )

    reasonings = extract_reasoning_from_files(
        file_data, better_model_name, worse_model_name
    )

    better_topic_model.fit_transform(reasonings[better_model_name])
    print(f"Better model topics: {better_topic_model.get_topic_info() }")
    better_topic_model.get_topic_info().to_excel(
        os.path.join(
            output_directory,
            f"better_model_name_topics_{better_model_name}_vs_{worse_model_name}.xlsx",
        ),
        index=False,
    )
    better_hierarchical_topics = better_topic_model.hierarchical_topics(
        reasonings[better_model_name]
    )
    print(better_topic_model.get_topic_tree(better_hierarchical_topics))

    worse_topic_model.fit_transform(reasonings[worse_model_name])

    print(f"Worse model topics: {worse_topic_model.get_topic_info() }")

    worse_topic_model.get_topic_info().to_excel(
        os.path.join(
            output_directory,
            f"worse_model_name_topics_{better_model_name}_vs_{worse_model_name}.xlsx",
        ),
        index=False,
    )

    worse_hierarchical_topics = worse_topic_model.hierarchical_topics(
        reasonings[worse_model_name]
    )
    print(worse_topic_model.get_topic_tree(worse_hierarchical_topics))
