import os
from typing import List, Optional
import json

from bertopic import BERTopic
import nltk
import re

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

import spacy
from utils import load_config
from model import get_model, Model

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")


REASON_EXTRACTION_PROMPT = """
You analyze a judge’s justification comparing Answer1 and Answer2.

Your task:
Extract the *minimal specific reasons* that explain **why one answer is better or worse** than the other.
Reasons can refer to qualities such as:
- correctness or factual accuracy
- clarity or coherence
- completeness or depth
- relevance
- logical soundness
- usefulness/helpfulness

Do NOT include:
- final verdicts without reasoning (e.g., "Answer1 is better")
- summaries of answer content
- restatements or paraphrases of the original answers
- statements that merely repeat the judge task

If there are no specific reasons given, return an empty list. The reason should be as minimal as possible.

Example:
This factual accuracy is crucial in historical analysis, making Answer2 the superior response.

Extracted reasons:
 - Factual accuracy

Return:
A JSON object:

{{
  "reasons": [
      "reason 1",
      "reason 2",
      ...
  ]
}}

Input text:
{reasoning_text}

Respond ONLY with valid JSON.
"""


def extract_reasons_with_llm(reasoning_text: str, model: Model) -> list[str]:
    prompt = REASON_EXTRACTION_PROMPT.format(reasoning_text=reasoning_text)


    responses = model.generate(
    system_prompts=["You extract reasoning in strict JSON format."],
    input_texts=[prompt],
    num_generations=1,
    max_output_tokens=500,
    temperature=0.1,
    )


    content = responses[0][0].strip()


    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(content)
        return data["reasons"]
    except json.JSONDecodeError:
        return []

def only_keep_sentences_with_answer(
    text: str, 
    answer: str, 
    use_llm_filter: bool = False,
    llm_model: Optional[Model] = None
) -> List[str]:
    """Remove sentences containing specific words from the text.

    Args:
        text: The input text to process.
        answer: Which answer to focus on ("answer1" or "answer2")
        use_llm_filter: Whether to use LLM filtering for reasoning
        llm_model: Model instance for filtering (required if use_llm_filter=True)

    Returns:
        List of filtered sentences about the specified answer.
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
    if use_llm_filter and llm_model is not None:
        llm_filtered = []
        for sentence in filtered_sentences:
            llm_filtered.extend(extract_reasons_with_llm(sentence, llm_model))
        filtered_sentences = llm_filtered

    
    return filtered_sentences


def split_sentences_more(sentences: List[str]) -> List[str]:
    """Split sentences into smaller sentences if they are too long."""
    new_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        conjunctions = [
            token.text for token in doc if token.pos_ in ["CCONJ", "SCONJ", "PUNCT"]
        ]

        if conjunctions:
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


def filter_reasonings(
    reasonings: list[str], 
    answer: str, 
    use_llm_filter: bool = False,
    llm_model: Optional[Model] = None
):
    """
    Filter reasonings to only keep the sentences that are about the answer and that have a minimal length.
    """
    filtered_reasonings = []
    filtered_reasoning = only_keep_sentences_with_answer(
        reasonings[1], answer, use_llm_filter, llm_model
    )
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
    file_data: dict, 
    better_model_name: str, 
    worse_model_name: str,
    use_llm_filter: bool = False, 
    llm_model: Optional[Model] = None
) -> dict[str, list[str]]:
    """
    Extract reasoning from a dictionary containing file data
    
    Args:
        file_data: Dictionary with evaluation data
        better_model_name: Name of the better model
        worse_model_name: Name of the worse model
        use_llm_filter: Whether to use LLM filtering
        llm_model: Model instance for filtering
    """
    counter = 0
    limit_data = 25
    reasonings = {model: [] for model in [better_model_name, worse_model_name]}
    for question_group_key in file_data.keys():
        counter = counter + 1
        if counter == 25:
            break
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
                sample_reasonings, better_answer, use_llm_filter, llm_model
            )
            sample_reasonings_worse = filter_reasonings(
                sample_reasonings, worse_answer, use_llm_filter, llm_model
            )
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
    use_llm_filter: bool = False,
    llm_filter_model_name: Optional[str] = None,
    config: Optional[dict] = None
):
    """
    Analyze results using topic modeling
    
    Args:
        file_data: Dictionary with evaluation data
        output_directory: Where to save results
        better_model_name: Name of the better model
        worse_model_name: Name of the worse model
        use_llm_filter: Whether to use LLM filtering for reasoning detection
        llm_filter_model_name: Model name/path for filtering (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        config: Configuration dictionary with API keys and settings
    """
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words.extend(["answer1", "answer2"])

    config = load_config("config.yml")
    api_key = config["openai_key"]

    cluster_model_worse = HDBSCAN(
    )

    cluster_model_better = HDBSCAN(
    )
    vectorizer_model = CountVectorizer(
        analyzer="word",
        stop_words=stop_words,
    )

    embedding_model = load_bert_model()

    better_topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model_better,
        vectorizer_model=vectorizer_model,
        language="english",
    )
    worse_topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model_worse,
        vectorizer_model=vectorizer_model,
        language="english",
    )

    llm_model = None
    if use_llm_filter:
        if not llm_filter_model_name:
            raise ValueError("llm_filter_model_name is required when use_llm_filter=True")
        
        print(f"Loading LLM filter model: {llm_filter_model_name}")
        llm_model = get_model(llm_filter_model_name, config)
    
    reasonings = extract_reasoning_from_files(
        file_data, better_model_name, worse_model_name, use_llm_filter, llm_model
    )

    better_topic_model.fit_transform(reasonings[better_model_name])
    print(f"Better model topics: {better_topic_model.get_topic_info()}")
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
    print(f"Worse model topics: {worse_topic_model.get_topic_info()}")
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