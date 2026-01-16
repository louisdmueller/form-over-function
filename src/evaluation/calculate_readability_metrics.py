import json
from statistics import mean
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import textstat as ts
from bert_score import score
from datasets import load_dataset

# from easse.sari import corpus_sari  # type: ignore
# easse package is not available on PyPI
from spacy.matcher import Matcher

from utils.utils import read_jsonl_file


def calculate_readability_metrics(texts: List[str] | str, verbose: bool = True) -> dict:
    """
    Calculate readability metrics for a list of texts.

    Args:
      texts (List[str] | str): List of texts or path to JSON file containing
      texts to analyze.
      verbosity (int): If true, print the mean of each calculated metric.

    Returns:
      dict: A dictionary containing the readability metrics.
    """
    # if a file path is provided, read the file and extract texts
    if isinstance(texts, str):
        sae_dicts = read_jsonl_file(texts)
        texts = [entry["answers"]["answer1"]["answer"] for entry in sae_dicts] + [
            entry["answers"]["answer2"]["answer"] for entry in sae_dicts
        ]

    readability_functions = {
        "flesch_reading_ease": ts.flesch_reading_ease,  # type: ignore
        "dale_chall_readability_score": ts.dale_chall_readability_score,  # type: ignore
        "syllable_count": ts.syllable_count,  # type: ignore
        "lexicon_count": ts.lexicon_count,  # type: ignore
        "polysllabic_word_count": ts.polysyllabcount,  # type: ignore
        "mcAlpine_EFLAW_readability_score": ts.mcalpine_eflaw,  # type: ignore
        "consensus_readability_score": lambda text: ts.text_standard(  # type: ignore
            text, float_output=True
        ),
    }

    metrics = {
        metric_name: [func(text) for text in texts]
        for metric_name, func in readability_functions.items()
    }

    metrics["passive_constructions_ratio"] = calculate_passive_ratio(texts)

    if verbose:
        print("Readability metrics:")
        for metric, values in metrics.items():
            print(f"### {metric} (mean): {mean(values):.2f}")
        print("############################################################\n")
    return metrics


def plot_readability_metrics(
    readability_metrics: List[dict], output_dir: str = "data/readability_metrics"
) -> None:
    """
    Save a boxplot for each readability metric; each plot compares the five
    different texts: Input (advanced), simple prompt output, complex prompt
    output, Basic English prompt output, and reference (elementary).

    Args:
      readability_metrics (tuple[dict, dict]): A tuple containing two
        dictionaries with readability metrics for SAE and Basic English
        answers, respectively. Each dictionary should be an output of
        'calculate_readability_metrics'.
      output_dir (str): Directory to save the plots.
    """
    # title = "Readability Comparison: SAE vs Basic English Answers --"
    for metric in readability_metrics[0].keys():
        # Set plot style
        sns.set_theme(style="whitegrid")
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=[
                readability_metric[metric] for readability_metric in readability_metrics
            ]
        )
        plt.xticks(
            [0, 1, 2, 3, 4],
            [
                "Advanced Article (Input)",
                "Simple Prompt Rewrite",
                "Complex Prompt Rewrite",
                "Basic English Prompt Rewrite",
                "Elementary Article (Reference)",
            ],
        )
        plt.xticks(rotation=15, ha="right")
        plt.ylabel(metric.replace("_", " ").title())
        # plt.title(f"{title} {metric.replace("_", " ").title()}")
        # Save to file
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}.png")
        plt.close()


def calculate_passive_ratio(texts: List[str]) -> List[float]:
    """
    Calculate the ratio of passive constructions per sentence in a list of
    texts.

    Args:
      texts (List[str]): List of texts to analyze.
    Returns:
      float: Ratio of passive voice in the texts.
    """
    nlp = spacy.load("en_core_web_lg")
    matcher = Matcher(nlp.vocab)
    passive_ratios = []
    for text in texts:
        doc = nlp(text)
        sents = list(doc.sents)
        passive_rule = [
            {"DEP": "nsubjpass"},
            {"DEP": "aux", "OP": "*"},
            {"DEP": "auxpass"},
            {"TAG": "VBN"},
        ]
        matcher.add("Passive", [passive_rule])
        matches = matcher(doc)
        passive_ratios.append(len(matches) / len(sents))

    return passive_ratios


def load_onestopqa(reference: bool = True) -> List[List[str]]:
    """
    Load articles from the OneStopQA dataset.

    Args:
      reference (bool): If True, load articles in Basic English. If False, load
      the original articles (e.g. to then convert them to Basic English).

    Returns:
      List[List[str]]: A list of articles, where each article is a list of
        paragraphs.
    """

    dataset = load_dataset("malmaud/onestop_qa")
    texts = []
    text = []
    count = 0
    title = dataset["train"][0]["title"]  # type: ignore
    range_start = 6 if reference else 0
    train_data = dataset["train"].select(range(range_start, len(dataset["train"]), 9))  # type: ignore

    for paragraph in train_data:
        if paragraph["title"] == title:  # type: ignore
            text.append(paragraph["paragraph"])  # type: ignore
            count += 1
        else:
            title = paragraph["title"]  # type: ignore
            texts.append(text)
            text = [paragraph["paragraph"]]  # type: ignore

    return texts


def calculate_sari_onestopQA(
    path_to_system_output: str, paragraph_count: int = 3
) -> float:
    """
    Calculate SARI for system output generated from the OneStopQA dataset.

    Args:
      path_to_system_output (str): Path to a JSON file containing a list of
        system output articles (e.g. in 'data/readability_metrics').
      paragraph_count (int): Number of paragraphs to consider from each
        article. Should match the number of paragraphs used for system output.

    Returns:
      float: SARI score of the system output.
    """
    # easse package is not available, so this function is stubbed
    # original_articles = load_onestopqa(reference=False)
    # reference_articles = load_onestopqa(reference=True)
    # original_articles_truncated = [
    #     " ".join(paragraph[:paragraph_count]) for paragraph in original_articles
    # ]
    # # reference articles need to be list of lists
    # reference_articles_truncated = [
    #     [" ".join(paragraph[:paragraph_count]) for paragraph in reference_articles]
    # ]
    # with open(path_to_system_output, "r") as file:
    #     generated_articles = json.load(file)
    #
    # corpus_sari_sae = corpus_sari(
    #     orig_sents=original_articles_truncated,
    #     sys_sents=generated_articles,
    #     refs_sents=reference_articles_truncated,
    # )
    #
    # return corpus_sari_sae
    raise NotImplementedError("EASSE package is not available on PyPI. "
                            "This function requires manual installation from source.")


def compute_bertscore(
    system_outputs: List[List[str]],
    reference: List[str],
    verbose: bool = True,
    model_type: str = "distilbert-base-uncased",
) -> List[tuple]:
    """
    Compute BERTScore for a list of system outputs against references.

    Args:
      system_outputs (List[List[str]]): A list of system output
    """
    scores = [
        score(
            results,
            reference,
            lang="en",
            model_type=model_type,
            verbose=False,
        )
        for results in system_outputs
    ]
    if verbose:
        print("BERTScore results:")
        for score_set in scores:
            print(
                f"### Precision: {score_set[0].mean().item():.3f}, Recall: {score_set[1].mean().item():.3f}, F1: {score_set[2].mean().item():.3f}"
            )
        print("############################################################\n")
    return scores


if __name__ == "__main__":
    # this block calculates SARI scores for different prompts on OneStopQA test data
    print("SARI scores for different prompts on OneStopQA test data:")
    onestopqa_path = "data/readability_metrics/onestopqa_test_data/"
    print(
        f"### Basic English prompt: {calculate_sari_onestopQA(onestopqa_path + "basic_english_prompt.json")}"
    )
    print(
        f"### Simple prompt: {calculate_sari_onestopQA(onestopqa_path + "simple_prompt.json")}"
    )
    print(
        f"### Complex prompt: {calculate_sari_onestopQA(onestopqa_path + "complex_prompt.json")}\n"
    )

    # Here we load the input, reference, and the outputs from the three different prompts
    with open(onestopqa_path + "advanced_sentences.json", "r") as file:
        advanced_sentences = json.load(file)

    with open(onestopqa_path + "simple_prompt.json", "r") as file:
        simple_results = json.load(file)

    with open(onestopqa_path + "complex_prompt.json", "r") as file:
        complex_results = json.load(file)

    with open(onestopqa_path + "basic_english_prompt.json", "r") as file:
        basic_english_results = json.load(file)

    with open(onestopqa_path + "elementary_sentences.json", "r") as file:
        elementary_sentences = json.load(file)

    # Calculate readability metrics for input, reference, and outputs
    readability_metrics = [
        calculate_readability_metrics(sentences)
        for sentences in [
            advanced_sentences,
            simple_results,
            complex_results,
            basic_english_results,
            elementary_sentences,
        ]
    ]

    # Plot the calculated readability metrics
    plot_readability_metrics(readability_metrics)

    # Compute DistilBERTScores
    compute_bertscore(
        [
            basic_english_results,
            simple_results,
            complex_results,
        ],
        reference=elementary_sentences,
        verbose=True,
        model_type="distilbert-base-uncased",
    )
    compute_bertscore(
        [
            basic_english_results,
            simple_results,
            complex_results,
        ],
        reference=advanced_sentences,
        verbose=True,
        model_type="distilbert-base-uncased",
    )
