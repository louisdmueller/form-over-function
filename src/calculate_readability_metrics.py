from datasets import load_dataset
from easse.sari import corpus_sari
from bert_score import score
from spacy.matcher import Matcher
from typing import List
from utils import read_file, write_file
import json
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import textstat as ts


def calculate_readability_metrics(texts: List[str]) -> dict:
    """
    Calculate readability metrics for a list of texts.

    Args:
      texts (List[str]): List of texts to analyze.

    Returns:
      dict: A dictionary containing the readability metrics.
    """
    readability_functions = {
        "flesch_reading_ease": ts.flesch_reading_ease,
        "dale_chall_readability_score": ts.dale_chall_readability_score,
        "syllable_count": ts.syllable_count,
        "lexicon_count": ts.lexicon_count,
        "polysllabic_word_count": ts.polysyllabcount,
        "mcAlpine_EFLAW_readability_score": ts.mcalpine_eflaw,
        "consensus_readability_score": 
            lambda text: ts.text_standard(text, float_output=True)
    }

    metrics = {
        metric_name: [func(text) for text in texts]
        for metric_name, func in readability_functions.items()
    }

    metrics["passive_constructions"] = calculate_passive_ratio(texts)

    return metrics


def calculate_passive_ratio(texts: List[str]):
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
            {'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'},
            {'DEP': 'auxpass'}, {'TAG': 'VBN'}]
        matcher.add('Passive', [passive_rule])
        matches = matcher(doc)
        passive_ratios.append(len(matches) / len(sents))

    return passive_ratios


def load_onestopqa(reference: bool=True)-> List[List[str]]:
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
    title = dataset["train"][0]["title"]
    range_start = 6 if reference else 0
    train_data = dataset["train"].select(
        range(range_start, len(dataset["train"]), 9))

    for paragraph in train_data:
        if paragraph["title"] == title:
            text.append(paragraph["paragraph"])
            count += 1
        else:
            title = paragraph["title"]
            texts.append(text)
            text = [paragraph["paragraph"]]

    return texts

def calculate_sari(path_to_system_output:str, paragraph_count:int=3) -> float:
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
    original_articles = load_onestopqa(reference=False)
    reference_articles = load_onestopqa(reference=True)
    original_articles_truncated = [" ".join(
        paragraph[:paragraph_count]) for paragraph in original_articles]
    # reference articles need to be list of lists
    reference_articles_truncated = [[" ".join(
        paragraph[:paragraph_count]) for paragraph in reference_articles]]
    with open(path_to_system_output, "r") as file:
        generated_articles = json.load(file)

    corpus_sari_sae = corpus_sari(
        orig_sents=original_articles_truncated, 
        sys_sents=generated_articles, 
        refs_sents=reference_articles_truncated)

    return corpus_sari_sae


def plot_readability_metrics(
        readability_metrics:List[dict], 
        output_dir:str="data/readability_metrics"
    ) -> None:
    """
    Save a boxplot for each readability metric, comparing SAE and Basic English
    answers.

    Args:
      readability_metrics (tuple[dict, dict]): A tuple containing two
        dictionaries with readability metrics for SAE and Basic English
        answers, respectively. Each dictionary should be an output of
        'calculate_readability_metrics'.
      output_dir (str): Directory to save the plots.
    """
    title = "Readability Comparison: SAE vs Basic English Answers --"
    for metric in readability_metrics[0].keys():
        # Set plot style
        sns.set_theme(style="whitegrid")
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[readability_metric[metric] for readability_metric in readability_metrics])
        plt.xticks([0, 1, 2, 3, 4], ["Advanced Article (Input)", "Simple Prompt Rewrite", "Complex Prompt Rewrite", "Basic English Prompt Rewrite", "Elementary Article (Reference)"])
        plt.xticks(rotation=15, ha='right')
        plt.ylabel(metric.replace("_", " ").title())
        # plt.title(f"{title} {metric.replace("_", " ").title()}")
        # Save to file
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}.png")
        plt.close()



### Example usage of the functions defined above ###
# sae_file = "data/generated_answers/gpt-4-original-answers.json"
# sae_dicts = read_file(sae_file)
# basic_file = "data/generated_answers/gpt-4-original-answers_basic.json"
# basic_dicts = read_file(basic_file)

# readability_metrics_sae = calculate_readability_metrics(
#     [entry["answers"]["answer1"]["answer"] for entry in sae_dicts] +
#     [entry["answers"]["answer2"]["answer"] for entry in sae_dicts])

# readability_metrics_basic = calculate_readability_metrics(
#     [entry["answers"]["answer1"]["answer"] for entry in basic_dicts] +
#     [entry["answers"]["answer2"]["answer"] for entry in basic_dicts])

# print("Readability metrics for SAE answers:")
# for metric, values in readability_metrics_sae.items():
#     print(f"{metric}: {values[:5]}...")  # Print first 5 values for brevity
# print("\nReadability metrics for Basic English answers:")
# for metric, values in readability_metrics_basic.items():
#     print(f"{metric}: {values[:5]}...")  # Print first 5 values for brevity

# plot_readability_metrics((readability_metrics_sae, readability_metrics_basic))

# print("SARI scores for different prompts on OneStopQA test data:")
# print("Basic English prompt:")
# print(calculate_sari("data/readability_metrics/onestopqa_test_data/basic_english_prompt.json"))
# print("Simple prompt:")
# print(calculate_sari("data/readability_metrics/onestopqa_test_data/simple_prompt.json"))
# print("Complex prompt:")
# print(calculate_sari("data/readability_metrics/onestopqa_test_data/complex_prompt.json"))


# original_articles = load_onestopqa(reference=True)
# original_articles_truncated = [" ".join(
#         paragraph[:3]) for paragraph in original_articles]
# with open("data/readability_metrics/onestopqa_test_data/reference_sentences.json", "w") as file:
#     json.dump(original_articles_truncated, file, indent=4)



with open("data/readability_metrics/onestopqa_test_data/advanced_sentences.json", "r") as file:
        advanced_sentences = json.load(file)

with open("data/readability_metrics/onestopqa_test_data/simple_prompt.json", "r") as file:
        simple_results = json.load(file)

with open("data/readability_metrics/onestopqa_test_data/complex_prompt.json", "r") as file:
        complex_results = json.load(file)

with open("data/readability_metrics/onestopqa_test_data/basic_english_prompt.json", "r") as file:
        basic_english_results = json.load(file)

with open("data/readability_metrics/onestopqa_test_data/elementary_sentences.json", "r") as file:
        elementary_sentences = json.load(file)


# Compute BERTScore
# scores = [score(results, advanced_sentences, lang="en", model_type="distilbert-base-uncased", verbose=True) for results in [basic_english_results, complex_results, simple_results]]
# print("BERTScore results against original:")
# for score_set in scores:
#     print(f"Precision: {score_set[0].mean().item():.4f}, Recall: {score_set[1].mean().item():.4f}, F1: {score_set[2].mean().item():.4f}")

# scores = [score(results, elementary_sentences, lang="en", model_type="distilbert-base-uncased", verbose=True) for results in [basic_english_results, complex_results, simple_results]]
# print("BERTScore results against elementary reference:")
# for score_set in scores:
#     print(f"Precision: {score_set[0].mean().item():.4f}, Recall: {score_set[1].mean().item():.4f}, F1: {score_set[2].mean().item():.4f}")


readability_metrics = [calculate_readability_metrics(sentences) for sentences in
    [advanced_sentences, simple_results, complex_results, basic_english_results, elementary_sentences]]

# print("Readability metrics for simple answers:")
# for metric, values in readability_metrics_simple.items():
#     print(f"{metric}: {values[:5]}...")  # Print first 5 values for brevity
# print("\nReadability metrics for Basic English answers:")
# for metric, values in readability_metrics_basic.items():
#     print(f"{metric}: {values[:5]}...")  # Print first 5 values for brevity

plot_readability_metrics(readability_metrics)