import textstat as ts
from utils import read_file, write_file
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.matcher import Matcher
from easse.sari import corpus_sari


def calculate_readability_metrics(texts: List[str]):
    """
    Calculate readability metrics for a list of texts.

    Args:
        texts (List[str]): List of texts to analyze.
    Returns:
        dict: Dictionary containing readability metrics.
    """
    readability_functions = {
        "flesch_reading_ease": ts.flesch_reading_ease,
        "dale_chall_readability_score": ts.dale_chall_readability_score,
        "syllable_count": ts.syllable_count,
        "lexicon_count": ts.lexicon_count,
        "polysllabic_word_count": ts.polysyllabcount,
        "consensus_readability_score": lambda text: ts.text_standard(text, float_output=True)
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
        passive_rule = [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'},
                        {'DEP': 'auxpass'}, {'TAG': 'VBN'}]
        matcher.add('Passive', [passive_rule])
        matches = matcher(doc)
        passive_ratios.append(len(matches) / len(sents))

    return passive_ratios


sae_file = "data/generated_answers/gpt-4-original-answers.json"
sae_dicts = read_file(sae_file)
basic_file = "data/generated_answers/gpt-4-original-answers_basic.json"
basic_dicts = read_file(basic_file)


readability_metrics_sae = calculate_readability_metrics(
    [entry["answers"]["answer1"]["answer"] for entry in sae_dicts] +
    [entry["answers"]["answer2"]["answer"] for entry in sae_dicts])

readability_metrics_basic = calculate_readability_metrics(
    [entry["answers"]["answer1"]["answer"] for entry in basic_dicts] +
    [entry["answers"]["answer2"]["answer"] for entry in basic_dicts])

print("Readability metrics for SAE answers:")
for metric, values in readability_metrics_sae.items():
    print(f"{metric}: {values[:5]}...")  # Print first 5 values for brevity
print("\nReadability metrics for Basic English answers:")
for metric, values in readability_metrics_basic.items():
    print(f"{metric}: {values[:5]}...")  # Print first 5 values for brevity

exit(0)
for metric in readability_metrics_sae.keys():

    # Set plot style
    sns.set_theme(style="whitegrid")
    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[readability_metrics_sae[metric],
                      readability_metrics_basic[metric]],)
    plt.xticks([0, 1], ["SAE Answers", "Basic English Answers"])
    plt.ylabel(metric.replace("_", " ").title())
    plt.title("Readability Comparison: SAE vs Basic English Answers -- " +
              metric.replace("_", " ").title())
    # Save to file
    plt.tight_layout()
    plt.savefig("data/readability_metrics/" +
                metric + "_boxplot.png")
    plt.close()
