import textstat
from utils import read_file, write_file
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

def calculate_readability_metrics(texts: List[str]):
    """
    Calculate readability metrics for a list of texts.
    
    Args:
        texts (List[str]): List of texts to analyze.
    Returns:
        dict: Dictionary containing readability metrics.
    """
    metrics = {
        "flesch_reading_ease": [textstat.flesch_reading_ease(text) for text in texts],
        "dale_chall_readability_score": [textstat.dale_chall_readability_score(text) for text in texts],
        "syllable_count": [textstat.syllable_count(text) for text in texts],
        "lexicon_count": [textstat.lexicon_count(text) for text in texts],
        "polysllabic_word_count": [textstat.polysyllabcount(text) for text in texts],
        "consensus_readability_score": [textstat.text_standard(text, float_output=True) for text in texts],
        "passive_constructions": calculate_passive_ratio(texts)
    }
    
    return metrics

def calculate_passive_ratio(texts: List[str]):
    """
    Calculate the ratio of passive voice in a list of texts.
    
    Args:
        texts (List[str]): List of texts to analyze.
    Returns:
        float: Ratio of passive voice in the texts.
    """
    nlp = spacy.load("en_core_web_sm")
    passive_ratios = []
    for text in texts:
        passive_count = 0
        total_sentences = 0
        doc = nlp(text)

        for sent in doc.sents:
            total_sentences += 1
            if any(token.dep_ in ("aux:pass", "nsubjpass") for token in sent):
                passive_count += 1
                
        passive_ratios.append(passive_count / total_sentences)
        
    return passive_ratios

sae_file = "data/generated_answers/gpt-4-original-answers.json"
sae_dicts = read_file(sae_file)
basic_file = "data/generated_answers/gpt-4-original-answers_basic.json"
basic_dicts = read_file(basic_file)


readability_metrics_sae = calculate_readability_metrics([entry["answers"]["answer1"]["answer"] for entry in sae_dicts] +
                              [entry["answers"]["answer2"]["answer"] for entry in sae_dicts])

readability_metrics_basic = calculate_readability_metrics([entry["answers"]["answer1"]["answer"] for entry in basic_dicts] +
                              [entry["answers"]["answer2"]["answer"] for entry in basic_dicts])

print(readability_metrics_basic["passive_constructions"])
print(readability_metrics_sae["passive_constructions"])
for metric in readability_metrics_sae.keys():

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[readability_metrics_sae[metric], readability_metrics_basic[metric]],)
    plt.xticks([0, 1], ["SAE Answers", "Basic English Answers"])
    plt.ylabel(metric.replace("_", " ").title())
    plt.title("Readability Comparison: SAE vs Basic English Answers -- " + metric.replace("_", " ").title())

    # Save to file
    plt.tight_layout()
    plt.savefig("data/readability_metrics/" + metric + "_boxplot.png")  # Save as PNG
    plt.close()  # Close the figure