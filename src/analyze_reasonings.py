import json
import re
from typing import Any

import nltk
import openai
import spacy
from bertopic import BERTopic
from bertopic.representation import OpenAI
from gguf import Path
from pandas import DataFrame
from wordcloud import WordCloud

from prompts import cluster_descriptions_prompt, extract_reasons_prompt
from utils import load_config

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")

def extract_reasoning_from_files(
    file_data: dict, better_model_name: str, worse_model_name: str
) -> list[dict[str, Any]]:
    """
    Extract reasoning from a dictionary containing file data
    """
    results = []
    for question_group_key in file_data.keys():
        if question_group_key == "metadata":
            continue

        question_group = file_data[question_group_key]

        for comparison in question_group:
            for idx, result_text in enumerate(comparison["result"]):
                extracted = comparison["extracted_answers"][idx]

                if extracted == better_model_name:
                    winner = better_model_name
                    loser = worse_model_name
                elif extracted == worse_model_name:
                    winner = worse_model_name
                    loser = better_model_name
                else:
                    winner = None
                    loser = None
                           

                results.append(
                    {
                        "question": comparison["question"],
                        "result_text": result_text,
                        "winner": winner,
                        "loser": loser,
                        "answer_order": {better_model_name: "Answer1" if comparison["answer1"]["label"] == better_model_name else "Answer2",
                                         worse_model_name: "Answer1" if comparison["answer1"]["label"] == worse_model_name else "Answer2"}
                    }
                )

    return results

def extract_strengths_weaknesses_llm(results: list[dict[str, Any]], openai_client, file_name) -> list[dict[str, Any]]:
    """Extract structured strengths and weaknesses using an OpenAI LLM."""
    all_extractions = []
    if Path(file_name).exists():
        with open(file_name, "r") as f:
            all_extractions = json.load(f)
        return all_extractions

    for result in results:
        if not result["winner"]:
            continue
        
        prompt = extract_reasons_prompt(result['result_text'], result["answer_order"].get(result['winner']), result["answer_order"].get(result['loser']))


        try:
            response = openai_client.responses.create(model="gpt-4.1", instructions= "You are an expert at extracting key information from text.", input=prompt, max_output_tokens=500)

            content = response.output_text
            content = re.sub(r"```json\n?|\n?```", "", content).strip()
            extraction = json.loads(content)

            all_extractions.append(
                {
                    "winner": result["winner"],
                    "loser": result["loser"],
                    "strengths": extraction.get("strengths", []),
                    "weaknesses": extraction.get("weaknesses", []),
                    "original_text": result["result_text"],
                }
            )

        except Exception as e:
            print(f"  Warning: Failed to extract from one result: {e}")
            continue

    with open(file_name, "w") as f:
        json.dump(all_extractions, f, indent=2)

    return all_extractions

def cluster_descriptions(descriptions: list[str], config: dict) -> DataFrame:
    """
    Cluster similar descriptions using a topic model
    """

    client = openai.OpenAI(api_key=config["openai_key"])
    representation_model = OpenAI(client, model="gpt-4.1", chat=True, prompt=cluster_descriptions_prompt)
    topic_model = BERTopic(representation_model=representation_model, n_gram_range=(1, 3), min_topic_size=20)
    topics, _ = topic_model.fit_transform(descriptions)
    topic_info = topic_model.get_topic_info()
    return topic_info

def print_clustered_descriptions(clustered_descriptions: dict[int, list[str]]):
    """
    Print clustered descriptions
    """

    for cluster_id, descriptions in clustered_descriptions.items():
        counter = 0
        print(f"Cluster {cluster_id}:")
        for desc in descriptions:
            if counter >= 10:
                break
            counter += 1
            print(f" - {desc}")
        print("\n")
        
def extract_model_descriptions(
    strengths_weaknesses: list[dict[str, Any]], model_name: str):
    """
    Extract strengths and weaknesses for a specific model
    """
    descriptions = []
    for extraction in strengths_weaknesses:
        if extraction["winner"] == model_name:
            descriptions.extend(extraction["strengths"])
        elif extraction["loser"] == model_name:
            descriptions.extend(extraction["weaknesses"])
    return descriptions

def word_cloud_from_cluster_df(cluster_df: DataFrame, output_path: str):
    """
    Generate word clouds from clustered descriptions DataFrame
    """
    word_freqs = {}
    for _, row in cluster_df.iterrows():
        if row["Topic"] == -1:
            continue
        topic_name = row["Representation"][0]
        count = row["Count"]
        word_freqs[topic_name] = count
    word_cloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freqs)
    word_cloud.to_file(f"{output_path}")

def analyze_reasonings_topic_model(
    file_data: dict,
    output_directory: str,
    better_model_name: str,
    worse_model_name: str,
    strengts_weaknesses_file: str = "extracted_strengths_weaknesses.json",
):
    """
    Analyze results using topic modeling
    """
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words.extend(["answer1", "answer2"])

    config = load_config("config.yml")
    api_key = config["openai_key"]
    client = openai.OpenAI(api_key=api_key)

    reasonings = extract_reasoning_from_files(
        file_data, better_model_name, worse_model_name
    )
    strengths_weaknesses = extract_strengths_weaknesses_llm(reasonings, client, strengts_weaknesses_file)
    
    better_model_descriptions = extract_model_descriptions(strengths_weaknesses, better_model_name)
    better_model_clusters = cluster_descriptions(better_model_descriptions, config)
    print("Better Model Clusters:")
    print(better_model_clusters)
    word_cloud_from_cluster_df(better_model_clusters, f"{output_directory}/{better_model_name}_wordcloud.png")
    
    worse_model_descriptions = extract_model_descriptions(strengths_weaknesses, worse_model_name)
    
    worse_model_clusters = cluster_descriptions(worse_model_descriptions, config)
    print("Worse Model Clusters:")
    print(worse_model_clusters)
    word_cloud_from_cluster_df(worse_model_clusters, f"{output_directory}/{worse_model_name}_wordcloud.png")
    
    
if __name__ == "__main__":
    with open("/home/hd/hd_hd/hd_go226/projects/research-project/debug/judgements/gpt-4.1/vs_Llama-3.1-8B-Instruct/phi-4/judgements.json", "r") as f:
        file_data_baseline = json.load(f)
    analyze_reasonings_topic_model(file_data_baseline, "wordclouds/baseline", "gpt-4.1", "Llama-3.1-8B-Instruct", "output/baseline/extracted_strengths_weaknesses_baseline.json"
    )
    with open("/home/hd/hd_hd/hd_go226/projects/research-project/debug/judgements/gpt-4.1_aae/vs_Llama-3.1-8B-Instruct/phi-4/judgements.json", "r") as f:
        file_data_aae = json.load(f)
    analyze_reasonings_topic_model(file_data_aae, "wordclouds/aae", "gpt-4.1", "Llama-3.1-8B-Instruct", "output/aae/extracted_strengths_weaknesses_aae.json"
    )