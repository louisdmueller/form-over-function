# Importing necessary libraries
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calc_similarities(list1, list2):
    """Calculate cosine similarities between pairs of texts."""
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sims = []
    for a, b in zip(list1, list2):
        emb = model.encode([a, b])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        sims.append(sim)
    return np.array(sims)


if __name__ == "__main__":
    exclude_files = [
        "gpt-3.5-turbo-answers_aae.json",
        "gpt-3.5-turbo-answers.json",
        "gpt-4-original-answers_aae.json",
        "gpt-4-original-answers.json",
        "gpt-neo-1.3B-answers.json",
        "gpt-neox-20b-answers-temperature-0.5.json",
        "gpt-neox-20b-answers.json",
        "gpt2-answers.json",
        "llama-13b-answers.json",
        "Mistral-7B-Instruct-v0.1-answers.json"
    ]

    data_dir = "data"
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json") and f not in exclude_files])

    # Load GPT-4.1 file as reference
    gpt_file = [f for f in files if "gpt-4.1" in f][0]
    with open(os.path.join(data_dir, gpt_file), "r") as file:
        gpt_data = [json.loads(line) for line in file]
        gpt_answers1 = [item['answers']["answer1"]["answer"] for item in gpt_data]
        gpt_answers2 = [item['answers']["answer2"]["answer"] for item in gpt_data]

    # 1. answer1 vs answer2 from GPT-4.1
    sims_gpt = calc_similarities(gpt_answers1, gpt_answers2)
    print(f"GPT-4.1 answer1 vs answer2:")
    print(f"  Avg similarity: {np.mean(sims_gpt):.4f}")
    print(f"  Median: {np.median(sims_gpt):.4f}")
    print(f"  Std: {np.std(sims_gpt):.4f}\n")

    # 2. answer1 from GPT-4.1 vs answer1 from all other files
    for other_file in files:
        if other_file == gpt_file:
            continue
        with open(os.path.join(data_dir, other_file), "r") as file:
            other_data = [json.loads(line) for line in file]
            other_answers1 = [item['answers']["answer1"]["answer"] for item in other_data]
        sims = calc_similarities(gpt_answers1, other_answers1)
        print(f"{gpt_file} answer1 vs {other_file} answer1:")
        print(f"  Avg similarity: {np.mean(sims):.4f}")
        print(f"  Median: {np.median(sims):.4f}")
        print(f"  Std: {np.std(sims):.4f}\n")