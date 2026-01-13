import random

import pandas as pd
from nltk.metrics import edit_distance

from src.utils.utils import read_jsonl_file


def compute_length_difference(original_answer: str, other_answer: str) -> int:
    """
    Compute the difference in length between the original answer and the other answer.

    Args:
        original_answer (str): The original answer text.
        other_answer (str): The other answer text.
    Returns:
        int: The difference in length (other length - original length).
    """
    return len(other_answer) - len(original_answer)

def compute_type_token_ratio(answer: str) -> float:
    """
    Compute the type-token ratio of the given answer.

    Args:
        answer (str): The answer text.

    Returns:
        float: The type-token ratio.
    """
    tokens = answer.split()
    types = set(tokens)
    if len(tokens) == 0:
        return 0.0
    return len(types) / len(tokens)

def compute_type_token_ratio_difference(original_answer: str, other_answer: str) -> float:
    """
    Compute the difference in type-token ratio between the original answer and the other answer.

    Args:
        original_answer (str): The original answer text.
        other_answer (str): The other answer text.
    Returns:
        float: The difference in type-token ratio (other TTR - original TTR).
    """
    original_ttr = compute_type_token_ratio(original_answer)
    other_ttr = compute_type_token_ratio(other_answer)
    return other_ttr - original_ttr

def compute_edit_distance(original_answer: str, other_answer: str) -> int:
    """
    Compute the edit distance between the original answer and the other answer.

    Args:
        original_answer (str): The original answer text.
        other_answer (str): The other answer text.
    Returns:
        int: The edit distance.
    """
    return edit_distance(original_answer, other_answer)

def shuffle_answers(original_answers: list[str], aave_answers: list[str], simple_answers: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Shuffle the original and other answers.
    Args:
        original_answers (list[str]): List of original answers.
        aave_answers (list[str]): List of AAVE answers.
        simple_answers (list[str]): List of simple answers.
        
    Returns:
        tuple[list[str], list[str], list[str]]: Shuffled original, AAVE, and simple answers.
    """
    shuffled_original_answers = original_answers.copy()
    shuffled_aave_answers = aave_answers.copy()
    shuffled_simple_answers = simple_answers.copy()
    combined = list(zip(shuffled_original_answers, shuffled_aave_answers, shuffled_simple_answers))
    random.shuffle(combined)
    shuffled_original_answers[:], shuffled_aave_answers[:], shuffled_simple_answers[:] = zip(*combined)
    
    
    return shuffled_original_answers, shuffled_aave_answers, shuffled_simple_answers

def average(lst: list[float], round_digits: int=2) -> float:
    """
    Compute the average of a list of numbers.
    Args:
        lst (list[float]): List of numbers.
        
    Returns:
        float: The average value.
    """
    if not lst:
        return 0.0
    return round(sum(lst) / len(lst), round_digits)

if __name__ == "__main__":
    original_data = read_jsonl_file("data/generated_answers/gpt-4.1-answers.json")
    aave_data = read_jsonl_file("data/generated_answers/gpt-4.1-answers_aae.json")
    simple_data = read_jsonl_file("data/generated_answers/gpt-4.1-answers_simple.json")
    
    original_answers = [entry["answers"]["answer1"]["answer"] for entry in original_data]
    aave_data_answers = [entry["answers"]["answer1"]["answer"] for entry in aave_data]
    simple_data_answers = [entry["answers"]["answer1"]["answer"] for entry in simple_data]
    
    shuffled_original_answers, shuffled_aave_answers, shuffled_simple_answers = shuffle_answers(original_answers, aave_data_answers, simple_data_answers)
    
    results = []
    for original_answer, aave_answer, simple_answer in zip(shuffled_original_answers, shuffled_aave_answers, shuffled_simple_answers):
        length_diff_aae = compute_length_difference(original_answer, aave_answer)
        ttr_diff_aae = compute_type_token_ratio_difference(original_answer, aave_answer)
        edit_dist_aae = compute_edit_distance(original_answer, aave_answer)
        
        length_diff_simple = compute_length_difference(original_answer, simple_answer)
        ttr_diff_simple = compute_type_token_ratio_difference(original_answer, simple_answer)
        edit_dist_simple = compute_edit_distance(original_answer, simple_answer)
        
        results.append({
            "original_answer": original_answer,
            "aave_answer": aave_answer,
            "simple_answer": simple_answer,
            "length_difference_aae": length_diff_aae,
            "ttr_difference_aae": ttr_diff_aae,
            "edit_distance_aae": edit_dist_aae,
            "length_difference_simple": length_diff_simple,
            "ttr_difference_simple": ttr_diff_simple,
            "edit_distance_simple": edit_dist_simple,
        })
    
    result_df = pd.DataFrame(results)
    
    print("Average AAVE Length Difference:", average(result_df["length_difference_aae"].tolist()))
    print("Average AAVE Type-Token Ratio Difference:", average(result_df["ttr_difference_aae"].tolist()))
    print("Average AAVE Edit Distance:", average(result_df["edit_distance_aae"].tolist()))
    
    print("Average Simple Length Difference:", average(result_df["length_difference_simple"].tolist()))
    print("Average Simple Type-Token Ratio Difference:", average(result_df["ttr_difference_simple"].tolist()))
    print("Average Simple Edit Distance:", average(result_df["edit_distance_simple"].tolist()))
    
    result_df.to_excel("rewriting_analysis_results.xlsx", index=False)