from rouge_score import rouge_scorer

from utils.utils import read_data_file

original_answers = read_data_file("data/generated_answers/gpt-4.1-answers.json")
error_answers = read_data_file("data/generated_answers/gpt-4.1_errors-answers.json")

def compute_rouge2_scores(original_answers: list[dict], error_answers: list[dict]) -> list[dict]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    results = []
    for original, error in zip(original_answers, error_answers):
        original_text = original["answers"]["answer1"]["answer"]
        error_text = error["answers"]["answer1"]["answer"]

        scores = scorer.score(original_text, error_text)
        rouge2_score = scores['rouge2'].fmeasure

        result = {
            "question": original["question"],
            "original_answer": original_text,
            "error_answer": error_text,
            "rouge2": rouge2_score,
        }
        results.append(result)
    return results
        
def average_rouge2_scores(results: list[dict]) -> float:
    """Compute the average ROUGE-2 score from the results."""
    total_rouge2 = sum(result["rouge2"] for result in results)
    return total_rouge2 / len(results) if results else 0.0

def compute_standard_deviation(results: list[dict], average: float) -> float:
    """Compute the standard deviation of ROUGE-2 scores from the results."""
    import math
    variance = sum((result["rouge2"] - average) ** 2 for result in results) / len(results) if results else 0.0
    return math.sqrt(variance)
    
if __name__ == "__main__":
    results = compute_rouge2_scores(original_answers, error_answers)
    average_rouge2 = average_rouge2_scores(results)
    stddev_rouge2 = compute_standard_deviation(results, average_rouge2)
    print(f"Average ROUGE-2 Score: {average_rouge2:.4f}")
    print(f"Standard Deviation of ROUGE-2 Scores: {stddev_rouge2:.4f}")