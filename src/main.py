import transformers
import torch
import math
from huggingface_hub import login

from utils import get_df_from_file, parse_args, load_config

def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)

    # access to llama models is restricted
    login(token=config["huggingface_hub_token"])

    data_df = get_df_from_file(args.data_path)

    prompt = {
        "system": "### You are an excellent evaluator.\n### Your assignment involves providing evaluations for given responses.\n### Each evaluation will consist of *a question* and *two corresponding answers*. Your task is to discern which response is superior based on the **quality of the answer**. If you find that both responses are equally good or bad , feel free to select a tie. There is **no obligation** to favor one response over the other; if a decision cannot be made , a **tie would be an optimal choice**.\n### During the evaluation process , please focus exclusively on the **semantic quality** of the answers. Non semantic factors should not influence your evaluation. The primary focus should be on the ** quality and accuracy** of the answers.\n### You should ONLY output your vote 'Answer1 ', or 'Answer2 ', or 'Tie ' in the last line.", 
        "template": "~~~ Question\n{question}\n ~~~\n ~~~ Answer1 \n{answer1}\n ~~~\n ~~~ Answer2 \n{answer2}\n ~~~"
    }

    model = transformers.T5ForConditionalGeneration.from_pretrained(args.prompt_model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.prompt_model_name_or_path)
    tokenizer.model_max_length = 1024

    system_prompt = prompt["system"]

    with torch.no_grad():
        for idx, data in data_df.iterrows():
            question = data["question"]
            answer1 = data["answers"]["answer1"]["answer"]
            answer2 = data["answers"]["answer2"]["answer"]

            input_text = system_prompt + prompt["template"].format(
                question=question, answer1=answer1, answer2=answer2
            )
            inputs = tokenizer(input_text, return_tensors="pt")

            output = model.generate(
                inputs["input_ids"],
                num_beams=6,
                num_return_sequences=6,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )

            # Diese Scores sind log-probs: höher = besser (weniger negativ)
            sequence_scores = output.sequences_scores.tolist()

            results = []
            for i, (sequence, log_score) in enumerate(zip(output.sequences, sequence_scores)):
                decoded_text = tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # convert log prob to probability
                prob_score = math.exp(log_score)
                results.append((decoded_text, prob_score))

            for i, (text, score) in enumerate(results):
                print(f"Generated Text {i + 1}: {text}")
                print(f"Sequence Score: {score:.4f}")

            break

if __name__ == "__main__":
    main()