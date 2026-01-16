"""
This script calculates the cost of using Claude to generate answers. 
As reference it uses an already existing set of inputs and outputs.
This is because you cannot infer the length of the output by the instruction alone,
as the model might generate different lengths of output even if prompted not to do so.
In the future, this script should be extended to also calculate the cost of generating the inputs,
translation to aae and it should support different models.
"""

import json

import anthropic


def calculate_input_length(inputs: dict, prompt_style, model) -> tuple:
    with open("data/chen-et-al/prompts.json", "r") as f:
        prompts = json.load(f)
    if prompt_style not in prompts:
        raise ValueError(f"Prompt style '{prompt_style}' not found in prompts.json.")
    system_prompt = prompts[prompt_style]["system"]
    
    # input_words = 0
    # input_chars = 0
    # output_words = 0
    # output_chars = 0
    input_tokens = 0
    output_tokens = 0
    # for question in inputs.keys():
    #     if question == "metadata":
    #         continue
    #     for answer_perturbation in inputs[question]:
            # input_words += len(system_prompt.split())
            # input_words += len(answer_perturbation["question"].split())
            # input_words += len(answer_perturbation["answer1"]["text"].split())
            # input_words += len(answer_perturbation["answer2"]["text"].split())

            # input_chars += len(system_prompt)
            # input_chars += len(answer_perturbation["question"])
            # input_chars += len(answer_perturbation["answer1"]["text"])
            # input_chars += len(answer_perturbation["answer2"]["text"])

            # output_words += sum(len(answer.split()) for answer in answer_perturbation["result"])

            # output_chars += sum(len(answer) for answer in answer_perturbation["result"])

    # Accumulate all input and output messages
    input_messages_list = []
    output_messages_list = []
    for question in inputs.keys():
        if question == "metadata":
            continue
        for answer_perturbation in inputs[question]:
            input_content = system_prompt + answer_perturbation["question"] + answer_perturbation["answer1"]["text"] + answer_perturbation["answer2"]["text"]
            input_messages_list.append({"role": "user", "content": input_content})
            output_content = answer_perturbation["result"][0]
            output_messages_list.append({"role": "assistant", "content": output_content})

    # Count tokens in batch
    anthropic_client = anthropic.Anthropic()
    input_tokens = anthropic_client.messages.count_tokens(messages=input_messages_list, model=model).input_tokens
    output_tokens = anthropic_client.messages.count_tokens(messages=output_messages_list, model=model).input_tokens

    return input_tokens, output_tokens

def calculate_cost(
    inputs: dict,
    prompt_style: str,
    models: list,
    batch: bool = False,
    upper_limit: int = 9999999,
    num_generations: int = 3
):
    with open("model-prices.json") as f:
        model_to_price = json.load(f)
        
    for model in models:
        if model not in model_to_price:
            raise ValueError(f"Model '{model}' not found in model-prices.json.")
        inp_tokens, out_tokens = calculate_input_length(inputs, prompt_style, model)

        # multiply tokens by number of generations
        ## this also needs to be done for input tokens, since we do not set the model to generate three times
        ## but prompt the model three times with the same prompt
        inp_tokens *= num_generations
        out_tokens *= num_generations

        total_cost = 0
        price_basis = 1_000_000  # price is calculated per 1 mio tokens
        if batch:
            input_token_price = model_to_price[model]["input"] / 2
            output_token_price = model_to_price[model]["output"] / 2
        else:
            input_token_price = model_to_price[model]["input"]
            output_token_price = model_to_price[model]["output"]
        inp_cost = (inp_tokens / price_basis) * input_token_price
        out_cost = (out_tokens / price_basis) * output_token_price
        cost = inp_cost + out_cost
        total_cost += cost
        print(
            f"Cost for {model}:\n"
            f"\tInput words:  {inp_tokens:.0f} input tokens  ${inp_cost:.2f}\n"
            f"\tOutput words: {out_tokens:.0f} output tokens ${out_cost:.2f}\n"
            f"\tTotal cost:   ${cost:.2f}\n"
        )


if __name__ == "__main__":
    with open("data/judgements/GPT4.1-vs-Llama3.1-8B/Mistral-7B-Instruct-v0.2/merged_data.json", "r") as f:
        inputs = json.load(f)
    calculate_cost(
        inputs=inputs,
        prompt_style="directly_answer_question_without_cot",
        models=["claude-3-5-haiku-latest", "claude-sonnet-4-20250514"],
        batch=False,
    )
