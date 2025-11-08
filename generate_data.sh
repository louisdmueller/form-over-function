#!/bin/bash
set -e # Exit on error
set -u # Treat unset variables as an error

: ' 
Usage: ./generate_data.sh [answer_generation_model]
Arguments:
    answer_generation_model (optional) Name / path of model used to generate answers
                                Example models:
                                - gemini-1.5-flash
                                - meta-llama/Llama-3.1-8B-Instruct
                                - openai-community/gpt2
                                - EleutherAI/gpt-neo-1.3B
                                - Qwen/Qwen2-0.5B-Instruct

Notes: 
    The optional command line argument will overwrite the value specified in the script.
'
answer_generation_model="gemini-1.5-flash"

if [[ $# -ge 1 ]]; then
    answer_generation_model=$1
fi

echo "Running generation of answers."
echo "Chosen answer generation model: $answer_generation_model"
python src/generate_answers.py \
    --answer_generation_model_name_or_path $answer_generation_model \
    --output_path "data/generated_answers/$answer_generation_model-answers.json"

# TODO: implement input_path so already generated answers can be translated to aae