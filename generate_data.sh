#!/bin/bash
set -e # Exit on error
set -u # Treat unset variables as an error

: ' 
Usage: ./generate_data.sh <answer_generation_model> [aae_conversion_model]
Arguments:
    answer_generation_model (required) Name / path of model used to generate answers
                                Example models:
                                - gemini-1.5-flash
                                - meta-llama/Llama-3.1-8B-Instruct
                                - openai-community/gpt2
                                - EleutherAI/gpt-neo-1.3B
                                - EleutherAI/gpt-neox-20b

    aae_conversion_model    (optional) Converts SAE answers to AAE
'

if [[ $# -ge 2 ]]; then
    echo "Converting SAE answers to AAE, since conversion model was given."
    python src/generate_answers.py \
        --answer_generation_model_name_or_path "$1" \
        --prompt_model_name_or_path "$2" \
        --output_path "data/$1-answers.json" \
        --aae
else
   echo "Running generation without converting of answers to AAE."
    python src/generate_answers.py \
        --answer_generation_model_name_or_path "$1" \
        --output_path "data/$1-answers.json"
fi

# TODO: implement input_path so already generated answers can be translated to aae
