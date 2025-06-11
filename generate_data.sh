#!/bin/bash
set -e # Exit on error
set -u # Treat unset variables as an error

: ' 
Usage: ./generate_data.sh [answer_generation_model] [aae_conversion_model]
Arguments:
    answer_generation_model (optional) Name / path of model used to generate answers
                                Example models:
                                - gemini-1.5-flash
                                - meta-llama/Llama-3.1-8B-Instruct
                                - openai-community/gpt2
                                - EleutherAI/gpt-neo-1.3B
                                - EleutherAI/gpt-neox-20b

    aae_conversion_model    (optional) Converts SAE answers to AAE

Notes: 
    The optional command line arguments will overwrite the values specified in the script.
'
answer_generation_model="gemini-1.5-flash"
aae_conversion_model=""

if [[ $# -ge 1 ]]; then
    answer_generation_model=$1
fi

if [[ $# -ge 2 ]]; then
    aae_conversion_model=$2
fi

if [[ -n "$aae_conversion_model" ]]; then
    echo "Converting SAE answers to AAE, since conversion model was given.
    Chosen answer generation model: $answer_generation_model
    Chosen AAE conversion model: $aae_conversion_model"
    python src/generate_answers.py \
        --answer_generation_model_name_or_path "$answer_generation_model" \
        --prompt_model_name_or_path "aae_conversion_model" \
        --output_path "data/$answer_generation_model-answers.json" \
        --aae
else
   echo "Running generation without converting of answers to AAE."
    echo "Chosen answer generation model: $answer_generation_model"
    python src/generate_answers.py \
        --answer_generation_model_name_or_path $answer_generation_model \
        --output_path "data/$answer_generation_model-answers.json"
fi

# TODO: implement input_path so already generated answers can be translated to aae