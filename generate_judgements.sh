#!/bin/bash

# In this approach we are comparing answers from a worse (weaker) model to the answers from a better (stronger) model.
# All answers are in SAE. Only in further steps we compare SAE answers to AAV answers.
# Original answers are from the paper "Chen et al.: Humans or LLMs as the Judge? A Study on Judgement Bias" 
# https://github.com/FreedomIntelligence/Humans_LLMs_Judgement_Bias

set -e # Exit on error
set -u # Treat unset variables as an error

answer_generation_model_name_or_path="gemini-1.5-flash-8b"
better_model_name="gemini-1.5-pro-8b"
#judge_model_name="meta-llama/Llama-3.3-70B-Instruct"
#judge_model_name="RandomAnswer"
judge_model_name="meta-llama/Llama-3.1-8B-Instruct"


### 0. Generate worse answers
# python src/generate_answers.py \
#     --answer_generation_model_name_or_path "$answer_generation_model_name_or_path" \
#     --output_path "data/chen-et-al/$answer_generation_model_name_or_path-answers.json"
#     --aae

### 1. Compare answers from the worse model to the answers from the better model
python src/compare_model_answers_batched.py \
    --judge_model_name_or_path "$judge_model_name" \
    --data_1_path "data/gpt-4-original-answers.json" \
    --data_2_path "data/gemini-1.5-flash-answers.json" \
    --start_index 0.5
#    --end_index 0.5
    # --question_switching # TODO
    # --prompt_switchging # TODO

