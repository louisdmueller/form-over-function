set -e # Exit on error
set -u # Treat unset variables as an error

# answer_generation_model_name_or_path="gemini-1.5-flash"  # model that generates answers to the questions
# answer_generation_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
# answer_generation_model_name_or_path="openai-community/gpt2"
# answer_generation_model_name_or_path="EleutherAI/gpt-neo-1.3B"
answer_generation_model_name_or_path="EleutherAI/gpt-neox-20b"
# answer_generation_model_name_or_path="huggyllama/llama-13b"
prompt_model_name_or_path="gpt-4o-mini" # model that converts answers to aae answers # only used if --aae is set

python src/generate_answers.py \
    --answer_generation_model_name_or_path "$answer_generation_model_name_or_path" \
    --prompt_model_name_or_path "$prompt_model_name_or_path" \
    --output_path "data/$answer_generation_model_name_or_path-answers.json" \
#     --aae
    # TODO: implement input_path so already generated answers can be translated to aae
