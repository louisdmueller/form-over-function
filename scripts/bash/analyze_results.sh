set -e # Exit on error
set -u # Treat unset variables as an error

export PYTHONPATH="$(pwd):$PYTHONPATH"
# Analysis script for comparing judgements
cd src
python -u evaluation/analyze_results.py \
    --file1 "data/judgements/GPT4.1-vs-Qwen2-0.5B-Instruct/mistral-7B/merged_data.json" \
    --file2 "data/judgements/GPT4.1_aee-vs-Qwen2-0.5B-Instruct/mistral-7B/merged_data.json" \
    --better_model gpt-4.1 \
    --worse_model Qwen/Qwen2-0.5B-Instruct
cd ..