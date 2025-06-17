set -e # Exit on error
set -u # Treat unset variables as an error


# gpt4_vs_gpt2_file="data/chen-et-al/gpt-4_vs_gpt2/merged_data.json"
# gpt4_aae_vs_gpt2_file="data/chen-et-al/gpt-4_aae_vs_gpt2/merged_data.json"


#python src/analyze_results.py \
#    --file1 "data/chen-et-al/gpt-4_vs_gemini-1.5-flash/merged_data.json" \
#    --file2 "data/chen-et-al/gpt-4_aae_vs_gemini-1.5-flash/merged_data.json" \

python src/analyze_results.py \
    --file1 "data/GPT4.1-vs-Llama3.1-8B/llama/merged_data.json" \
    --file2 "data/GPT4.1_aae-vs-Llama3.1-8B/llama/merged_data.json" \
    --better_model gpt-4.1\
    --worse_model meta-llama/Llama-3.1-8B-Instruct

python src/analyze_results.py \
    --file1 "data/GPT4.1-vs-Llama3.1-8B/qwen/merged_data.json" \
    --file2 "data/GPT4.1_aae-vs-Llama3.1-8B/qwen/merged_data.json" \
    --better_model gpt-4.1\
    --worse_model meta-llama/Llama-3.1-8B-Instruct

python src/analyze_results.py \
    --file1 "data/GPT4.1-vs-Llama3.1-8B/mistral/merged_data.json" \
    --file2 "data/GPT4.1_aae-vs-Llama3.1-8B/mistral/merged_data.json" \
    --better_model gpt-4.1\
    --worse_model meta-llama/Llama-3.1-8B-Instruct

