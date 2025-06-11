set -e # Exit on error
set -u # Treat unset variables as an error


# gpt4_vs_gpt2_file="data/chen-et-al/gpt-4_vs_gpt2/merged_data.json"
# gpt4_aae_vs_gpt2_file="data/chen-et-al/gpt-4_aae_vs_gpt2/merged_data.json"


python src/analyze_results.py \
    "data/chen-et-al/gpt-4_vs_gemini-1.5-flash/merged_data.json" \
    "data/chen-et-al/gpt-4_aae_vs_gemini-1.5-flash/merged_data.json" \

python src/analyze_results.py \
    "data/chen-et-al/gpt4_vs_gpt2/results-2025-06-10_10-48-12-gpt-4-openai-community_gpt2.json" \
    "data/chen-et-al/gpt-4_aae_vs_gpt2/results-2025-06-10_11-35-05-gpt-4-openai-community_gpt2.json" \

