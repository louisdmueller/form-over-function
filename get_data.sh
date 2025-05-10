!/bin/bash

# make this script fail if any command fails
set -e

mkdir -p ./data/chen-et-al
cd ./data/chen-et-al
curl -O https://raw.githubusercontent.com/FreedomIntelligence/Humans_LLMs_Judgement_Bias/refs/heads/main/data/raw.json