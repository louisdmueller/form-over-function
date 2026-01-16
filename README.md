# Research Project

This project requires Python 3.12 or higher. It's recommended to use a virtual environment.

# Installation

You can use venv or conda.
## venv
```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## conda
```bash
conda env create -f environment.yml
conda activate venv
python -m pip install -e .
```

To access the newest Llama models one needs to accept the Meta tos on HuggingFace Hub and provide an api key. Create a file `config.yml` or adapt our example config `config.yml.example` using the following format:
```yaml
huggingface_hub_token: <key>
```

To generate the translations using GPT-4.1, we use the OpenAI API. Include your API key as follows:
```yaml
openai_key: <key>
```

---

For better project structure, we use multiple scripts to handle different tasks. Always run the scripts from the project root:

## Generating data and (optionally) translating it to AAE
```bash
bash scripts/slurm/start-generate-data.sh [answer_generation_model]
```

## Generating the models judgements
(Local)
```bash
bash scripts/bash/generate-judgements.sh
```

(SLURM Cluster)
```bash
sbatch scripts/slurm/start-generate-judgements.sh
```

## Rewriting the GPT-4.1 answers to AAVE/simple language
```bash
python src/rewrite_text.py
```

## Evaluating the models judgements
First merge the judgement outputs in to a single file:

Then evaluate the merged judgements:

```bash

```