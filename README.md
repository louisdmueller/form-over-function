# Research Project

This is the repository for our research project on evaluating how linguistic biases effect the decisions of modern day LLM-as-a-Judge systems.

# Installation

This project requires Python 3.12 or higher. It's recommended to use a virtual environment.
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

# Config

Create a file `config.yml` or adapt our example config `config.yml.example` and rename it to `config.yml`.

To access the newest Llama models one needs to accept the Meta tos on HuggingFace Hub and provide an api key using the following format:
```yaml
huggingface_hub_token: <key>
```

To generate the translations using GPT-4.1, we use the OpenAI API. Gemini and Claude also require API keys. Include them as follows:
```yaml
openai_key: <key>
anthropic_api_key: <key>
gemini_api_key: <key>
```

Set a `cache_dir` to prevent excessive I/O operations in your home directory.

---

# Project Structure

```
.
├── data/           # Raw and processed data, input for judges
├── outputs/        # Judge outputs and evaluation outputs
├── src/            # Main source code
│   ├── utils/      # Helper functions we use across codebase
│   ├── evaluation/ # Scripts to analyze and evaluate results
│   └── utils/      # Helper functions
├── tasks_files/    # Code and json files to run judges with params
├── closed_models_tasks_files/    # Task files for closed models
├── config.yml      # YAML config file
├── scripts/        # Scripts to run GPU code
```

## Using the code


For better project structure, we use multiple scripts to handle different tasks. Always run the scripts from the project root:

### Generating data by letting models answer the questions
```bash
sbatch scripts/slurm/start-generate-data.sh [answer_generation_model] [output_path]
```

### Rewriting the GPT-4.1 answers to AAVE/simple language/answers with error
```bash
python src/rewrite_text.py
```


### Generating judgements & evaluating them

```bash
sbatch scripts/slurm/start-generate-judgements.sh
```

This script runs through the specified task files until all the judgements have been generated, then computes our metrics and puts the scores in a Excel Table to compare them.

We run the reasoning analysis using 

```bash
python src/evaluation/reasoning_analysis.py
```

