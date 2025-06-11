# Research Project

Currently this project uses Python version 3.12.10. 

This project uses venv. To create a venv execute `python -m venv venv`. Activate it via `source venv/bin/activate` and install the packages via `pip install -r requirements.txt`.

To access the newest Llama models one needs to accept the Meta tos on HuggingFace Hub and provide an api key. Create a file `config.yml` in the following format:
```yaml
huggingface_hub_token: <key>
```

To generate the translations using GPT-4.1, we use the OpenAI API. Include your API key as follows:
```yaml
openai_key: <key>
```

---

For better project structure, we use multiple scripts to handle different tasks:

## Generating data and (optionally) translating it to AAE
```bash
bash generate_data.sh [answer_generation_model] [aae_conversion_model]
```

## Generating the models judgements
(Local)
```bash
bash generate_judgements.sh
```

(SLURM Cluster)
```bash
sbatch start-generate-judgements.sh
```

## Evaluating the models judgements
First merge the judgement outputs in to a single file:
```bash
python merge_json_files.py
```

Then evaluate the merged judgements:

```bash

```