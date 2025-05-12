# Research Project

Currently this project uses Python version 3.12.10. 

This project uses venv. To create a venv execute `python -m venv venv`. Activate it via `source venv/bin/activate` and install the packages via `pip install -r requirements.txt`.

For testing purposes the default model is set to t5-small. This can be changed in the future.

To access the newest Llama models one needs to accept the Meta tos on HuggingFace Hub and provide an api key. Create a file `config.yml` in the following format:
```yaml
huggingface_hub_token: <key>
```

To generate the translations using GPT4o mini, we use the OpenAI API. Include your API key as follows:
```yaml
openai_key: <key>
```
In the future this config file may be used to store other configurations as well.