from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import repo_exists


class Model(ABC):

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path

    @abstractmethod
    def prompt(
        self,
        messages: List[Dict[str, str]],
        num_generations: int,
        max_output_tokens: int,
        **kwargs,
    ) -> Dict:
        pass

    def extract_answer(self, text: str) -> str | None:
        for line in text.strip().split("\n"):
            line_stripped = line.strip()
            if "1" in line_stripped:
                return "answer1"
            elif "2" in line_stripped:
                return "answer2"
            elif "tie" in line_stripped.lower():
                return "tie"
        return None


class HuggingfaceModel(Model):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.model.to(self.model.device)
        self.model.eval()

    def prompt(
        self,
        messages: List[Dict[str, str]],
        num_generations: int,
        max_output_tokens: int,
        **kwargs,
    ) -> Dict:
        with torch.no_grad():
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                tokenize=False,
                add_generation_prompt=True,
                max_length=self.tokenizer.model_max_length,
            )
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )
            inputs = inputs.to(self.model.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                num_beams=num_generations,
                num_return_sequences=num_generations,
                do_sample=False,
                return_dict_in_generate=True,
                **kwargs,
            )

        sequences = output.sequences
        decoded_sequences = self.tokenizer.batch_decode(
            sequences, skip_special_tokens=True
        )
        assistant_responses = [
            # We remove the first part of the output
            # But since the output could also be solely the answer, 
            # we ensure to only strip the last part of the sequence
            sequence[ - len(formatted_input) :].strip() for sequence in decoded_sequences
        ]
        extracted_answers = [
            self.extract_answer(response) for response in assistant_responses
        ]

        result_dict = {
            "output": (
                assistant_responses if num_generations > 1 else assistant_responses[0]
            ),
            "extracted_answers": (
                extracted_answers if num_generations > 1 else extracted_answers[0]
            ),
        }

        return result_dict


class OpenAIModel(Model):
    def __init__(self, model_name_or_path: str, api_key: str):
        super().__init__(model_name_or_path)
        self.openai_client = OpenAI(api_key=api_key)

    def prompt(
        self,
        messages: List[Dict[str, str]],
        num_generations: int,
        max_output_tokens: int,
        **kwargs,
    ) -> Dict:
        assistant_responses = []
        for _ in range(num_generations):
            response = self.openai_client.responses.create(
                model=self.model_name_or_path,
                input=messages,  # type: ignore
                max_output_tokens=max_output_tokens,
                **kwargs,
            )
            assistant_responses.append(response.output_text)

        extracted_answers = [
            self.extract_answer(response) for response in assistant_responses
        ]

        result_dict = {
            "output": (
                assistant_responses if num_generations > 1 else assistant_responses[0]
            ),
            "extracted_answers": (
                extracted_answers if num_generations > 1 else extracted_answers[0]
            ),
        }

        return result_dict


def get_model(model_name_or_path: str, openai_key: Optional[str], **kwargs) -> Model:
    if repo_exists(model_name_or_path, repo_type="model"):
        return HuggingfaceModel(model_name_or_path, **kwargs)
    elif "gpt" in model_name_or_path:
        api_key = openai_key
        if not api_key:
            raise ValueError("API key is required for OpenAI models.")
        return OpenAIModel(model_name_or_path, api_key)
    else:
        raise ValueError(f"Model {model_name_or_path} not supported.")
