from abc import ABC, abstractmethod
import re
from typing import List, Dict
from openai import OpenAI
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
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
        # If there are multiple answers in the last 100 characters, return None
        if sum([re.search(rf'{answer}', text[-100:], flags=re.IGNORECASE) is not None 
                for answer in ["Answer1", "Answer2", "Tie"]]) > 1:
            return None
        
        # Use regex to find the last occurrence of 'Answer1', 'Answer2', or 'Tie'
        matches = re.findall(r'^\s*(Answer1|Answer2|Tie)\s*$', text, flags=re.MULTILINE | re.IGNORECASE)

        if matches:
            # Take the last match as the vote
            return matches[-1].lower()
        else:
            return None


class HuggingfaceModel(Model):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "92GB", 1: "92GB"} # H100 has 94GB, leave some reserve
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

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
            input_length = inputs['input_ids'].shape[1]

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
        # Remove the input part from the generated sequences
        generated_sequences = sequences[:, input_length:]
        decoded_sequences = self.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )
        assistant_responses = [
            # We remove the first part of the output
            # But since the output could also be solely the answer, 
            # we ensure to only strip the last part of the sequence
            sequence.strip() for sequence in decoded_sequences
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
    
class GeminiModel(OpenAIModel):
    def __init__(self, model_name_or_path: str, api_key: str):
        Model.__init__(self, model_name_or_path)
        self.client = genai.Client(api_key=api_key)

    def generate_answers(
            self,
            message: str,
            num_generations: int,
        ) -> list[str]:
        assistant_responses = []
        for _ in range(num_generations):
            response = self.client.models.generate_content(
                model=self.model_name_or_path,
                contents=[message],
            )

            assistant_responses.append(
                response.text if hasattr(response, "text") else str(response)
            )

        return assistant_responses

def get_model(model_name_or_path: str, config: dict, **kwargs) -> Model:
    if repo_exists(model_name_or_path, repo_type="model"):
        # Some models are restricted and require a token to access
        from huggingface_hub import login
        if not (token := config.get("huggingface_hub_token")):
            raise ValueError("Hugging Face Hub token is required for Hugging Face models.")
        login(token)
        
        return HuggingfaceModel(model_name_or_path, **kwargs)
    
    elif "gpt" in model_name_or_path:
        if not (api_key := config.get("openai_key")):
            raise ValueError("API key is required for OpenAI models.")
        return OpenAIModel(model_name_or_path, api_key)
    
    elif "gemini" in model_name_or_path:
        if not (api_key := config.get("gemini_key")):
            raise ValueError("API key is required for OpenAI models.")
        return GeminiModel(model_name_or_path, api_key)
    
    else:
        raise ValueError(f"Model {model_name_or_path} not supported.")
