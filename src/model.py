from abc import ABC, abstractmethod
import re
from typing import List, Dict
from openai import OpenAI
from google import genai
from google.genai.types import GenerateContentConfig
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
import torch
from huggingface_hub import repo_exists


class Model(ABC):

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path

    @abstractmethod
    def prompt(
        self,
        system_prompt: str,
        message: str,
        # num_generations: int,
        # max_output_tokens: int,
        **kwargs,
    ) -> Dict:
        pass

    def query_model(
        self,
        message: str,
        num_generations: int,
        system_prompt: str = "",
        # max_output_tokens: int,
        **kwargs: dict[str, int],
    ) -> List[str]:
        raise NotImplementedError(
            "This model does not support generating answers directly from a message."
        )
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]] | str,
    ) -> List[Dict[str, str]] | str:
        raise NotImplementedError(
            "This model does not support applying chat templates directly."
        )
    
    def prompt_batched(
        self,
        system_prompts: list[str],
        input_texts: list[str],
        num_generations: int = 1,
        max_output_tokens: int = 512,
        batch_size: int = 8,
        **kwargs,
    ) -> dict:
        raise NotImplementedError(
            "This model does not support applying chat templates directly."
        )
        
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
        self.tokenizer.padding_side = "left"
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
    
    def prompt_batched(
        self,
        system_prompts: list[str],
        input_texts: list[str],
        num_generations: int = 1,
        max_output_tokens: int = 512,
        batch_size: int = 12,
        **kwargs,
    ) -> dict:
        """
        Batched inference for HuggingfaceModel.
        """
        print(f"Processing {len(input_texts)} examples")
        print(f"Batch size: {batch_size}")
        print(f"Number of generations: {num_generations}")
        print(f"Max output tokens: {max_output_tokens}")

        all_outputs = []
        all_extracted_answers = []

        # Create batches
        for i in trange(0, len(input_texts), batch_size, desc="Processing batches", unit="batch"):
            batch_system_prompts = system_prompts[i:i+batch_size]
            batch_input_texts = input_texts[i:i+batch_size]

            # Build messages for each example in the batch
            messages_batch = [
                [
                    {"role": "system", "content": sys_prompt} if sys_prompt else None,
                    {"role": "user", "content": input_text}
                ]
                for sys_prompt, input_text in zip(batch_system_prompts, batch_input_texts)
            ]
            # Remove None if there is no system_prompt
            messages_batch = [
                [msg for msg in msgs if msg is not None]
                for msgs in messages_batch
            ]

            with torch.no_grad():
                formatted_inputs = [
                    self.tokenizer.apply_chat_template(
                        msgs,
                        return_tensors="pt",
                        tokenize=False,
                        add_generation_prompt=True,
                        max_length=self.tokenizer.model_max_length,
                    )
                    for msgs in messages_batch
                ]
                # Tokenize all inputs in the batch
                inputs = self.tokenizer(
                    formatted_inputs,
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
                    top_p=None, # only set if do_sample=False
                    top_k=None, # only set if do_sample=False
                    temperature=None, # only set if do_sample=False
                    return_dict_in_generate=True,
                    **kwargs,
                )

            sequences = output.sequences
            # Remove the input part from the generated sequences
            generated_sequences = sequences[:, input_length:]
            decoded_sequences = self.tokenizer.batch_decode(
                generated_sequences, skip_special_tokens=True
            )
            # Extract the answers for each example in the batch
            for j in range(len(batch_input_texts)):
                # For num_generations > 1: slice the correct sequences
                start = j * num_generations
                end = (j + 1) * num_generations
                responses = [seq.strip() for seq in decoded_sequences[start:end]]
                extracted = [self.extract_answer(resp) for resp in responses]
                all_outputs.append(responses if num_generations > 1 else responses[0])
                all_extracted_answers.append(extracted if num_generations > 1 else extracted[0])

        return {
            "output": all_outputs,
            "extracted_answers": all_extracted_answers,
        }


class OpenAIModel(Model):
    def __init__(self, model_name_or_path: str, api_key: str):
        super().__init__(model_name_or_path)
        self.openai_client = OpenAI(api_key=api_key)

    def apply_chat_template(
        self,
        system_prompt: str,
        message: str,
    ) -> List[Dict[str, str]]:
        """
        Applies a chat template to the given message and system prompt.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return messages

    def prompt(
        self,
        system_prompt: str,
        message: str,
        # num_generations: int,
        # max_output_tokens: int,
        **kwargs,
    ) -> Dict:
        # messages = self.apply_chat_template(
        #     system_prompt, message
        # )
        # assistant_responses = []
        # for _ in range(num_generations):
        #     response = self.openai_client.responses.create(
        #         model=self.model_name_or_path,
        #         input=messages, # type: ignore
        #         max_output_tokens=max_output_tokens,
        #         **kwargs,
        #     )
        #     assistant_responses.append(response.output_text)
        assistant_responses = self.query_model(
            system_prompt,
            message,
            # num_generations=num_generations,
            # max_output_tokens=max_output_tokens,
            **kwargs,
        )

        extracted_answers = [
            self.extract_answer(response) for response in assistant_responses
        ]

        result_dict = {
            "output": (
                assistant_responses if kwargs.get("num_generations", 0) > 1 else assistant_responses[0]
            ),
            "extracted_answers": (
                extracted_answers if kwargs.get("num_generations", 0) > 1 else extracted_answers[0]
            ),
        }

        return result_dict
    
    def query_model(
        self,
        system_prompt: str,
        message: str,
        num_generations: int,
        # max_output_tokens: int,
        **kwargs: dict,
    ) -> list[str]:
        messages = self.apply_chat_template(
            system_prompt, message
        )

        assistant_responses = []
        for _ in range(num_generations):
            response = self.openai_client.responses.create(
                model=self.model_name_or_path,
                input=messages, # type: ignore
                max_output_tokens=kwargs.get("max_output_tokens", 512), # type: ignore
                # **kwargs,
            )
            assistant_responses.append(response.output_text)

        return assistant_responses
    
class GeminiModel(Model):
    def __init__(self, model_name_or_path: str, api_key: str):
        super().__init__(model_name_or_path)
        self.client = genai.Client(api_key=api_key)
    
    def prompt(
        self,
        system_prompt: str,
        message: str,
        num_generations: int = 1,
        max_output_tokens: int = 512,
        **kwargs,
    ) -> Dict:
        assistant_responses = self.query_model(
            message=message,
            num_generations=num_generations,
            system_prompt=system_prompt
        )

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
    
    def query_model(
            self,
            message: str,
            num_generations: int,
            system_prompt: str = "",
            **kwargs: dict,
        ) -> list[str]:

        assistant_responses = []
        for _ in range(num_generations):
            response = self.client.models.generate_content(
                model=self.model_name_or_path,
                contents=message,
                config=GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    max_output_tokens=kwargs.get("max_output_tokens", 512), # type: ignore
                ),
            )

            assistant_responses.append(
                response.text if hasattr(response, "text") else str(response)
            )

        return assistant_responses

class RandomAnswer(Model):
    def __init__(self):
        """
        This model is for debugging purposes only.
        It randomly chooses between "Answer1", "Answer2", and "Tie" as the output.
        This is useful to test the pipeline without needing the hardware to run a big LLM
        or using API credits.
        """
        super().__init__("random_choice_model")

    def prompt(
        self,
        system_prompt: str,
        input_text: str,
        num_generations: int,
        max_output_tokens: int = 512,
    ) -> Dict:
        assistant_responses = self.query_model(num_generations)
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

    def query_model(
        self,
        num_generations
    ) -> list[str]:
        from random import choice
        return [choice(["Answer1", "Answer2", "Tie"]) for _ in range(num_generations)]

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
    elif model_name_or_path == "RandomAnswer":
        return RandomAnswer()
    else:
        raise ValueError(f"Model {model_name_or_path} not supported.")
