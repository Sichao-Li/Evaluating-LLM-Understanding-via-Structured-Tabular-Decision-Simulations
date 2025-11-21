import os
import gc
import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# Load environment variables from .env file
load_dotenv()

class LLM(ABC):
    """Abstract Base Class for all Language Models."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The user instruction/input.
            system_prompt: Optional system instruction (supported by some models).
            **kwargs: Generation arguments (temperature, max_tokens, etc.)
        """
        pass

    def cleanup(self):
        """Optional cleanup for freeing resources (GPU memory)."""
        pass


class HuggingFaceLLM(LLM):
    def __init__(self, model_id: str, device: str = "auto", quantization: str = None):
        """
        Args:
            model_id: HuggingFace hub ID (e.g., "meta-llama/Llama-3.1-8B").
            device: "cuda", "cpu", "mps", or "auto".
            quantization: defeaul is None, "4bit", or "8bit" for MPS.
        """
        super().__init__(model_id)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=os.getenv("HF_TOKEN")
        )
        
        # Handle Quantization Config
        bnb_config = None
        dtype = torch.float16
        if quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        print(f"Loading {model_id} on {device} (Quantization: {quantization})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=dtype,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # Defaults
        max_new_tokens = kwargs.get("max_new_tokens", 8192)
        temperature = kwargs.get("temperature", 0.1)
        do_sample = kwargs.get("do_sample", False)

        # 1. Apply Chat Template (if available)
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
        else:
            # Fallback for base models without chat templates
            text_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # 2. Tokenize and Move to Device
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

        # 3. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 4. Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return generated_text.strip()

    def cleanup(self):
        """Aggressive memory cleanup for research loops."""
        print(f"Unloading {self.model_id}...")
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


class OpenAIGPT(LLM):
    def __init__(self, model_id: str = "gpt-4o-mini"):
        super().__init__(model_id)
        if not OpenAI:
            raise ImportError("OpenAI library not installed. pip install openai")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_new_tokens", 4096),
        )
        return response.choices[0].message.content


class GoogleGemini(LLM):
    def __init__(self, model_id: str = "gemini-2.5-pro"):
        super().__init__(model_id)
        if not genai:
            raise ImportError("Google GenAI library not installed. pip install google-genai")
        
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:        
        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", 0.2),
            max_output_tokens=kwargs.get("max_new_tokens", 4096),
            thinking_config=types.ThinkingConfig(thinking_budget=512)
        )

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System Instruction: {system_prompt}\n\nUser: {prompt}"

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=full_prompt,
            config=config
        )
        return response.text