import os
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen3-8B",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
]


def download_model(model_id):
    print(f"Downloading model and tokenizer for {model_id}...")
    try:
        AutoTokenizer.from_pretrained(model_id, token=True)
        AutoModelForCausalLM.from_pretrained(model_id, token=True)
        print(f"Successfully downloaded {model_id}.")
    except Exception as e:
        print(f"Failed to download {model_id}: {e}")


if __name__ == "__main__":
    for model_id in MODELS:
        download_model(model_id)
    print("All models downloaded.")