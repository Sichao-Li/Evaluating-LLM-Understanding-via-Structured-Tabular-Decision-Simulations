import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

from src.models import HuggingFaceLLM, OpenAIGPT, GoogleGemini
from src.data_loader import load_and_clean_data, stratified_sample, prepare_prediction_context
from src.prompt import PromptEngine
from src.parser import OutputParser

load_dotenv()

def get_model(model_id: str, quantization: str = None):
    if "gpt" in model_id.lower():
        return OpenAIGPT(model_id)
    elif "gemini" in model_id.lower():
        return GoogleGemini(model_id)
    else:
        return HuggingFaceLLM(model_id, quantization=quantization)

def main():
    parser = argparse.ArgumentParser(description="Run tabular feature attribution (ranking).")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sample_rows", type=int, default=100)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/attribution")
    args = parser.parse_args()

    # 1. Setup
    print(f"Initializing {args.model}...")
    model = get_model(args.model, args.quantization)
    prompt_engine = PromptEngine()
    
    # Use GPT-4o-mini for parsing if available, else fallback
    try:
        parser_model = OpenAIGPT("gpt-4o-mini")
    except:
        parser_model = model
    output_parser = OutputParser(parser_model)

    # 2. Data
    data_dir = Path("data") / args.dataset
    train_df = load_and_clean_data(data_dir / "train.csv")
    test_df = load_and_clean_data(data_dir / "test.csv")
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    if len(full_df) > args.sample_rows:
        full_df = stratified_sample(full_df, "class", args.sample_rows)

    # For attribution, we provide the FULL table.
    table_str, _, _ = prepare_prediction_context(
        full_df, 
        mask_train=False, 
        mask_test=False, 
        move_unknowns_to_end=False
    )

    # 3. Prompts
    prompts_list = [
        (
            "Based on your global understanding of the dataset and the prediction problem, "
            "please rank all input features (attributes) in descending order of importance for predicting the target variable (class). "
            "Return your answer as EXACTLY one line in the following format:\n"
            "feature1, feature2, feature3, ...\n"
            "Rules: Do NOT include the feature 'class'. Do NOT include numbering, bullets, or explanations. "
            "Use ONLY the valid dataset feature names. Output nothing else."
        ),
    ]

    results = []
    
    instruction = prompt_engine.render_attribution_instruction(args.dataset)

    # 4. Execution Loop
    for i, question in enumerate(prompts_list, 1):
        print(f"Running Prompt Style {i}...")
        
        full_prompt = (
            f"Below is an instruction that describes a task, paired with an input table that provides further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{table_str}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Response:"
        )
        
        raw_output = model.generate(full_prompt, max_new_tokens=1024)
        
        # Extract ranking list
        ranked_features = output_parser.extract_attributions(raw_output)
        
        results.append({
            "style_id": i,
            "question": question,
            "raw_output": raw_output,
            "ranked_features": ranked_features
        })

    # 5. Save
    out_path = Path(args.output_dir) / args.dataset
    out_path.mkdir(parents=True, exist_ok=True)
    
    file_name = f"{args.model.replace('/', '_')}_attribution.json"
    with open(out_path / file_name, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model,
            "input_table_snippet": table_str[:500],
            "runs": results
        }, f, indent=2)
        
    print(f"Saved attribution results to {out_path / file_name}")
    model.cleanup()

if __name__ == "__main__":
    main()