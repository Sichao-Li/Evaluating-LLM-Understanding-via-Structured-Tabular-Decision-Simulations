import argparse
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from src.models import HuggingFaceLLM, OpenAIGPT, GoogleGemini
from src.data_loader import (
    load_and_clean_data, 
    stratified_sample, 
    prepare_prediction_context, 
    drop_attributes
    )
from src.prompt import PromptEngine
from src.parser import OutputParser

# Load environment variables
load_dotenv()

def get_model(model_id: str, quantization: str = None):
    """Factory to get the right model instance."""
    if "gpt" in model_id.lower():
        return OpenAIGPT(model_id)
    elif "gemini" in model_id.lower():
        return GoogleGemini(model_id)
    else:
        return HuggingFaceLLM(model_id, quantization=quantization)

def run_single_experiment(
    model, 
    prompt_engine, 
    dataset_name, 
    df_train, 
    df_test, 
    drop_cols=None, 
    few_shot=False
):
    """
    Runs one prediction pass.
    Used for both standard prediction and one step of the LAO loop.
    """
    # 1. Apply Attribute Dropping (if any)
    current_train = drop_attributes(df_train.copy(), drop_cols or [])
    current_test = drop_attributes(df_test.copy(), drop_cols or [])
    
    # Combine for context generation (Train + Test)
    current_train["dataset"] = "train"
    current_test["dataset"] = "test"
    df_combined = pd.concat([current_train, current_test], ignore_index=True)

    # 2. Prepare Table String & Ground Truth
    table_str, ground_truths, masked_indices = prepare_prediction_context(
        df_combined, 
        mask_train=False, # Typically we see train labels in context 
        mask_test=True,   # And predict test labels
        move_unknowns_to_end=True
    )

    # 3. Construct the specific Question
    n_predict = len(masked_indices)
    question = (
        f"Predict the integer-encoded class for each row where 'class=?'. "
        f"There are exactly {n_predict} rows with 'class=?' in the input table. "
        f"Do not output more or fewer predictions than {n_predict}."
    )
    
    # 3. Render Prompt
    instruction = prompt_engine.render_prediction_instruction(
        dataset_name, 
        few_shot=few_shot,
        drop_attributes=drop_cols
    )
    
    full_prompt = (
        f"Below is an instruction that describes a task, paired with an input table that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{table_str}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Response:"
    )
    
    # 4. Generate
    start_time = time.time()
    raw_output = model.generate(full_prompt, temperature=0.1)
    duration = time.time() - start_time

    return {
        "drop_cols": drop_cols,
        "instruction": instruction,
        "input_table_snippet": table_str[:500] + "...", # Log partial table to save space
        "raw_output": raw_output,
        "ground_truths": ground_truths,
        "masked_indices": masked_indices,
        "duration_seconds": duration
    }

def main():
    parser = argparse.ArgumentParser(description="Run tabular LLM predictions.")
    
    # Experiment Settings
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID (e.g., iris)")
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g., meta-llama/Llama-3.1-8B)")
    parser.add_argument("--task", type=str, choices=["standard", "lao"], default="standard", help="Task type")
    parser.add_argument("--few_shot", action="store_true", help="Providing demonstrations")
    parser.add_argument("--sample_rows", type=int, default=100, help="Max rows to process (stratified sample)")
    
    # Model Settings
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit"], default=None, help="HF Quantization")
    parser.add_argument("--parse", action="store_true", help="Immediately parse output with LLM")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()

    # --- 1. Setup Resources ---
    print(f"Initializing {args.model}...")
    model = get_model(args.model, quantization=args.quantization)
    prompt_engine = PromptEngine()
    
    if args.parse:
        try:
            parser_model = OpenAIGPT("gpt-4o-mini")
        except:
            print("OpenAI not available for parsing, using main model (slower).")
            parser_model = model
        output_parser = OutputParser(parser_model)

    # --- 2. Load Data ---
    data_dir = Path("data") / args.dataset
    train_df = load_and_clean_data(data_dir / "train.csv")
    test_df = load_and_clean_data(data_dir / "test.csv")
    
    # Sampling if too large
    full_df = pd.concat([train_df, test_df])
    if len(full_df) > args.sample_rows:
        print(f"Sampling {args.sample_rows} rows...")
        test_df = stratified_sample(test_df, "class", args.sample_rows)

    # --- 3. Define Experiments ---
    experiments = []
    
    if args.task == "standard":
        experiments.append({"drop": None, "name": "standard"})
        
    elif args.task == "lao":
        # Leave-Attribute-Out: Iterate over all columns except 'class'
        feature_cols = [c for c in train_df.columns if c != "class"]
        # Add baseline (no drop)
        experiments.append({"drop": None, "name": "baseline"})
        # Add drop experiments
        for col in feature_cols:
            experiments.append({"drop": [col], "name": f"drop_{col}"})

    # --- 4. Execution Loop ---
    results_dir = Path(args.output_dir) / args.dataset / args.model.replace("/", "_")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting {len(experiments)} experiments for task: {args.task}")
    
    combined_results = []

    for exp in tqdm(experiments):
        print(f"\nRunning: {exp['name']}")
        
        result_data = run_single_experiment(
            model=None,
            prompt_engine=prompt_engine,
            dataset_name=args.dataset,
            df_train=train_df,
            df_test=test_df,
            drop_cols=exp["drop"],
            few_shot=args.few_shot
        )
        
        # Optional: Parse immediately
        if args.parse:
            preds = output_parser.extract_predictions(
                result_data["raw_output"], 
                expected_count=len(result_data["ground_truths"])
            )
            result_data["parsed_predictions"] = preds
            
            # Simple Accuracy Check
            if len(preds) == len(result_data["ground_truths"]):
                correct = sum(1 for p, g in zip(preds, result_data["ground_truths"]) if p == g)
                acc = correct / len(preds)
                print(f"  -> Accuracy: {acc:.2%}")
                result_data["accuracy"] = acc

        # Save individual file (optional, good for backup) or accumulate
        combined_results.append({**exp, **result_data})

    # --- 5. Save Final Consolidated JSON ---
    output_file = results_dir / f"{args.task}_results.json"
    with open(output_file, "w") as f:
        json.dump(combined_results, f, indent=2, default=str)
    
    print(f"Saved all results to {output_file}")

    # --- 6. Cleanup ---
    model.cleanup()

if __name__ == "__main__":
    main()