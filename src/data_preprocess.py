#!/usr/bin/env python3
"""
generate_dataset_yaml.py

Scan a directory of raw dataset files and emit one YAML metadata file per dataset.
Each YAML file is named <dataset_id>.yaml and contains an empty skeleton
which you can manually fill in.
"""

import os
import argparse
import yaml

YAML_SKELETON = {
    "dataset_id": "",           # will be set to the filename (no ext)
    "role": "",                 # act as specific professional, e.g. "data_scientist", "analyst"
    "name": "",                 # human-readable title
    "task_type": "",            # e.g. classification, regression
    "target_description": "",   # short description of what the target is
    "target_mapping": {},       # e.g. {0: "setosa", 1: "versicolor", 2: "virginica"}
    "input_type": "",           # one_hot | numeric | text | image | hybrid
    "attribute_glossary": {},   # for one-hot: {col_name: description}
    "feature_summary": {},      # for numeric: {feat: {min:…, max:…, mean:…, sd:…, corr:…}}
    "class_priors": {},         # e.g. {"unacc": 0.7, "acc": 0.22, …}
    "missing_values": "",       # e.g. "none" or "col1, col2"
}

def generate_yaml_skeletons(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        print(f"Processing: {fname}")
        if fname.startswith("."):
            continue
        base, ext = os.path.splitext(fname)
        # you could filter by ext if desired, e.g. .csv/.tsv/.json
        meta = YAML_SKELETON.copy()
        meta["dataset_id"] = base
        out_path = os.path.join(output_dir, f"{base}.yaml")
        if os.path.exists(out_path):
            print(f"Skipped (already exists): {out_path}")
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False)
        print(f"Created: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate empty YAML metadata files for datasets"
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        default="data",
        help="Directory containing raw dataset files"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="data/metadata",
        help="Directory to write YAML metadata files"
    )
    args = parser.parse_args()
    generate_yaml_skeletons(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
