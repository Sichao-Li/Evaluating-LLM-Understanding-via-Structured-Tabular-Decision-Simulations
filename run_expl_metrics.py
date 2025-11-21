#!/usr/bin/env python
"""
Toy example: evaluate the Iris/Qwen prediction run with accuracy metrics.py under Zero-shot Setting.
"""

import json
import os
import re
from typing import List
import numpy as np
from src.metrics import STaDSMetrics 
import ast


DATASET = "iris"
MODEL = "Qwen3_8B"

BASE_DIR = "results_analysis"          # adjust if needed
DATASET_DIR = os.path.join(BASE_DIR, DATASET)

# === 1. Prediction (baseline) ===
PRED_JSON = os.path.join(
    DATASET_DIR,
    f"{DATASET}_standard_prediction",
    f"{DATASET}_output_part2_Qwen_{MODEL}.json"
)

PRED_NPY = os.path.join(
    DATASET_DIR,
    f"{DATASET}_standard_prediction",
    f"{DATASET}_output_part2_Qwen_{MODEL}_post_process.npy"
)

# === 2. LAO directory ===
LAO_DIR = os.path.join(DATASET_DIR, "iris_lao")

# === 3. Self-Attribution ===
SELF_ATTR_JSON = os.path.join(
    DATASET_DIR,
    "iris_self_attribution",
    f"{DATASET}_self_attribution_Qwen_{MODEL}.json"
)

SELF_ATTR_POST_NPY = os.path.join(
    DATASET_DIR,
    "iris_self_attribution",
    f"{DATASET}_self_attribution_Qwen_{MODEL}_post_process.npy"
)

def main():
    # ------------------------------------------------------------------
    # 1. Load the Iris/Qwen result JSON
    # ------------------------------------------------------------------
    with open(PRED_JSON, "r") as f:
        data = json.load(f)

    masked_gt = data["masked_gt"]
    print(f"[INFO] Loaded {PRED_JSON}")
    print(f"[INFO] Number of gold labels (masked_gt): {len(masked_gt)}")

    # ------------------------------------------------------------------
    # 2. Extract prediction_list from processed files
    # ------------------------------------------------------------------
    prediction_list = np.load(PRED_NPY).tolist()
    print(f"[INFO] Loaded baseline predictions from NPZ (length={len(prediction_list)}).")

    # ------------------------------------------------------------------
    # 3. Compute prediction metrics
    # ------------------------------------------------------------------

    metrics = STaDSMetrics.compute_all(prediction_list, masked_gt)
    print("\n=== Prediction Metrics (Iris / Qwen3-8B) ===")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"F1 (macro):           {metrics['f1_macro']:.4f}")
    print(f"Penalised accuracy:   {metrics['penalised_accuracy']:.4f}")
    print(f"Length F1:   {metrics['length_f1']:.4f}")
    print(f"Unknown label rate:   {metrics['unknown_label_rate']:.4f}")
    print("\n[INFO] Done.")

    # Column names reference
    col_names_dir = {3: 'sepal_length', 2: 'sepal_width', 1: 'petal_length', 0: 'petal_width'}
    print("\n[INFO] Loading LAO files…")

    # detect which features were used (number of LAO JSONs)
    lao_jsons = sorted(
        f for f in os.listdir(LAO_DIR)
        if f.endswith(f"_{MODEL}.json") or f.endswith(".json")
    )
    lao_jsons = [os.path.join(LAO_DIR, f) for f in lao_jsons]

    lao_scores = []
    feature_ids = []

    lao_scores: List[float] = []

    LAO_PATTERN = re.compile(r"_col(\d+)_.*_post_process\.npy$")
    for fname in os.listdir(LAO_DIR):
        m = LAO_PATTERN.search(fname)
        if not m:
            continue
        col_idx = int(m.group(1))
        path_npy = os.path.join(LAO_DIR, fname)
        preds_col = np.load(path_npy).tolist()
        score = STaDSMetrics.penalised_accuracy(preds_col, data["masked_gt"])
        lao_scores.append(metrics['penalised_accuracy']-score)
        feature_ids.append(col_idx)

    lao_scores = dict(zip(feature_ids, lao_scores))
    print("\n===== LAO Behavioral Attribution =====")
    for fid, drop in lao_scores.items():
        print(f"Column {fid}: drop = {drop:.4f}")

    # ==============================================================
    # 4. Self-attribution metrics
    # ==============================================================
    print("\n[INFO] Loading self-claimed attribution…")

    # Load post-processed self-attr (list of rankings)
    self_rank_raw = np.load(SELF_ATTR_POST_NPY, allow_pickle=True).tolist()
    self_rank = ast.literal_eval(self_rank_raw)
    print(f"\n[INFO] self-claimed attributions: {self_rank}")
    rho, p = STaDSMetrics.self_faith(self_rank, lao_scores, col_names_dir)
    print(f"\n[INFO] Decision faithfulness is: {rho}")    
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()