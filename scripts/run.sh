#!/usr/bin/env bash
set -euo pipefail

DATASET="breast"
MODEL="Qwen/Qwen3-8B"
TASK="ablation"
OUTPUT_DIR="results_test"

METHODS=("constant" "mean" "sample_marginal" "permutation")

for method in "${METHODS[@]}"; do
  echo "Running ablation_method=${method} ..."
  CUDA_VISIBLE_DEVICES=0 python scripts/run_prediction.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --task "${TASK}" \
    --output_dir "${OUTPUT_DIR}" \
    --ablation_method "${method}"
done

echo "All runs completed."
