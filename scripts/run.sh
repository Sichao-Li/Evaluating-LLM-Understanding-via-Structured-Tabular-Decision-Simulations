#!/usr/bin/env bash
set -euo pipefail

MODEL="google/medgemma-4b-it"
OUTPUT_DIR="results_test"

DATASETS=("heart")
TASKS=("standard" "lao")
METHODS=("constant" "mean" "sample_marginal" "permutation")

for dataset in "${DATASETS[@]}"; do
  for task in "${TASKS[@]}"; do
    if [[ "${task}" == "ablation" ]]; then
      for method in "${METHODS[@]}"; do
        echo "Running dataset=${dataset}, task=${task}, ablation_method=${method} ..."
        CUDA_VISIBLE_DEVICES=6 python scripts/run_prediction.py \
          --dataset "${dataset}" \
          --model "${MODEL}" \
          --task "${task}" \
          --output_dir "${OUTPUT_DIR}" \
          --ablation_method "${method}"
      done
    else
      echo "Running dataset=${dataset}, task=${task} ..."
      CUDA_VISIBLE_DEVICES=6 \
       python scripts/run_prediction.py \
        --dataset "${dataset}" \
        --model "${MODEL}" \
        --task "${task}" \
        --output_dir "${OUTPUT_DIR}"
    fi
  done
done

