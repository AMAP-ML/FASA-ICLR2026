#!/bin/bash
# Example FASA workflow on a single GPU.
#
# Set FASA_MODEL_DIR to the folder that holds your local Hugging Face
# checkpoints (each in a subfolder named exactly as --model_name), e.g.:
#   export FASA_MODEL_DIR=/path/to/hf_models
export CUDA_VISIBLE_DEVICES=0

MODEL="Meta-Llama-3.1-8B-Instruct"

# -----------------------------------------------------------------------------
# Step 1: Profile dominant frequency chunks (offline).
# Produces dimension_rankings/<model>/<dataset>_agreement.pkl, which the
# FASA (dynamic_frequency) method relies on.
# -----------------------------------------------------------------------------
python -m identify_important_fre \
    --model_name "$MODEL" \
    --task_type "longbench" \
    --dataset_name "qasper" \
    --evaluate_num 8

# -----------------------------------------------------------------------------
# Step 2: Run LongBench evaluation with FASA.
# --compression_methods can be: dynamic_frequency (FASA) / base / oracle /
#                               snapkv / streamingllm / quest
# -----------------------------------------------------------------------------
python -m longbench.longbench_pred \
    --model_name "$MODEL" \
    --compression_methods "dynamic_frequency" \
    --keep_high_dim 32 \
    --output_dir "results" \
    --threshold_ratio 0.8 \
    --min_k 16 \
    --max_k 16 \
    --budget 256 \
    --dataset_name "narrativeqa" \
    --cal_dataset "qasper" \
    --layer_control -1
