#!/bin/bash
# Math CoT reasoning evaluation on a single GPU.
#
# Set FASA_MODEL_DIR to the folder that holds your local Hugging Face
# checkpoints before running.
export CUDA_VISIBLE_DEVICES=0

python -m reasoning.run_math \
    --seed 42 \
    --dataset_name "math" \
    --model_name "DeepSeek-R1-Distill-Qwen-14B" \
    --method "dynamic_frequency" \
    --budget 300 \
    --threshold_ratio 0.8 \
    --min_k 16 \
    --max_k 16 \
    --keep_high_dim 32 \
    --cal_dataset "qasper"
