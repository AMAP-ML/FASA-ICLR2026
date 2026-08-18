#!/bin/bash
# Latency / memory micro-benchmark for FASA decoding on a single GPU.
#
# Set FASA_MODEL_DIR to the folder that holds your local Hugging Face
# checkpoints before running.
export CUDA_VISIBLE_DEVICES=0

for ctx in 1000 2000 4000 8000 16000 32000; do
    python -m efficiency.bench_texgen \
        --model "Meta-Llama-3.1-8B-Instruct" \
        --context_len "$ctx" \
        --decode_len 512 \
        --min_k 16 \
        --max_k 16 \
        --threshold_ratio 0.8 \
        --token_budget 256 \
        --layer_control -1 \
        --apply_fasa 1 \
        --batch_size 1 \
        --iteration 3
done
