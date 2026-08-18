#!/bin/bash
# Perplexity (language-modeling) evaluation on a single GPU.
#
# Set FASA_MODEL_DIR to the folder that holds your local Hugging Face
# checkpoints before running.
export CUDA_VISIBLE_DEVICES=0

MODEL="Llama-3.2-3B-Instruct"
DATASET="c4"          # c4 / pg19 / wikitext

for budget in 256 512 1024; do
    python -m perplexity.evaluate_ppl \
        --model_name "$MODEL" \
        --dataset_name "$DATASET" \
        --compression_methods "dynamic_frequency" \
        --min_k 16 \
        --max_k 16 \
        --threshold_ratio 0.8 \
        --budget "$budget" \
        --prefilling_len 8000 \
        --decoding_len 256 \
        --chunk_size 16 \
        --cal_dataset "${DATASET}_8192_256" \
        --limit 60
done
