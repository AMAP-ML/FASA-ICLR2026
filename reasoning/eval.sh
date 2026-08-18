#!/bin/bash
# Grade the generations produced by reasoning/run_math.py.
# --base_dir should point at the folder of prediction JSONL files written by
# run_math.py (default output layout: reasoning/<dataset>_<cal_dataset>/).

python reasoning/evaluation/eval_math.py \
    --exp_name "evaluation" \
    --output_dir "." \
    --base_dir "reasoning/math_qasper" \
    --dataset math
