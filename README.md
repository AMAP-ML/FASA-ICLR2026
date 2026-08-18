# FASA: Frequency-aware Sparse Attention

Official code for the paper **"FASA: Frequency-aware Sparse Attention"**.

> The deployment of Large Language Models (LLMs) faces a critical bottleneck when
> handling lengthy inputs: the prohibitive memory footprint of the Key-Value (KV)
> cache. FASA is a query-aware token-eviction framework that dynamically predicts
> token importance. It stems from a novel insight into RoPE: *functional sparsity
> at the frequency-chunk (FC) level*. A small, identifiable subset of "dominant"
> FCs consistently exhibits high contextual agreement with the full attention head,
> providing a robust and computationally free proxy for identifying salient tokens.
> FASA first identifies a critical set of tokens using dominant FCs, then performs
> focused attention computation solely on this pruned subset. Across long-context
> tasks — from sequence modeling to CoT reasoning — FASA consistently outperforms
> token-eviction baselines and approaches near-oracle accuracy. On LongBench-V1 it
> reaches nearly 100% of full-KV performance while keeping only 256 tokens, and
> achieves a 2.56x speedup using just 18.9% of the cache on AIME24.

Paper: https://huggingface.co/papers/2602.03152

---

## Method overview

FASA works in two stages:

1. **Offline FC-agreement profiling.** For a given model and calibration dataset,
   we measure how well each frequency chunk (a 2-dimensional RoPE frequency pair)
   agrees with the full attention head in ranking token importance. This produces
   a per-layer / per-head ranking of "dominant" FCs, saved as
   `dimension_rankings/<model_name>/<dataset>_agreement.pkl`.
2. **Online query-aware eviction.** At inference time, FASA uses only the dominant
   FCs to cheaply estimate token importance, selects the top-`budget` tokens, and
   runs focused attention over that pruned KV subset. This is implemented as a
   set of FlashAttention-2 monkey patches (`dynamic_frequency`).

The repository also includes reimplementations of the token-eviction baselines
used in the paper: **H2O**, **Quest**, **SnapKV**, **StreamingLLM**, and an
**Oracle** (full-attention top-k) upper bound.

## Repository layout

```
.
├── identify_important_fre.py            # Stage 1: FC-agreement profiler (main entry)
├── identify_important_fre_distribution.py  # FC distribution analysis / dominant-FC extraction
├── identify_important_fre_alibi.py      # Variant for ALiBi-position models
├── monkey_patch/                        # Attention monkey patches
│   ├── frequency/                       # FASA (dynamic_frequency) — the proposed method
│   ├── oracle/                          # Oracle full-attention top-k baseline
│   ├── h2o/                             # H2O baseline
│   ├── quest/                           # Quest baseline
│   ├── snapkv/                          # SnapKV baseline
│   └── streamingllm/                    # StreamingLLM baseline
├── longbench/                           # LongBench-V1 evaluation harness
├── perplexity/                          # Language-modeling perplexity evaluation
├── reasoning/                           # Math CoT reasoning evaluation (GSM8K / MATH / AIME24)
├── efficiency/                          # Latency / memory micro-benchmarks
├── config/compress_params.json          # Baseline hyper-parameters per budget
└── bash.sh                              # End-to-end single-GPU example
```

## Installation

```bash
# Python 3.10 is recommended (matches the paper's environment).
pip install -r requirements.txt

# flash-attn must match your CUDA / torch / python build. For CUDA 12.4 + torch 2.5:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Model weights

The evaluation scripts load models from a local directory. Set the
`FASA_MODEL_DIR` environment variable to the folder containing your Hugging Face
model checkpoints (each in a subfolder named exactly as `--model_name`):

```bash
export FASA_MODEL_DIR=/path/to/your/hf_models
# e.g. $FASA_MODEL_DIR/Meta-Llama-3.1-8B-Instruct/
```

If unset, it defaults to `./models`.
Supported model families: **LLaMA / Llama-3.x**, **Qwen2.5**, **Mistral**
(and DeepSeek-R1-Distill variants of Llama/Qwen for reasoning).

The LongBench and perplexity datasets (LongBench-V1, c4, pg19, wikitext) are
downloaded automatically via the `datasets` library on first use, or you can
place them locally. Small reasoning inputs (GSM8K / MATH / AIME24) are included
under `reasoning/data/`.

## Usage

### Step 1 — Profile dominant frequency chunks (offline)

This produces the `dimension_rankings/<model_name>/<dataset>_agreement.pkl` files
that the FASA (`dynamic_frequency`) method depends on.

```bash
python -m identify_important_fre \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --task_type "longbench" \
    --dataset_name "qasper"
```

### Step 2a — LongBench evaluation

```bash
python -m longbench.longbench_pred \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --compression_methods "dynamic_frequency" \   # or: base / oracle / snapkv / streamingllm / quest
    --budget 256 \
    --threshold_ratio 0.8 \
    --min_k 16 --max_k 16 \
    --keep_high_dim 32 \
    --dataset_name "narrativeqa" \
    --cal_dataset "qasper" \
    --output_dir "results"
```

### Step 2b — Perplexity (language modeling)

```bash
python -m perplexity.evaluate_ppl \
    --model_name "Llama-3.2-3B-Instruct" \
    --compression_methods "dynamic_frequency" \
    --dataset_name "c4" \
    --budget 256 \
    --prefilling_len 8000 --decoding_len 256 \
    --min_k 16 --max_k 16 --threshold_ratio 0.8
```

### Step 2c — Math CoT reasoning

```bash
python -m reasoning.run_math \
    --model_name "DeepSeek-R1-Distill-Qwen-32B" \
    --dataset_name "aime24" \
    --method "dynamic_frequency" \
    --budget 1500 \
    --min_k 16 --max_k 16 --threshold_ratio 0.8 \
    --keep_high_dim 32 \
    --cal_dataset "qasper" \
    --seed 1
```

### Efficiency benchmark

```bash
bash efficiency/bench_texgen.sh
```

See `bash.sh`, `perplexity/evaluate_ppl.sh`, `reasoning/run_math.sh`, and
`efficiency/bench_texgen.sh` for ready-to-run single-GPU examples.

## Key arguments

| Argument | Meaning |
| --- | --- |
| `--compression_methods` / `--method` | `dynamic_frequency` (FASA), `base`, `oracle`, `snapkv`, `streamingllm`, `quest` |
| `--budget` | KV cache token budget (e.g. 128 / 256 / 512 / 1024 …) |
| `--min_k` / `--max_k` | Min / max number of dominant frequency chunks retained per head |
| `--threshold_ratio` | Quantile threshold for adaptive FC selection |
| `--keep_high_dim` | Number of high-importance dimensions kept |
| `--cal_dataset` | Calibration dataset used to build the FC-agreement profile |
| `--layer_control` | Restrict FASA to a specific layer (`-1` = all layers) |

## Acknowledgements

The baseline implementations adapt code from the original H2O, Quest, SnapKV, and
StreamingLLM releases, and the reasoning evaluation harness builds on the
`latex2sympy2`-based math grader. We thank the authors of those projects.

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026fasa,
  title   = {FASA: Frequency-aware Sparse Attention},
  author  = {Wang, Yifei and Wang, Yueqi and Yue, Zhenrui and Zeng, Huimin and
             Wang, Yong and Lourentzou, Ismini and Tu, Zhengzhong and
             Chu, Xiangxiang and McAuley, Julian},
  journal = {arXiv preprint arXiv:2602.03152},
  year    = {2026}
}
```

## License

Released under the [Apache License 2.0](LICENSE).
