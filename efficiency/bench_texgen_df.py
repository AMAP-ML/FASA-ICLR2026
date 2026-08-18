# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import os
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM
import transformers 
from monkey_patch.frequency.dynamic_frequency import replace_llama_df
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaFlashAttention2,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaFlashAttention2}
transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES = LLAMA_ATTENTION_CLASSES
@dataclasses.dataclass
class ModelConfig:
  model_path: str
  dtype: str = dataclasses.field(default="float16")
  device: str = dataclasses.field(default="cuda:0")

MODEL_DIR = os.environ.get("FASA_MODEL_DIR", "./models")

MODEL_CFGS = {
    "Meta-Llama-3.1-8B-Instruct":
        ModelConfig(
            model_path=os.path.join(MODEL_DIR, "Meta-Llama-3.1-8B-Instruct")
        ),
}

def load_model(model_cfg: ModelConfig):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)

    with device:
        model = LlamaForCausalLM.from_pretrained(
            model_cfg.model_path,
            device_map=device,
            torch_dtype=dtype,
        )
    return model

@torch.inference_mode()
def benchmark_quest():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--context_len", type=int, default=2*1024)
    parser.add_argument("--decode_len", type=int, default=256)
    parser.add_argument("--min_k", type=int, default=16)
    parser.add_argument("--max_k", type=int, default=16)    
    parser.add_argument("--threshold_ratio", type=float, default=0.8)
    parser.add_argument("--layer_control", type=int, default=-1)

    parser.add_argument("--token_budget", type=int, default=256)
    parser.add_argument("--iteration", type=int, default=10)
    args = parser.parse_args()

    assert args.model in MODEL_CFGS, f"Model {args.model} not found in MODEL_CFGS"
    model_cfg = MODEL_CFGS[args.model]
    
    max_seq_len = args.context_len + args.decode_len + 512
    token_budget = args.token_budget
    context_len = args.context_len
    decode_len = args.decode_len

    topk = args.max_k
    threshold_ratio = args.threshold_ratio
    budget = args.token_budget

    model = load_model(model_cfg)
    
    dtype = getattr(torch, model_cfg.dtype)
    device = torch.device(model_cfg.device)

    hidden_size = model.config.hidden_size

    prefill_latency = []
    decode_latency = []

    
    for _ in tqdm(range(args.iteration)):
        past_key_values = None
        # clear cuda cache
        torch.cuda.empty_cache()

        # Prefill Stage
        ts = time.perf_counter()
        hidden_states = torch.randn(1, context_len, hidden_size, dtype=dtype, device=device)
        with torch.no_grad():
            outputs = model(
                inputs_embeds=hidden_states,
                past_key_values=past_key_values,
                use_cache=True
            )
        te = time.perf_counter()
        prefill_latency.append(te - ts)
        # Start decoding decode_len tokens
        # print("=========================================")
        for _ in range(decode_len):
            past_key_values = outputs.past_key_values
            # print(f"{past_key_values[0][0].shape}")
            hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
            ts = time.perf_counter()
            with torch.no_grad():
                outputs = model(
                    inputs_embeds=hidden_states,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            te = time.perf_counter()
            decode_latency.append(te - ts)
        # print("=========================================")

    avg_prefill_latency = np.mean(prefill_latency)
    avg_decode_latency = np.mean(decode_latency)
    print(f"===========context lens: {context_len}, decode lens: {decode_len}==============================")
    print("{:<12} {:<12} {:<11} {:<20} {:<20}".format(
    "token_budget", "context_len", "decode_len", "avg_prefill_latency", "avg_decode_latency"))
    print("{:<12} {:<12} {:<11} {:<20.7f} {:<20.7f}".format(
        token_budget, context_len, decode_len, avg_prefill_latency, avg_decode_latency))
    print("==============================================================================================")

if __name__ == "__main__":
    benchmark_quest()

# nsys profile --delay 20 --duration 1 --output "$(env TZ='US/Pacific' date +%Y%m%d-%H%M%S).nsys-rep" python text_gen.py