# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import os
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM,AutoModelForCausalLM
import transformers 
from monkey_patch.frequency.dynamic_frequency import replace_qwen_df
from monkey_patch.frequency.utils import constructe_adaptive_selection
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention
from monkey_patch.frequency.llama_df import LlamaFlashAttention2_forward_dynamic_frequency  
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaFlashAttention2,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaFlashAttention2}
from monkey_patch.frequency.cache_design import DynamicCache as DynamicCache_dynamic

def replace_llama_df(budget,records,layer_control):
    # global records
    # records = constructe_adaptive_selection(model_name,budget,threshold_ratio,min_k,max_k)

    def new_forward(self, *args, **kwargs):
        return LlamaFlashAttention2_forward_dynamic_frequency(
            self, *args, budget=budget, records=records,layer_control=layer_control, **kwargs
        )
    transformers.models.llama.modeling_llama.DynamicCache = DynamicCache_dynamic
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = new_forward
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
    "Qwen2.5-7B-Instruct":
        ModelConfig(
            model_path=os.path.join(MODEL_DIR, "Qwen2.5-7B-Instruct")
        ),
        "Mistral-7B-Instruct-v0.3":
        ModelConfig(
            model_path=os.path.join(MODEL_DIR, "Mistral-7B-Instruct-v0.3")
        ),
}

def load_model(model_cfg: ModelConfig):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)

    with device:
        model = AutoModelForCausalLM.from_pretrained(
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
    parser.add_argument("--apply_fasa",type=int,default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    assert args.model in MODEL_CFGS, f"Model {args.model} not found in MODEL_CFGS"
    model_cfg = MODEL_CFGS[args.model]
    records = constructe_adaptive_selection(args.model,args.token_budget,args.threshold_ratio,args.min_k,args.max_k,dataset_name="qasper")
    if args.apply_fasa:
        print("=====================apply fasa====================")
        replace_qwen_df(args.token_budget, records)
        replace_llama_df(args.token_budget, records, args.layer_control)


    # max_seq_len = args.context_len + args.decode_len + 512
    token_budget = args.token_budget
    context_len = args.context_len
    decode_len = args.decode_len

    topk = args.max_k
    threshold_ratio = args.threshold_ratio


    model = load_model(model_cfg)
    
    dtype = getattr(torch, model_cfg.dtype)
    device = torch.device(model_cfg.device)

    hidden_size = model.config.hidden_size

    prefill_latency = []
    decode_latency = []
    decode_latency_per_decode = []
    take_up_ratios = []

    
    for _ in tqdm(range(args.iteration)):
        local_prefill_latency = 0
        local_decode_latency = []
        past_key_values = None
        # clear cuda cache
        torch.cuda.empty_cache()

        # Prefill Stage
        ts = time.perf_counter()
        hidden_states = torch.randn(args.batch_size, context_len, hidden_size, dtype=dtype, device=device)
        with torch.no_grad():
            outputs = model(
                inputs_embeds=hidden_states,
                past_key_values=past_key_values,
                use_cache=True
            )
        te = time.perf_counter()
        prefill_latency.append(te - ts)
        local_prefill_latency+=te - ts

        # Start decoding decode_len tokens
        # print("=========================================")
        for _ in range(decode_len):
            past_key_values = outputs.past_key_values
            # print(f"{past_key_values[0][0].shape}")
            hidden_states = torch.randn(args.batch_size, 1, hidden_size, dtype=dtype, device=device)
            ts = time.perf_counter()
            with torch.no_grad():
                outputs = model(
                    inputs_embeds=hidden_states,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            te = time.perf_counter()
            decode_latency.append(te - ts)
            local_decode_latency.append(te - ts)
        decode_latency_per_decode.append(sum(local_decode_latency))
        # print("=========================================")

        take_up_ratios.append(local_prefill_latency/(sum(local_decode_latency)+local_prefill_latency))

    avg_prefill_latency = np.mean(prefill_latency)
    avg_decode_latency = np.mean(decode_latency)
    print(f"===========context lens: {context_len}, decode lens: {decode_len}==============================")
    print("{:<12} {:<12} {:<11} {:<20} {:<20}".format(
    "token_budget", "context_len", "decode_len", "avg_prefill_latency", "avg_decode_latency"))
    print("{:<12} {:<12} {:<11} {:<20.7f} {:<20.7f}".format(
        token_budget, context_len, decode_len, avg_prefill_latency, avg_decode_latency))
    print("{:<20} {:<20} {:<20}".format(
         "Prefilling latency","decoding latency", "take up ratio"))
    print("{:<20.7f} {:<20.7f} {:<20.7f}".format(
        avg_prefill_latency, np.mean(decode_latency_per_decode), np.mean(take_up_ratios)))
    print("==============================================================================================")

if __name__ == "__main__":
    benchmark_quest()

# nsys profile --delay 20 --duration 1 --output "$(env TZ='US/Pacific' date +%Y%m%d-%H%M%S).nsys-rep" python text_gen.py