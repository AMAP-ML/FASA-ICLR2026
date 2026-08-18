import json
import random
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import multiprocessing  as mp
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from monkey_patch.frequency.dynamic_frequency import replace_llama_df,replace_mistral_df,replace_qwen_df
from monkey_patch.frequency.utils import constructe_adaptive_selection
from monkey_patch.snapkv import snapkv_ as snapkv
from monkey_patch.streamingllm import streamingllm_ as streamingllm
from monkey_patch.oracle.oracle_static import replace_oracle_static
from longbench.longbench_pred import set_model_params
from monkey_patch.oracle.oracle_static import replace_models_flash

from monkey_patch.quest.quest_attention import enable_quest_attention_eval
from monkey_patch.quest.llama import enable_tuple_kv_cache_for_llama
from monkey_patch.quest.mistral import enable_tuple_kv_cache_for_mistral
from monkey_patch.quest.qwen import enable_tuple_kv_cache_for_qwen
cache_dir = os.environ.get("FASA_MODEL_DIR", "./models")
dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 8192,
    "aime24": 32768,
    "math": 16384,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"
def load_prompts_and_data(dataset_name,limit):
    prompt_tokenizer = AutoTokenizer.from_pretrained(os.path.join(cache_dir,"DeepSeek-R1"))
    prompts = []
    test_data = []
    with open(f"reasoning/data/{dataset_name}.jsonl") as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[dataset_name][0]
            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)
            prompt = prompt_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True ,
                tokenize=False
            ) 
            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)
            if len(prompts) >= limit:
                break
    return prompts, test_data

def load_tokenizer_and_model(model_name,device):
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(cache_dir,model_name), use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
       os.path.join(cache_dir,model_name),
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        device_map="auto",
        # tp_plan="auto",
        use_cache=True,
    )
    model.eval()
    return model,tokenizer

def evaluate_reasoning_worker(args,model_name,prompts,test_data,save_path,dataset_name,device,eval_batch_size=1):
    if args.method == "dynamic_frequency":
        # records = constructe_adaptive_selection(model_name,args.budget,args.threshold_ratio,args.min_k,args.max_k,dataset_name=args.cal_dataset)
        if "llama" in model_name.lower():
            replace_llama_df(args.budget,args.records,args.layer_control)
        elif "mistral" in model_name.lower():
            replace_mistral_df(args.budget,args.records)
        elif "qwen" in model_name.lower():
            replace_qwen_df(args.budget,args.records)
        else:
            raise NotImplementedError
        model,tokenizer = load_tokenizer_and_model(model_name,device)
    elif args.method == "base":
        replace_models_flash(model_name)
        model,tokenizer = load_tokenizer_and_model(model_name,device)
    elif args.method == "oracle":
        replace_oracle_static(model_name,"oracle",int(args.budget),int(args.keep_high_dim))
        model, tokenizer = load_tokenizer_and_model(model_name,device)
    elif args.method in ["snapkv", "streamingllm"]:
        if args.method == "snapkv":
            snapkv.replace_llama()
            snapkv.replace_mistral()
            snapkv.replace_qwen()
        else:
            streamingllm.replace_llama()
            streamingllm.replace_mistral()
            streamingllm.replace_qwen()
        model, tokenizer = load_tokenizer_and_model(model_name,device)
        if args.budget > 1:
            compress_params = json.load(open("config/compress_params.json","r"))[args.method][str(int(args.budget))]
        else:
            compress_params = json.load(open("config/compress_params.json","r"))[args.method][str(args.budget)]
        set_model_params(model, args.method, **compress_params)
    elif args.method == "quest":
        print(f"applying quest===================")
        if "llama" in model_name.lower() or "longchat" in model_name.lower():
            enable_tuple_kv_cache_for_llama()
        elif "mistral" in model_name.lower():
            enable_tuple_kv_cache_for_mistral()
        elif "qwen" in model_name.lower():
            enable_tuple_kv_cache_for_qwen()
        model, tokenizer = load_tokenizer_and_model(model_name,device)
        model = model.eval()
        enable_quest_attention_eval(model, args)
    else:
        raise NotImplementedError
    
    fout = open(f"{save_path}", "a")

    for i in tqdm(range(0, len(prompts), eval_batch_size)):

        batch_prompts = prompts[i : i + eval_batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
        ).to(model.device)

        prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()
        output = model.generate(
            **tokenized_prompts,
            max_length=dataset2max_length[dataset_name],
            temperature=0.6,
            # do_sample=False,
            # num_beams=1,
        )
        batch_token_stats = []
        for j in range(output.size(0)):
            total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())

            prefill = prefill_lengths[j]
            output_tokens = total_tokens - prefill

            batch_token_stats.append(
                {
                    "sample_idx": i + j,
                    "prefill_tokens": prefill,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )

        batch_outputs = tokenizer.batch_decode(
            [output[j][prefill_lengths[j] :] for j in range(output.size(0))],
            skip_special_tokens=True,
        )

        torch.cuda.empty_cache()

        for j in range(len(batch_outputs)):
            sample_idx = batch_token_stats[j]["sample_idx"]
            test_data[sample_idx]["prompt"] = batch_prompts[j]
            test_data[sample_idx]["output"] = batch_outputs[j]
            test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
            test_data[sample_idx]["output_tokens"] = batch_token_stats[j]["output_tokens"]
            test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
            test_data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]

            fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")

    fout.close()



def main(args):
    
    prompts, test_data = load_prompts_and_data(args.dataset_name,limit=10000)
    evaluate_reasoning_worker(args,args.model_name,prompts,test_data,args.save_path,args.dataset_name,0,args.eval_batch_size)
    


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--cal_dataset", type=str, default="gsm8k") 
    parser.add_argument("--layer_control", type=int, default=-1)

    # method config
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["dynamic_frequency", "base","oracle","snapkv","quest"],
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--keep_high_dim", type=int, default=32)
    parser.add_argument("--min_k",type=int, default=16)
    parser.add_argument("--max_k",type=int, default=16)
    parser.add_argument("--threshold_ratio",type=float, default=0.8)

    parser.add_argument("--chunk_size", type=int, default=16)

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    if args.max_length == -1: args.max_length = dataset2max_length[args.dataset_name]
    if args.method == "dynamic_frequency":
        records = constructe_adaptive_selection(args.model_name,args.budget,args.threshold_ratio,args.min_k,args.max_k,dataset_name=args.cal_dataset)
        
        args.records = records
    
    os.makedirs(f"reasoning/{args.dataset_name}", exist_ok=True)
    if args.method == "base":
        save_path = f"reasoning/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.method}_{args.seed}.jsonl"
    elif args.method in [ "oracle","snapkv"]:
        save_path = f"reasoning/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.method}_{args.budget}_{args.seed}.jsonl"
    elif args.method == "dynamic_frequency":
        save_path = f"reasoning/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.method}_{args.budget}_{args.min_k}_{args.max_k}_{args.seed}.jsonl"
    elif args.method == "quest":
        save_path = f"reasoning/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.method}_{args.budget}_{args.seed}.jsonl"
    args.save_path = save_path
    if os.path.exists(args.save_path):     # 检查文件是否存在
        os.remove(args.save_path)          # 删除文件
        print(f"{args.save_path} 已删除")
    else:
        print(f"{args.save_path} 不存在")
    main(args)
