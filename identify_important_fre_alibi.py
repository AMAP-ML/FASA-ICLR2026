from transformers.models.llama.modeling_llama import *
from transformers.models.mistral.modeling_mistral import *
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from monkey_patch.oracle.llama_oracle_static import LLAMA_ATTENTION_CLASSES_flash
from monkey_patch.oracle.mistral_oracle_static import MISTRAL_ATTENTION_CLASSES_flash
from monkey_patch.oracle.qwen_oracle_static import QWEN2_ATTENTION_CLASSES_flash

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as apply_rotary_pos_emb_qwen
import transformers
from collections import defaultdict
from typing import Optional, Tuple, Dict
import pickle
import shutil
import random
import dill
from transformers import AutoTokenizer, LlamaForCausalLM,AutoModelForCausalLM   
from argparse import ArgumentParser
from longbench.longbench_pred import eval_longbench,full_longeval_datasets,dataset2maxlen
from perplexity.evaluate_ppl import evaluate_perplexity_monkey_patch
from reasoning.run_math import evaluate_reasoning_df
import torch
import os
import sys


cache_dir = os.environ.get("FASA_MODEL_DIR", "./models")


def main(model_name,task_type,prefilling_len,decoding_len,dataset_name):
    global ranks_dir
    global sparsity_list 
    global ranks_file_name
    global total_tokens
    

    root_dir = "dimension_rankings"
    output_dir = f"{root_dir}/{model_name}/outputs"
    os.makedirs(output_dir, exist_ok=True)
    ranks_dir =f"{root_dir}/{model_name}"
    os.makedirs(ranks_dir,exist_ok=True)
    model_path = os.path.join(cache_dir, model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if task_type == "longbench":
        for dataset in [dataset_name]:  
            total_tokens=0
            ranks_file_name = os.path.join(ranks_dir,f"{dataset}.pkl")
            if dataset2maxlen[dataset] >=512:
                eval_longbench(model, tokenizer, model_name, datasets=[dataset],output_dir=output_dir,evaluate_num=1)
            else:
                eval_longbench(model, tokenizer, model_name, datasets=[dataset],output_dir=output_dir,evaluate_num=8)
            # with open(ranks_file_name,"wb") as f:
            #     dill.dump(res_dict,f)
    elif task_type == "language_modeling":
        for dataset_name in ["c4"]:
            total_tokens=0
            ranks_file_name = os.path.join(ranks_dir,f"{dataset_name}_{prefilling_len}_{decoding_len}.pkl")
            evaluate_perplexity_monkey_patch(model, tokenizer,prefilling_len,dataset_name,model_name,decoding_len,limit=1)
            with open(ranks_file_name,"wb") as f:
                dill.dump(res_dict,f)
    elif task_type == "reasoning":
        for dataset_name in ["aime24","math"]:
            total_tokens=0
            ranks_file_name = os.path.join(ranks_dir,f"{dataset_name}.pkl")
            evaluate_reasoning_df(model, tokenizer,dataset_name,eval_batch_size=1,limit=1)
            with open(ranks_file_name,"wb") as f:
                dill.dump(res_dict,f)
    else:
        raise NotImplementedError
if __name__ == "__main__":
    # model_name = "Llama-3.2-3B-Instruct"
    # model_name = "Llama-2-7b-chat-hf"
    # model_name="Meta-Llama-3-70B-Instruct"
    # model_name="Meta-Llama-3.1-8B-Instruct"
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.2-3B-Instruct")
    parser.add_argument("--task_type", type=str, default="longbench")
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--prefilling_len",type=int,default=4096)
    parser.add_argument("--decoding_len",type=int,default=128)
    # parser.add_argument("--frequency_group", type=int)
    args = parser.parse_args()
    main(args.model_name,args.task_type,args.prefilling_len,args.decoding_len,args.dataset_name)