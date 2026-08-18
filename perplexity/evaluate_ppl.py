import os
import numpy as np
import torch
from datasets import load_dataset
import random
import io
import json
from transformers import AutoTokenizer,AutoModelForCausalLM
import argparse
from tqdm import tqdm
import torch.nn as nn
from monkey_patch.snapkv import snapkv_ as snapkv
from monkey_patch.streamingllm import streamingllm_ as streamingllm
from monkey_patch.oracle.oracle_static import replace_oracle_static
from monkey_patch.frequency.dynamic_frequency import replace_llama_df,replace_mistral_df,replace_qwen_df
from monkey_patch.frequency.utils import constructe_adaptive_selection
from longbench.longbench_pred import set_model_params,load_model_and_tokenizer
from monkey_patch.frequency.cache_design import DynamicCache as DynamicCache_dynamic
from monkey_patch.oracle.oracle_static import replace_models_flash
from torch.nn import CrossEntropyLoss
import transformers
from monkey_patch.quest.quest_attention import enable_quest_attention_eval
from monkey_patch.quest.llama import enable_tuple_kv_cache_for_llama
from monkey_patch.quest.mistral import enable_tuple_kv_cache_for_mistral
from monkey_patch.quest.qwen import enable_tuple_kv_cache_for_qwen
cache_dir = os.environ.get("FASA_MODEL_DIR", "./models")

def get_eval_loaders(name, tokenizer):
    if "wikitext" in name:
        testdata = load_dataset(
            "data_dir/wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "data_dir/c4",
            data_files={"en.noblocklist": "en.noblocklist/c4-validation.00000-of-00008.json.gz"},
            split="en.noblocklist",
            
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc

    if "pg19" in name:
        testdata = load_dataset(
            "data_dir/pg19-test",
            split="test",
            # trust_remote_code=True,
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    raise NotImplementedError



def evaluate_perplexity_monkey_patch(model,tokenizer,prefilling_len,dataset_name,model_name,decoding_len=128,limit=2,reverse=False):
    testloader = get_eval_loaders(dataset_name, tokenizer)
    total_len = prefilling_len + decoding_len
    testenc = testloader.input_ids
    nsamples = testenc.numel() // total_len
    use_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    limit = min(nsamples,limit)
    pool = list(range(nsamples))
    if reverse:
        pool = pool[::-1]

    for k in tqdm(pool[:limit]):
        batch = testenc[:, (k * total_len) : ((k + 1) * total_len)].to(model.device)
        batch_len = batch.size()[-1]
        assert batch_len>prefilling_len+1
        pbar = tqdm(enumerate(range(prefilling_len-1,batch_len-1)))
        past_key_values = None

        for idx,i in pbar:
            if idx==0:
                input_ids = batch[:,:(i+1)].to(model.device)
            else:
                input_ids = batch[:,i:(i+1)]

            with torch.inference_mode():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True    
                )
                
                outputs.logits = outputs.logits[:,-1:,:]
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = batch[:, i + 1 : i + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
            nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.tensor(nlls).mean())
    print(dataset_name, ppl.item())
    model.config.use_cache = use_cache
    return ppl.item()



def main(model_name,prefilling_len,dataset_name,decoding_len,limit,min_k,max_k,threshold_ratio,reverse):
    if args.compression_methods in ["snapkv", "streamingllm"]:
        if args.compression_methods == "snapkv":
            snapkv.replace_llama()
            snapkv.replace_mistral()
            snapkv.replace_qwen()
        else:
            streamingllm.replace_llama()
            streamingllm.replace_mistral()
            streamingllm.replace_qwen()
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
        compress_params = json.load(open("config/compress_params.json","r"))[args.compression_methods][str(int(args.budget))]
            
        set_model_params(model, args.compression_methods, **compress_params)
    elif args.compression_methods == "base":
        replace_models_flash(model_name)
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
    elif args.compression_methods == "oracle":
        replace_oracle_static(model_name,"oracle",args.budget,int(args.keep_high_dim))
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
    elif args.compression_methods == "dynamic_frequency":
        records = constructe_adaptive_selection(model_name,args.budget,args.threshold_ratio,args.min_k,args.max_k,dataset_name=f"{args.cal_dataset}")
        if "llama" in model_name.lower() or "longchat" in model_name.lower():
            transformers.models.llama.modeling_llama.DynamicCache = DynamicCache_dynamic
            replace_llama_df(args.budget,records,args.layer_control)
        elif "mistral" in model_name.lower():
            replace_mistral_df(args.budget,records)
        elif "qwen" in model_name.lower():
            replace_qwen_df(args.budget,records)
        else:
            raise NotImplementedError
        
        # replace_llama_df(model_name,args.budget,args.threshold_ratio,args.min_k,args.max_k)
        # replace_mistral_df(model_name,args.budget,args.threshold_ratio,args.min_k,args.max_k)
        # replace_qwen_df(model_name,args.budget,args.threshold_ratio,args.min_k,args.max_k)
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
    elif args.compression_methods == "quest":
        if "llama" in model_name.lower() or "longchat" in model_name.lower():
            enable_tuple_kv_cache_for_llama()
        elif "mistral" in model_name.lower():
            enable_tuple_kv_cache_for_mistral()
        elif "qwen" in model_name.lower():
            enable_tuple_kv_cache_for_qwen()
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
        model = model.eval()
        args.layer_id = model.config.num_hidden_layers
        enable_quest_attention_eval(model, args)
    else:
        raise NotImplementedError


    ppl= evaluate_perplexity_monkey_patch(model,tokenizer,prefilling_len,dataset_name,model_name,decoding_len,limit=limit,reverse=reverse)
    os.makedirs("perplexity/results",exist_ok=True)
    output_dir = f"perplexity/results/{args.compression_methods}_{dataset_name}_{prefilling_len}_{decoding_len}_{limit}.json"
    if  args.compression_methods in ["snapkv", "streamingllm","oracle"]:
        specified_key = f"{model_name}_{args.compression_methods}_{args.budget}"
    elif  args.compression_methods == "base":
        specified_key = f"{model_name}_{args.compression_methods}"
    elif  args.compression_methods == "dynamic_frequency":
        specified_key = f"{model_name}_{args.compression_methods}_{args.budget}_{args.threshold_ratio}_{args.min_k}_{args.max_k}"
    elif args.compression_methods == "quest":
        specified_key = f"{model_name}_{args.compression_methods}_{args.budget}"
    if os.path.exists(output_dir):
        with open(output_dir,"r") as f:
            old_results = json.load(f)
            old_results.update({specified_key:ppl})
    else:
        old_results = {specified_key:ppl}
    with open(output_dir,"w") as f:
        json.dump(old_results,f,indent=4)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.2-3B-Instruct")
    parser.add_argument("--compression_methods", type=str, default=None)
    parser.add_argument("--budget", type=float, default=None)
    #oracle noargs
    #staic 
    parser.add_argument('--keep_high_dim', type=int, default=32)
    #frequency
    parser.add_argument("--min_k",type=int, default=16)
    parser.add_argument("--max_k",type=int, default=16)
    parser.add_argument("--threshold_ratio",type=float, default=0.8)
    parser.add_argument("--cal_dataset",type=str, default="wikitext2")
    parser.add_argument("--layer_control",type=int, default=-1)

    parser.add_argument("--chunk_size",type=int, default=16)




    parser.add_argument("--prefilling_len", type=int, default=2048)
    parser.add_argument("--dataset_name", type=str, default="wikitext2")
    parser.add_argument("--decoding_len", type=int, default=128)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--reverse", action="store_true")

    args = parser.parse_args()
    main(args.model_name,args.prefilling_len,args.dataset_name,args.decoding_len,args.limit,args.min_k,args.max_k,args.threshold_ratio,args.reverse)