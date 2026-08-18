import os
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monkey_patch.snapkv import snapkv_ as snapkv
from monkey_patch.streamingllm import streamingllm_ as streamingllm
from monkey_patch.oracle.oracle_static import replace_oracle_static


from monkey_patch.frequency.dynamic_frequency import replace_llama_df, replace_mistral_df, replace_qwen_df
from monkey_patch.frequency.utils import constructe_adaptive_selection
from longbench.eval import scorer


from monkey_patch.quest.quest_attention import enable_quest_attention_eval
from monkey_patch.quest.llama import enable_tuple_kv_cache_for_llama
from monkey_patch.quest.mistral import enable_tuple_kv_cache_for_mistral
from monkey_patch.quest.qwen import enable_tuple_kv_cache_for_qwen

cache_dir = os.environ.get("FASA_MODEL_DIR", "./models")

dataset2prompt={
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

dataset2maxlen={
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2maxlen={
    "Mistral-7B-Instruct-v0.3":31500,
    "Qwen2.5-7B-Instruct": 127500,
    "Qwen2.5-14B-Instruct": 127500,
    "Qwen2.5-32B-Instruct": 127500,
    "Llama-3.2-3B-Instruct": 127500,
    "Qwen1.5-7B-Chat": 30000,
    "Meta-Llama-3-8B-Instruct": 7500,
    "Meta-Llama-3.1-8B-Instruct": 127500,
    "Meta-Llama-3-70B-Instruct":7500,
    "longchat-7b-v1.5-32k":31500,
    "Qwen2.5-3B-Instruct": 30000,
    "GLM-4-9B-Chat": 120000,
    "Llama-3.1-8B-Instruct": 8000,
    "Llama-3.1-70B-Instruct": 120000,
    "Llama-3.3-70B-Instruct": 120000,
    "Llama-3.1-Nemotron-70B-Instruct": 120000,
    "Llama-2-7b-chat-hf": 3500,
    "Llama-2-7B-32K-Instruct": 31500 ,
    "Qwen2.5-14B-Instruct-1M": 500000,
    "DeepSeek-R1-Distill-Llama-8B": 8000,
    "DeepSeek-R1-Distill-Qwen-14B": 8000,
    "DeepSeek-R1-Distill-Qwen-32B": 8000,
    "bloom-3b": 4000,
    "Baichuan-13B-Chat": 3000,
    "mpt-30b-instruct": 8000,
}
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    return data
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    #common
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--compression_methods', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--budget', type=float, default=None)

    #oracle noargs
    #staic 
    parser.add_argument('--keep_high_dim', type=int, default=32)
    #frequency
    parser.add_argument("--min_k",type=int, default=1)
    parser.add_argument("--max_k",type=int, default=16)
    parser.add_argument("--threshold_ratio",type=float, default=0.8)
    parser.add_argument("--dataset_name",type=str, default="")
    parser.add_argument("--cal_dataset",type=str, default="qasper")
    parser.add_argument("--layer_control",type=int, default=-1)
    #quest
    parser.add_argument("--chunk_size",type=int, default=16)
    parser.add_argument("--evaluate_num",type=int, default=8)
    #snapkv
    
    
    
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # if "chatglm3" in model_name:
    #     prompt = tokenizer.build_chat_input(prompt)
    # elif "chatglm" in model_name:
    #     prompt = tokenizer.build_prompt(prompt)
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import get_conversation_template
    #     conv = get_conversation_template("vicuna")
    #     conv.append_message(conv.roles[0], prompt)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    # elif "llama2"  in model_name:
    #     print('llama2', model_name)
    #     prompt = f"[INST]{prompt}[/INST]"
    # elif "xgen" in model_name:
    #     header = (
    #         "A chat between a curious human and an artificial intelligence assistant. "
    #         "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    #     )
    #     prompt = header + f" ### Human: {prompt}\n###"
    # elif "internlm" in model_name:
    #     print('internlm')
    #     prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    # elif "qwen" in model_name.lower():
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],tokenize=False,
            add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, 
                        prompt_format, dataset, model_name, out_path,tokenizer,model):

    rsts=[]
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            input = prompt.to(model.device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]
        print(f"====original input length {len(tokenized_prompt)}. context length: {context_length}===")
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                min_length=context_length+1,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            rst = {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
            json.dump(rst, f, ensure_ascii=False)
            f.write("\n")
            rsts.append(rst)   
    return rsts


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",trust_remote_code=True).to("cuda")
    model = model.eval()
    return model, tokenizer
full_longeval_datasets = [
    # "narrativeqa",
    "qasper",
    # "multifieldqa_en",
    # "multifieldqa_zh",
    # "hotpotqa",
    # "2wikimqa",
    # "musique",

    # "dureader",
    # "gov_report",
    # "qmsum",
    # "multi_news",

    # "vcsum",
    # "trec",
    # "triviaqa",
    # "samsum",
    # "lsht",

    # "passage_count",
    # "passage_retrieval_en",
    # "passage_retrieval_zh",
    # "lcc",
    # "repobench-p",
]

def set_model_params(model, compression_methods,**kwargs):
    if compression_methods == "streamingllm":
        start_size = kwargs["start_size"]
        recent_size = kwargs["recent_size"]
        max_capacity_prompts = kwargs["max_capacity_prompts"]
        assert start_size + recent_size == max_capacity_prompts
        layers = len(model.model.layers)
        for i in range(layers):
            model.model.layers[i].self_attn.config.start_size = start_size
            model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts
            model.model.layers[i].self_attn.config.recent_size = recent_size
    elif compression_methods == "snapkv":
        window_sizes = kwargs["window_sizes"]
        max_capacity_prompts = kwargs["max_capacity_prompts"]
        kernel_sizes = kwargs["kernel_sizes"]
        pooling = kwargs["pooling"]
        layers = len(model.model.layers)
        for i in range(layers):
            model.model.layers[i].self_attn.config.window_sizes = window_sizes
            model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts
            model.model.layers[i].self_attn.config.kernel_sizes = kernel_sizes
            model.model.layers[i].self_attn.config.pooling = pooling
    else:
        raise NotImplementedError   
    return model
def eval_longbench(model,tokenizer,model_name,datasets,output_dir,evaluate_num=-1,layer_control=-1):
    rsts = {}
    max_length = model2maxlen[model_name]
    # predict on each dataset
    for dataset in datasets:
        file_path = f"{output_dir}/{dataset}_preds.jsonl"
        data = read_jsonl(f"../longbench/{dataset}.jsonl") if evaluate_num==-1 else read_jsonl(f"../longbench/{dataset}.jsonl")[:evaluate_num]
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        if os.path.exists(file_path):
            os.remove(file_path)
        rst = get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, model_name, file_path,tokenizer,model)
        rsts[dataset] = rst
    scores = dict()
    print("Evaluating on:", datasets)
    print(f"{model_name}")
    for dataset, all_data in rsts.items():
        predictions, answers, lengths = [], [], []
        for data in all_data:
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])
        score = scorer(dataset, predictions, answers, all_classes)
        scores[f"{dataset}_{layer_control}"] = score
        print(f"{dataset}_layer-{layer_control}: {scores[f'{dataset}_{layer_control}']}")
    out_path = f"{output_dir}/{model_name}_result.json"
    with open(out_path, "a") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model_name = args.model_name
    # model2maxlen = json.load(open("longbench/config/model2maxlen.json", "r"))
    
    # max_length = model2maxlen[model_name]
    # dataset2prompt = json.load(open("longbench/config/dataset2prompt.json", "r"))
    # dataset2maxlen = json.load(open("longbench/config/dataset2maxlen.json", "r"))
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
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
    elif args.compression_methods == "oracle":
        replace_oracle_static(model_name,"oracle",int(args.budget),int(args.keep_high_dim))
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
    elif args.compression_methods == "dynamic_frequency":

        records = constructe_adaptive_selection(model_name,args.budget,args.threshold_ratio,args.min_k,args.max_k,dataset_name=args.cal_dataset)
        if "llama" in model_name.lower():
            replace_llama_df(args.budget,records,args.layer_control)
        elif "mistral" in model_name.lower():
            replace_mistral_df(args.budget,records)
        elif "qwen" in model_name.lower():
            replace_qwen_df(args.budget,records)
        else:
            raise NotImplementedError
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
    elif args.compression_methods == "quest":
        if 'llama' in model_name.lower() or 'longchat' in model_name.lower():
            enable_tuple_kv_cache_for_llama()
        if 'mistral' in model_name.lower():
            enable_tuple_kv_cache_for_mistral()
        if "qwen" in model_name.lower():
            enable_tuple_kv_cache_for_qwen()
        model, tokenizer = load_model_and_tokenizer(os.path.join(cache_dir,model_name), model_name)
        model.eval()
        enable_quest_attention_eval(model, args)
    else:
        raise NotImplementedError
    output_dir = os.path.join(args.output_dir, args.compression_methods)
    if args.compression_methods in ["snapkv", "streamingllm","oracle"]:
        output_dir = os.path.join(output_dir, f"{model_name}_{args.budget}")
    elif args.compression_methods == "base":
        output_dir = os.path.join(output_dir, f"{model_name}")
    elif args.compression_methods == "dynamic_frequency":
        output_dir = os.path.join(output_dir, f"{model_name}_{args.budget}_{args.threshold_ratio}_{args.min_k}_{args.max_k}")
    else:
        output_dir = os.path.join(output_dir, f"{args.compression_methods }_{model_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"================layer_control == {args.layer_control}======================")
    eval_longbench(model,tokenizer,model_name,[args.dataset_name],output_dir,-1,args.layer_control)
    
    
    
    
    