from transformers.models.llama.modeling_llama import *
from transformers.models.mistral.modeling_mistral import *
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from monkey_patch.oracle.llama_oracle_static import LLAMA_ATTENTION_CLASSES_flash
from monkey_patch.oracle.mistral_oracle_static import MISTRAL_ATTENTION_CLASSES_flash
from monkey_patch.oracle.qwen_oracle_static import QWEN2_ATTENTION_CLASSES_flash
import json
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
def extract_group(file_path):
    with open(file_path, "rb") as f:
        return dill.load(f)

dominant_distribution = defaultdict(list)
res_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

def partial_frequency_against_full_frequency(query_states,key_states,value_states,layer_idx,num_hidden_layers):
    if layer_idx <3:
        return
    bs,q_heads,_,head_dim = query_states.shape
    bs,k_heads,seq_len,_ = key_states.shape
    attn_weights = torch.matmul(query_states,key_states.transpose(2,3))
    # attn_scores = torch.softmax(attn_weights, dim=-1)
    sorted_indices_full = torch.argsort(attn_weights, dim=-1, descending=True)
    layer_fc_map = fc_map[1][layer_idx]
    
    frequency_groups,head_groups = torch.where(layer_fc_map>0.54) if dominant else torch.where(layer_fc_map<0.15)
    size = min(len(frequency_groups),100)
    with torch.no_grad():
        for head_group_idx in head_groups[:size]:
            for frequency_group_idx in frequency_groups[:size]:
                query_states_partial = query_states[:,head_group_idx,:,2*frequency_group_idx:(2*frequency_group_idx+2)]
                key_states_partial = key_states[:,head_group_idx,:,2*frequency_group_idx:(2*frequency_group_idx+2)]
                attn_weights_partial = torch.matmul(query_states_partial, key_states_partial.transpose(1, 2))
                # attn_scores_partial = torch.softmax(attn_weights_partial, dim=-1)
                sorted_indices_partial = torch.argsort(attn_weights_partial, dim=-1, descending=True)

                for sparsity in sparsity_list:
                    preceding_token_num = int(seq_len*sparsity) if sparsity<1 else sparsity
                    preceding_token_num = min(seq_len,preceding_token_num)

                    sorted_indices_p = sorted_indices_partial[:,:,:preceding_token_num]
                    sorted_indices_f = sorted_indices_full[:,head_group_idx,:,:preceding_token_num]
                  
                    

                    a = sorted_indices_f  # 得到 [1, 24, 1107]
                    b = sorted_indices_p  # 得到 [1, 24, 1107]

                    # 比较，得到交集掩码 [1, 24, 1107, 1107]
                    intersection_mask_matrix = (a.unsqueeze(-1) == b.unsqueeze(-2))
                    intersection_mask  = intersection_mask_matrix.any(-1)  # 对每个位置全量比较
                    # 统计 a 中每个元素是否能在 b 中找到（即行有至少一个True），然后求和
                    # intersection_mask = cmp.any(-1)   # [1, 24, 1107]
                    intersection_count = intersection_mask.sum(-1)  # [1, 24]，即每组有几个a中的值能在b找到
                    temporal_res = intersection_count.squeeze(dim=0)/preceding_token_num
                    #######################################################################
                    
                    
                    split_groups = [0,50,100,150,200,256]
                    for idx in range(5):
                        dominant_mask_topk = intersection_mask[:,:,split_groups[idx]:split_groups[idx+1]].sum(-1)
                        dominant_distribution[f"{split_groups[idx]}~{split_groups[idx+1]}"].extend((dominant_mask_topk/(split_groups[idx+1]-split_groups[idx])).tolist()[0])
                    # print(f"dominant_distribution: {dominant_distribution}")
                    keys = dominant_distribution.keys()
                    domi_stats = [ sum(dominant_distribution[key])/len(dominant_distribution[key]) for key in keys]
                    # non_domi_stats = [ sum(non_domi_dict[key])/len(non_domi_dict[key]) for key in keys]
        for key, domi_stat in zip(keys, domi_stats):
            print(f"key: {key}, domi_stat: {domi_stat}")
        breakpoint()
                    
                    
            
                    

                


    # if len(res_dict[sparsity][frequency_group_idx][layer_idx]) % 10 ==0:
    #     with open(ranks_file_name,"wb") as f:
    #         dill.dump(res_dict,f)

def LlamaFlashAttention2_forward_partial(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin,position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        #############################################################################################################
        ## selection logic
        if query_states.shape[2] == 1:
            _,query_head_num,_,head_dim = query_states.shape
            _,key_head_num,seq_len,_ = key_states.shape

            key_states = repeat_kv(key_states, query_head_num//key_head_num)
            value_states = repeat_kv(value_states, query_head_num//key_head_num)
            partial_frequency_against_full_frequency(query_states,key_states,value_states,self.layer_idx,self.config.num_hidden_layers)
            
            if self.layer_idx == self.config.num_hidden_layers-1:
                global total_tokens 
                total_tokens+=1
                print(f"total_tokens: {total_tokens}")
        ##############################################################################################################

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def MistralFlashAttention2_forward_partial(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin,position_ids)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #######################################################################
        if query_states.shape[2] == 1:
            partial_frequency_against_full_frequency(query_states,key_states,value_states,self.layer_idx,self.config.num_hidden_layers)
            
            if self.layer_idx == self.config.num_hidden_layers-1:
                global total_tokens 
                total_tokens+=1
                print(f"total_tokens: {total_tokens}")
        #######################################################################
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype


            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self.config, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def Qwen2FlashAttention2_forward_partial(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb_qwen(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #############################################################################################################
        ## selection logic
        if query_states.shape[2] == 1:
            partial_frequency_against_full_frequency(query_states,key_states,value_states,self.layer_idx,self.config.num_hidden_layers)
            global total_tokens 
            if self.layer_idx == self.config.num_hidden_layers-1:
                total_tokens+=1
                print(f"total_tokens: {total_tokens}")
        ##############################################################################################################

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
def replace_llama_partial():
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = LlamaFlashAttention2_forward_partial
    transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES = LLAMA_ATTENTION_CLASSES_flash
def replace_mistral_partial():
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = MistralFlashAttention2_forward_partial
    transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES = MISTRAL_ATTENTION_CLASSES_flash

def replace_qwen_partial():
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = Qwen2FlashAttention2_forward_partial
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES = QWEN2_ATTENTION_CLASSES_flash



replace_llama_partial()
replace_mistral_partial()
replace_qwen_partial()
cache_dir = os.environ.get("FASA_MODEL_DIR", "./models")


def main(model_name,task_type,prefilling_len,decoding_len,dataset_name,dominant_arg):
    global ranks_dir
    global sparsity_list 
    global ranks_file_name
    global total_tokens
    global fc_map
    global dominant
    dominant = True  if dominant_arg else False
    print(f"="*40)
    print(f"this is for dominant: {dominant}")
    print(f"="*40)
    sparsity_list=[256]
    fc_map = extract_group(f"dimension_rankings/{model_name}/qasper_agreement.pkl")
    root_dir = "dimension_rankings"
    output_dir = f"{root_dir}/{model_name}/outputs"
    os.makedirs(output_dir, exist_ok=True)
    ranks_dir =f"{root_dir}/{model_name}"
    os.makedirs(ranks_dir,exist_ok=True)
    model_path = os.path.join(cache_dir, model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if task_type == "longbench":
        for dataset in [dataset_name]:  
            total_tokens=0
            ranks_file_name = os.path.join(ranks_dir,f"{dataset}_dominant.json") if dominant else os.path.join(ranks_dir,f"{dataset}__dominant_non.json")
            if dataset2maxlen[dataset] >=512:
                eval_longbench(model, tokenizer, model_name, datasets=[dataset],output_dir=output_dir,evaluate_num=1)
            else:
                eval_longbench(model, tokenizer, model_name, datasets=[dataset],output_dir=output_dir,evaluate_num=8)
            with open(ranks_file_name,"w",encoding="utf-8") as f:
                json.dump(dominant_distribution,f,indent=4)
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
    parser.add_argument("--dominant",action="store_true")
    # parser.add_argument("--frequency_group", type=int)
    args = parser.parse_args()
    main(args.model_name,args.task_type,args.prefilling_len,args.decoding_len,args.dataset_name,args.dominant)