from transformers.models.llama.modeling_llama import *
from transformers.models.mistral.modeling_mistral import *
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from monkey_patch.oracle.llama_oracle_static import LLAMA_ATTENTION_CLASSES_flash
from monkey_patch.oracle.mistral_oracle_static import MISTRAL_ATTENTION_CLASSES_flash
from monkey_patch.oracle.qwen_oracle_static import QWEN2_ATTENTION_CLASSES_flash
from longbench.indentify_important_fre_qwen import Qwen2FlashAttention2_forward_partial

import transformers
from collections import defaultdict
from typing import Optional, Tuple, Dict
import pickle
import shutil
import random
import dill
from transformers import AutoTokenizer, LlamaForCausalLM,AutoModelForCausalLM   
from argparse import ArgumentParser
from longbench.longbench_pred import eval_longbench,full_longeval_datasets
import torch
import os
import sys


res_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
def partial_frequency_against_full_frequency(query_states,key_states,value_states,layer_idx,num_hidden_layers):
    bs,q_heads,_,head_dim = query_states.shape
    bs,k_heads,seq_len,_ = key_states.shape
    attn_weights = torch.matmul(query_states,key_states.transpose(2,3))
    sorted_indices_full = torch.argsort(attn_weights, dim=-1, descending=True)

    for frequency_group_idx in range(0,head_dim,2):
        query_states_partial = query_states[:,:,:,frequency_group_idx:(frequency_group_idx+2)]
        key_states_partial = key_states[:,:,:,frequency_group_idx:(frequency_group_idx+2)]
        attn_weights_partial = torch.matmul(query_states_partial, key_states_partial.transpose(2, 3))
        sorted_indices_partial = torch.argsort(attn_weights_partial, dim=-1, descending=True)
        del attn_weights_partial

        for sparsity in sparsity_list:
            preceding_token_num = int(seq_len*sparsity) if sparsity<1 else sparsity
            preceding_token_num = min(seq_len,preceding_token_num)

            sorted_indices_p = sorted_indices_partial[:,:,:,:preceding_token_num]
            sorted_indices_f = sorted_indices_full[:,:,:,:preceding_token_num]

            a = sorted_indices_f.squeeze(2)  # 得到 [1, 24, 1107]
            b = sorted_indices_p.squeeze(2)  # 得到 [1, 24, 1107]

            # 比较，得到交集掩码 [1, 24, 1107, 1107]
            intersection_mask  = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1)  # 对每个位置全量比较
            # 统计 a 中每个元素是否能在 b 中找到（即行有至少一个True），然后求和
            # intersection_mask = cmp.any(-1)   # [1, 24, 1107]
            intersection_count = intersection_mask.sum(-1)  # [1, 24]，即每组有几个a中的值能在b找到
            temporal_res = intersection_count.squeeze(dim=0)/preceding_token_num
            temporal_res = temporal_res.tolist()
            print(f"="*80)
            print(f"sparsity: {sparsity}")
            print(f"intersection: {temporal_res}")
            print(f"="*80)
            
            res_dict[sparsity][frequency_group_idx][layer_idx].append(temporal_res)
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
cache_dir = os.environ.get("FASA_MODEL_DIR", "./models")


def main(model_name):
    global ranks_dir
    global sparsity_list 
    global ranks_file_name
    sparsity_list=[256,512,1024,2048]
    root_dir = "dimension_rankings"
    output_dir = f"{root_dir}/{model_name}/outputs"
    os.makedirs(output_dir, exist_ok=True)
    ranks_dir =f"{root_dir}/{model_name}"
    os.makedirs(ranks_dir,exist_ok=True)
    model_path = os.path.join(cache_dir, model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    for dataset in full_longeval_datasets:  
        ranks_file_name = os.path.join(ranks_dir,f"{dataset}.pkl")
        eval_longbench(model, tokenizer, model_name, datasets=[dataset],output_dir=output_dir,evaluate_num=8)
        with open(ranks_file_name,"wb") as f:
            dill.dump(res_dict,f)
if __name__ == "__main__":
    # model_name = "Llama-3.2-3B-Instruct"
    # model_name = "Llama-2-7b-chat-hf"
    # model_name="Meta-Llama-3-70B-Instruct"
    # model_name="Meta-Llama-3.1-8B-Instruct"
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.2-3B-Instruct")
    # parser.add_argument("--frequency_group", type=int)
    args = parser.parse_args()
    main(args.model_name)