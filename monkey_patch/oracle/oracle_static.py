

from transformers.modeling_flash_attention_utils import _flash_attention_forward
import os 
from transformers.models.llama.modeling_llama import repeat_kv
from monkey_patch.oracle.qwen_oracle_static import Qwen2FlashAttention2_forward_oracle,QWEN2_ATTENTION_CLASSES_flash
from monkey_patch.oracle.llama_oracle_static import LlamaFlashAttention2_forward_oracle,LLAMA_ATTENTION_CLASSES_flash
from monkey_patch.oracle.mistral_oracle_static import MistralFlashAttention2_forward_oracle,MISTRAL_ATTENTION_CLASSES_flash
from monkey_patch.oracle.gemma2_oracle_static import Gemma2FlashAttention2_forward_oracle,GEMMA2_ATTENTION_CLASSES_flash
import transformers
import random
import torch


def static_frequency_selection_logic(query_states, key_states, value_states,budget,keep_high_dim):
    if query_states.shape[2] == 1 and self.layer_idx>0:
        _,query_head_num,_,head_dim = query_states.shape
        _,key_head_num,seq_len,_ = key_states.shape
        if query_head_num != key_head_num:
            key_states = repeat_kv(key_states, query_head_num//key_head_num)
            value_states = repeat_kv(value_states, query_head_num//key_head_num)
        query_states_low = query_states[:,:,:,-keep_high_dim:]
        key_states_low = key_states[:,:,:,-keep_high_dim:]
        attn_weights_low = torch.matmul(query_states_low, key_states_low.transpose(2, 3))
        sorted_indices = torch.argsort(attn_weights_low, dim=-1, descending=True)
        del query_states_low, key_states_low, attn_weights_low
        preceding_token_num = int(seq_len*budget) if budget<1 else budget
        preceding_token_num = min(preceding_token_num, seq_len)
        sorted_indices = sorted_indices[:,:,:,:preceding_token_num]
        index = sorted_indices.transpose(2,3).expand( -1, -1,-1,key_states.size(-1))  # [batch, heads, seq_len, top_k, head_dim]
        key_states = torch.gather( key_states,  dim=2,  index=index)  
        value_states = torch.gather(value_states,  dim=2, index=index)
        return key_states,value_states
    else:
        return key_states,value_states
def oracle_selection_logic(query_states, key_states, value_states,budget,keep_high_dim):
    _,query_head_num,q_len,_ = query_states.shape
    _,key_head_num,seq_len,head_dim = key_states.shape
    if q_len !=1:
        return key_states,value_states
    else:
        if query_head_num != key_head_num:
            key_states = repeat_kv(key_states, query_head_num//key_head_num)
            value_states = repeat_kv(value_states, query_head_num//key_head_num)
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3))
        sorted_indices = torch.argsort(attn_weights, dim=-1, descending=True)

        preceding_token_num = int(seq_len*budget) if budget<1 else budget
        preceding_token_num = min(preceding_token_num, seq_len)
        sorted_indices = sorted_indices[:,:,:,:int(preceding_token_num)]

        index = sorted_indices.transpose(2,3).expand( -1, -1,-1,head_dim)  # [batch, heads, seq_len, top_k, head_dim]
        key_states = torch.gather( key_states,  dim=2,  index=index)  
        value_states = torch.gather(value_states,  dim=2, index=index)
        return key_states,value_states




def replace_oracle_static(model_name,selection_logic, budget,keep_high_dim):
    """替换模型的注意力机制为 oracle 版本"""
    if selection_logic=="oracle":
        selection_logic_func = oracle_selection_logic  
    elif selection_logic=="df":
        selection_logic_func = static_frequency_selection_logic
    else:
        raise NotImplemented
    def create_llama_forward(budget,keep_high_dim):
        def forward(self, *args, **kwargs):
            kwargs['selection_func'] = selection_logic_func
            kwargs['budget'] = budget
            kwargs['keep_high_dim'] = keep_high_dim
            return LlamaFlashAttention2_forward_oracle(self, *args, **kwargs)
        return forward
    
    def create_mistral_forward(budget,keep_high_dim):
        def forward(self, *args, **kwargs):
            kwargs['selection_func'] = selection_logic_func
            kwargs['budget'] = budget
            kwargs['keep_high_dim'] = keep_high_dim
            return MistralFlashAttention2_forward_oracle(self, *args, **kwargs)
        return forward
    
    def create_qwen_forward(budget,keep_high_dim):
        def forward(self, *args, **kwargs):
            kwargs['selection_func'] = selection_logic_func
            kwargs['budget'] = budget
            kwargs['keep_high_dim'] = keep_high_dim
            return Qwen2FlashAttention2_forward_oracle(self, *args, **kwargs)
        return forward
    def create_gemma2_forward(budget,keep_high_dim):
        def forward(self, *args, **kwargs):
            kwargs['selection_func'] = selection_logic_func
            kwargs['budget'] = budget
            kwargs['keep_high_dim'] = keep_high_dim
            return Gemma2FlashAttention2_forward_oracle(self, *args, **kwargs)
        return forward
    if "llama" in model_name.lower() or "long" in model_name.lower():
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = create_llama_forward(budget,keep_high_dim)
        transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES = LLAMA_ATTENTION_CLASSES_flash
    elif "mistral" in model_name.lower():
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = create_mistral_forward(budget,keep_high_dim)
        transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES = MISTRAL_ATTENTION_CLASSES_flash
    elif "qwen" in model_name.lower():
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = create_qwen_forward(budget,keep_high_dim)
        transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES = QWEN2_ATTENTION_CLASSES_flash
    elif "gemma2" in model_name.lower():
        transformers.models.gemma2.modeling_gemma2.Gemma2FlashAttention2.forward = create_gemma2_forward(budget,keep_high_dim)
        transformers.models.gemma2.modeling_gemma2.GEMMA2_ATTENTION_CLASSES = GEMMA2_ATTENTION_CLASSES_flash
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def replace_models_flash(model_name):
    if "llama" in model_name.lower() or "long" in model_name.lower():
        transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES = LLAMA_ATTENTION_CLASSES_flash
    elif "mistral" in model_name.lower():
        transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES = MISTRAL_ATTENTION_CLASSES_flash
    elif "qwen" in model_name.lower():
        transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES = QWEN2_ATTENTION_CLASSES_flash
    elif "gemma2" in model_name.lower():
        transformers.models.gemma2.modeling_gemma2.GEMMA2_ATTENTION_CLASSES = GEMMA2_ATTENTION_CLASSES_flash
    else:
        raise ValueError(f"Unsupported model: {model_name}")
