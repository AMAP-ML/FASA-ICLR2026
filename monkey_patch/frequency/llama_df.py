from transformers.models.llama.modeling_llama import *
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from monkey_patch.oracle.llama_oracle_static import LLAMA_ATTENTION_CLASSES_flash
from monkey_patch.frequency.utils import core_module_with_padding
import transformers
import random
from collections import defaultdict
import pickle
import numpy as np
import torch
from functools import partial
from typing import Dict, Optional



def LlamaFlashAttention2_forward_dynamic_frequency(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    budget: int=4096,
    records:Dict = None,
    layer_control: int=1
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
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    # print(f"===========================enter=========================")
    if query_states.size()[2] !=1:
        # if self.layer_idx in list(range(self.config.num_hidden_layers)):
        #     _,key_selected,key_unselected,_,_  = core_module_with_padding(query_states,key_states,self.layer_idx,budget,records["16"])
        # else:
        _,key_selected,key_unselected,_,_  = core_module_with_padding(query_states,key_states,self.layer_idx,budget,records)
        _,_,_ = past_key_value.update(key_selected,key_unselected,value_states,self.layer_idx)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    else:
        # if self.layer_idx in list(range(self.config.num_hidden_layers)):
        #     query_selected,key_selected,key_unselected, selection_expanded,unselected_expanded = core_module_with_padding(query_states,key_states,self.layer_idx,budget,records["16"])
        # else:
        query_selected,key_selected,key_unselected, selection_expanded,unselected_expanded = core_module_with_padding(query_states,key_states,self.layer_idx,budget,records)
        key_selected,key_unselected,value_states = past_key_value.update(key_selected,key_unselected,value_states,self.layer_idx)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        control_layer_list = [layer_control] if layer_control in list(range(self.config.num_hidden_layers)) else list(range(1,self.config.num_hidden_layers))



        if self.layer_idx in control_layer_list:
            attn_weights = torch.matmul(query_selected, key_selected.transpose(2,3))
            seq_len = key_selected.size(2)
            ###############################
            #add layer allocation strategy
            # print(f"self.layer_idx:{self.layer_idx},budget:{budget}")
            # if self.layer_idx < 5:
            #     budget = budget*1.2
            # else:
            #     budget = (budget*32-budget*1.2*5)/27
            ###############################
            preceding_token_num = int(seq_len * budget) if budget < 1 else int(budget)
            if seq_len<preceding_token_num:
                preceding_token_num = seq_len
            top_indices = torch.topk(attn_weights, k=int(preceding_token_num), dim=-1).indices.transpose(2,3)

            key_selected = torch.gather( key_selected,  dim=2,  index=top_indices.expand(-1, -1, -1, key_selected.size()[-1]))
            key_unselected = torch.gather(key_unselected, dim=2, index=top_indices.expand(-1, -1, -1, key_unselected.size()[-1]))
            value_states = torch.gather(value_states, dim=2, index=top_indices.expand(-1, -1, -1, self.head_dim))
        else:
            pass
        # print(f"key_states.shape:{key_selected.size()}")
        key_states = torch.zeros((bsz,key_selected.size()[1],key_selected.size()[2],self.head_dim),device=query_states.device,dtype=query_states.dtype)
        key_states.scatter_(-1, selection_expanded.expand(-1, -1, key_selected.size()[2],-1), key_selected)
        key_states.scatter_(-1, unselected_expanded.expand(-1, -1, key_unselected.size()[2],-1), key_unselected)

  



        
        
    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
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

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

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





