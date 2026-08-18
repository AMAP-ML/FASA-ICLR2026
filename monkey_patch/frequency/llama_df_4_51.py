from transformers.models.llama.modeling_llama import *
from typing import Dict, Optional
from transformers.cache_utils import Cache,DynamicCache
import torch
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from monkey_patch.frequency.utils import core_module_with_padding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv,eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
def LlamaFlashAttention2_forward_dynamic_frequency_451(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        budget: int=4096,
        records:Dict = None,
        layer_control: int=1,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # budget = kwargs.get("budget", 4096)
        # records =  kwargs.get("records", None)
        # layer_control = kwargs.get("layer_control", None)


        input_shape = hidden_states.shape[:-1]
        bsz = input_shape[0]
        hidden_shape = (*input_shape, -1, self.head_dim)
        print(f"====={hidden_shape}=========")

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        #------------------------------------------------------------------------------------------------------------
        # original kv cache update
        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        #-------------------------------------------------------------------------------------------------------------------

        #----------------------------------------------------------------------------------------------
        # dynamic kv cache update
        key_states = repeat_kv(key_states, self.num_key_value_groups)
    
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
            #-------------------------------------------------------------------------------------------------




        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights