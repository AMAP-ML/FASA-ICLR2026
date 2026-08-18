
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import transformers


def init_streamingllm(self):
    if not hasattr(self,"kv_cluster"):
        if not hasattr(self.config,"start_size"):
            self.config.start_size = 4
        if not hasattr(self.config,"max_capacity_prompt"):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config,"recent_size"):
            self.config.recent_size = 2044
    self.kv_cluster = StreamingKVCluster(
        start_size = self.config.start_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        recent_size = self.config.recent_size,
    )
class StreamingKVCluster():
    def __init__(self, start_size, max_capacity_prompt, recent_size):
        self.start_size = start_size
        self.max_capacity_prompt = max_capacity_prompt
        self.recent_size = recent_size
        assert self.start_size + self.recent_size == self.max_capacity_prompt
    def update_kv(self,key_states,query_states,value_states,past_key_values,layer_idx):
        bs,q_heads,query_len,head_dim = query_states.shape
        bs,k_heads,seq_len,head_dim = key_states.shape
        current_len = query_len
        past_len = past_key_values.get_seq_length(layer_idx)

        if current_len + past_len <= self.max_capacity_prompt:
            pass
        else:
            if query_len!=1:
                key_states = torch.cat([key_states[:,:,:self.start_size,:], key_states[:,:,-self.recent_size:,:]],dim=2)
                value_states = torch.cat([value_states[:,:,:self.start_size,:], value_states[:,:,-self.recent_size:,:]],dim=2)
            else:
                past_key_states = past_key_values.key_cache[layer_idx]
                past_value_states = past_key_values.value_cache[layer_idx]

                cropped_past_key_states = torch.cat([past_key_states[:,:,:self.start_size,:], past_key_states[:,:,-self.recent_size+1:,:]],dim=2)
                cropped_past_value_states = torch.cat([past_value_states[:,:,:self.start_size,:], past_value_states[:,:,-self.recent_size+1:,:]],dim=2)
                past_key_values.key_cache[layer_idx] = cropped_past_key_states
                past_key_values.value_cache[layer_idx] = cropped_past_value_states
        key_states,value_states = past_key_values.update(key_states,value_states,layer_idx)
        # print(f"query len:{query_len}, seq_len:{seq_len},key_states_size:{key_states.shape[2]}")
        assert key_states.shape[2] <= self.max_capacity_prompt
        return key_states,value_states,past_key_values




                

