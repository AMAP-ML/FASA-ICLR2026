import transformers
import random
from collections import defaultdict
import pickle
import numpy as np
import torch
from functools import partial
from typing import Dict, Optional
from monkey_patch.frequency.llama_df import LlamaFlashAttention2_forward_dynamic_frequency  
from monkey_patch.frequency.mistral_df import MistralFlashAttention2_forward_dynamic_frequency
from monkey_patch.frequency.qwen_df import Qwen2FlashAttention2_forward_dynamic_frequency
# from monkey_patch.frequency.llama_df_4_51 import LlamaFlashAttention2_forward_dynamic_frequency_451

from monkey_patch.oracle.llama_oracle_static import LLAMA_ATTENTION_CLASSES_flash
from monkey_patch.oracle.mistral_oracle_static import MISTRAL_ATTENTION_CLASSES_flash
from monkey_patch.oracle.qwen_oracle_static import QWEN2_ATTENTION_CLASSES_flash

import sys
from monkey_patch.frequency.cache_design import DynamicCache as DynamicCache_dynamic
import transformers



def replace_llama_df(budget,records,layer_control):
    # global records
    # records = constructe_adaptive_selection(model_name,budget,threshold_ratio,min_k,max_k)

    def new_forward(self, *args, **kwargs):
        return LlamaFlashAttention2_forward_dynamic_frequency(
            self, *args, budget=budget, records=records,layer_control=layer_control, **kwargs
        )
    transformers.generation.utils.DynamicCache = DynamicCache_dynamic
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = new_forward
    transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES = LLAMA_ATTENTION_CLASSES_flash

# def replace_llama_df(budget,records,layer_control):
#     # global records
#     # records = constructe_adaptive_selection(model_name,budget,threshold_ratio,min_k,max_k)

#     def new_forward(self, *args, **kwargs):
#         return LlamaFlashAttention2_forward_dynamic_frequency_451(
#             self, *args, budget=budget, records=records,layer_control=layer_control, **kwargs
#         )
#     transformers.generation.utils.DynamicCache = DynamicCache_dynamic
#     transformers.models.llama.modeling_llama.LlamaAttention.forward = new_forward


def replace_mistral_df(budget,records):
    # global records
    # records = constructe_adaptive_selection(model_name,budget,threshold_ratio,min_k,max_k)


    def new_forward(self, *args, **kwargs):
        return MistralFlashAttention2_forward_dynamic_frequency(
            self, *args, budget=budget,records=records, **kwargs
        )
    
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = new_forward
    transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES= MISTRAL_ATTENTION_CLASSES_flash


def replace_qwen_df(budget,records):
    # global records
    # records = constructe_adaptive_selection(model_name,budget,threshold_ratio,min_k,max_k)
    def new_forward(self, *args, **kwargs):
        return Qwen2FlashAttention2_forward_dynamic_frequency(
            self, *args, budget=budget, records=records,**kwargs
        )
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = new_forward
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES = QWEN2_ATTENTION_CLASSES_flash