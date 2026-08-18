import transformers
import random
from collections import defaultdict
import pickle
import numpy as np
import torch
from functools import partial
from typing import Dict, Optional
import dill

import time

top_indices = torch.tensor(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, -1, -1, 128).to("cuda")

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
def adaptive_selection_topk(matrix, threshold_ratio=0.9, min_k=1, max_k=None):
    # matrix: numpy 2d array, shape=(rows, cols)
    topk_indices_per_col = []
    n_rows, n_cols = matrix.shape
    for col in range(n_cols):
        col_data = matrix[:, col]
        threshold = np.quantile(col_data.numpy(), threshold_ratio)
        indices = np.where(col_data >= threshold)[0]
        if len(indices) < min_k:
            indices = np.argsort(col_data)[-min_k:]
        elif max_k is not None and len(indices) > max_k:
            indices = indices[np.argsort(col_data[indices])][-max_k:]
        # breakpoint()
        head_selection = torch.tensor(sorted((indices*2).tolist()+(indices*2+1).tolist()))###########
        # head_selection = torch.tensor(sorted((indices).tolist()))
        topk_indices_per_col.append(head_selection)
    return topk_indices_per_col


def adaptive_selection_dynamic(matrix, threshold_ratio=0.9, min_k=1, max_k=None):
    # matrix: numpy 2d array, shape=(rows, cols)
    topk_indices_per_col = []
    n_rows, n_cols = matrix.shape
    
    for col in range(n_cols):
        col_data = matrix[:, col]
        # print(f"max col data:{max(col_data)}")
        if max(col_data)>0.4: # 
            # threshold = np.quantile(col_data.numpy(), threshold_ratio)
            # indices = np.where(col_data >= threshold)[0]
            # if len(indices) > min_k:
            #     indices = np.argsort(col_data)[-min_k:]
            # elif max_k is not None and len(indices) > max_k:
            #     indices = indices[np.argsort(col_data[indices])][-max_k:]
            indices = np.argsort(col_data)[-min_k:]
            head_selection = torch.tensor(sorted((indices*2).tolist()+(indices*2+1).tolist()))
            topk_indices_per_col.append(head_selection)
        else:
            indices = np.argsort(col_data)[-max_k:]
            topk_indices_per_col.append(torch.tensor(sorted((indices*2).tolist()+(indices*2+1).tolist())))

            
    return topk_indices_per_col


def constructe_adaptive_selection(model_name,budget,threshold_ratio,min_k,max_k,pad_value=0,dataset_name="qasper",device="cuda",mode="topk",evaluate_num=8):
    # budget_dict = {512:0,1024:1,2048:2,4096:3}
    budget_dict = {128:0,256:1,512:2,1024:3,2048:4,4096:5}
    frequency_group_agreement = load_pickle(f"dimension_rankings/{model_name}/{dataset_name}_agreement.pkl") # [budget_num,layer_num,frequency_group_num,head_num]
    budget_num,layer_num,frequency_group_num,head_num = frequency_group_agreement.shape
    if float(budget) < 1 or int(budget) not in list(budget_dict.keys()):
        selected_budget_agreement = frequency_group_agreement[budget_dict[1024],:,:,:]
    else:
        # selected_budget_agreement = frequency_group_agreement[budget_dict[int(budget)],:,:,:]
        selected_budget_agreement = frequency_group_agreement[budget_dict[int(budget)],:,:,:]
    

    records = defaultdict(dict)
    for layer_idx in range(layer_num):
        if mode == "topk":
            head_selections = adaptive_selection_topk(selected_budget_agreement[layer_idx],threshold_ratio=threshold_ratio,min_k=min_k,max_k=max_k)
        else:
            head_selections = adaptive_selection_dynamic(selected_budget_agreement[layer_idx],threshold_ratio=threshold_ratio,min_k=min_k,max_k=max_k)
        layer_max_len = max(len(sel) for sel in head_selections)
    
        padded_selections = torch.full((len(head_selections), layer_max_len), 
                                fill_value=pad_value,  # 或其他填充值
                                dtype=head_selections[0].dtype)

        attention_masks = torch.full((len(head_selections), layer_max_len), 
                                    fill_value=False, 
                                    dtype=torch.bool)
        for head_idx,selection in enumerate(head_selections):
            # 填充到max_len
            padded_selections[head_idx, :len(selection)] = selection
            attention_masks[head_idx, :len(selection)] = True
        records[layer_idx]["head_selection"]=padded_selections.to(device)
        records[layer_idx]["selection_masks"]=attention_masks.to(device)
    
    return records


def core_module_with_padding_old(query_states, key_states, value_states, layer_idx, budget,records): #the correct version
    
    bs, num_heads, query_len, head_dim = query_states.shape
    _, _, seq_len, _ = key_states.shape
    preceding_token_num = int(seq_len * budget) if budget < 1 else int(budget)
    head_selections = records[layer_idx]["head_selection"].to(query_states.device)
    selection_masks = records[layer_idx]["selection_masks"].to(query_states.device)
    
    
    # 批量gather - 注意这里需要expand到正确的维度
    # print(f"head_selections:{head_selections.shape}")
    selection_expanded = head_selections.unsqueeze(0).unsqueeze(2).expand(bs, -1, query_len, -1) # [bs, num_heads, query_len, head_dim]
    query_selected = torch.gather(query_states, dim=-1, index=selection_expanded)
    
    # 对key也做同样操作
    key_selection_expanded = head_selections.unsqueeze(0).unsqueeze(2).expand(bs, -1, seq_len, -1)
    key_selected = torch.gather(key_states, dim=-1, index=key_selection_expanded)
    
    # 应用mask：将填充位置的值设为0
    selection_masks_expanded_query = selection_masks.unsqueeze(0).unsqueeze(2).expand(bs, -1, query_len, -1)
    selection_masks_expanded_key = selection_masks.unsqueeze(0).unsqueeze(2).expand(bs, -1, seq_len, -1)
    # 修正mask逻辑
    query_selected = torch.where(selection_masks_expanded_query, query_selected, 0)
    key_selected = torch.where(selection_masks_expanded_key, key_selected, 0)
    
    # 批量计算attention
    attn_weights = torch.matmul(query_selected, key_selected.transpose(2,3))
    
    # 选择top-k tokens
    if seq_len<preceding_token_num:
        preceding_token_num = seq_len
    top_indices = torch.topk(attn_weights, k=int(preceding_token_num), dim=-1).indices.transpose(2,3).expand(-1, -1, -1, head_dim)
    del attn_weights,selection_expanded,query_selected,key_selected,key_selection_expanded,selection_masks_expanded_query,selection_masks_expanded_key
    # 批量gather原始key/value
    key_output = torch.gather(key_states, dim=2, index=top_indices)
    value_output = torch.gather(value_states, dim=2, index=top_indices)
    
    return key_output, value_output           

# def core_module_with_padding_old(query_states, key_states, value_states, layer_idx, budget,records): #the correct version
    
#     bs, num_heads, query_len, head_dim = query_states.shape
#     _, _, seq_len, _ = key_states.shape
#     preceding_token_num = int(seq_len * budget) if budget < 1 else int(budget)
#     head_selections = records[layer_idx]["head_selection"]
#     selection_masks = records[layer_idx]["selection_masks"]
    
    
#     # 批量gather - 注意这里需要expand到正确的维度
#     # print(f"head_selections:{head_selections.shape}")
#     selection_expanded = head_selections.unsqueeze(0).unsqueeze(2).expand(bs, -1, query_len, -1) # [bs, num_heads, query_len, head_dim]
#     query_selected = torch.gather(query_states, dim=-1, index=selection_expanded)
    
#     # 对key也做同样操作
#     key_selection_expanded = head_selections.unsqueeze(0).unsqueeze(2).expand(bs, -1, seq_len, -1)
#     key_selected = torch.gather(key_states, dim=-1, index=key_selection_expanded)
    
#     # 应用mask：将填充位置的值设为0
#     # selection_masks_expanded_query = selection_masks.unsqueeze(0).unsqueeze(2).expand(bs, -1, query_len, -1)
#     # selection_masks_expanded_key = selection_masks.unsqueeze(0).unsqueeze(2).expand(bs, -1, seq_len, -1)
#     # # 修正mask逻辑
#     # query_selected = torch.where(selection_masks_expanded_query, query_selected, 0)
#     # key_selected = torch.where(selection_masks_expanded_key, key_selected, 0)
    
#     # 批量计算attention
#     attn_weights = torch.matmul(query_selected, key_selected.transpose(2,3))
    
#     # 选择top-k tokens
#     if seq_len<preceding_token_num:
#         preceding_token_num = seq_len

#     # top_indices = torch.topk(attn_weights, k=int(preceding_token_num), dim=-1).indices.transpose(2,3).expand(-1, -1, -1, head_dim)


#     # top_indices = torch.tensor(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, -1, -1, head_dim).to(attn_weights.device)
#     # del attn_weights,selection_expanded,query_selected,key_selected,key_selection_expanded

#     key_output = torch.gather(key_states, dim=2, index=top_indices)
#     value_output = torch.gather(value_states, dim=2, index=top_indices)


    
#     return key_output, value_output    


def core_module_with_padding(query_states, key_states, layer_idx, budget,records):
    

    bs, num_heads, query_len, head_dim = query_states.shape
    _, key_num_heads, seq_len, _ = key_states.shape
    preceding_token_num = int(seq_len * budget) if budget < 1 else int(budget)
    head_selections = records[layer_idx]["head_selection"].to(query_states.device)
    selection_masks = records[layer_idx]["selection_masks"].to(query_states.device)

    # 1 & 2. 直接创建mask并填充
    mask = torch.zeros(num_heads, head_dim, dtype=torch.bool, device=query_states.device)
    mask.scatter_(1, head_selections, True)
    
    # 3 & 4. 获取未选择的索引
    full_idx = torch.arange(head_dim, device=query_states.device).expand(num_heads, -1)
    head_unselecteds = full_idx[~mask].view(num_heads, head_dim - head_selections.size(1))

    # 使用expand而不是unsqueeze+expand
    selection_expanded = head_selections.unsqueeze(0).unsqueeze(2).expand(bs, num_heads, seq_len, -1)
    unselected_expanded = head_unselecteds.unsqueeze(0).unsqueeze(2).expand(bs, num_heads, seq_len, -1)

    # 合并query和key的处理
    def gather_selected_and_unselected(states, sel_exp, unsel_exp):
        selected = torch.gather(states, dim=-1, index=sel_exp)
        unselected = torch.gather(states, dim=-1, index=unsel_exp)
        return selected, unselected

    query_selected, query_unselected = gather_selected_and_unselected(query_states, selection_expanded[..., :query_len,:], unselected_expanded[..., :query_len,:])
    key_selected, key_unselected = gather_selected_and_unselected(key_states, 
                                                                selection_expanded, 
                                                                unselected_expanded)

    # 移除不必要的CPU传输
    # query_unselected = query_unselected.detach()
    # key_unselected = key_unselected.detach()

    
    # 应用mask：将填充位置的值设为0 因为每个head选择的都是相同个数的dim nums所以暂时不需要mask填充
    # selection_masks_expanded_query = selection_masks.unsqueeze(0).unsqueeze(2).expand(bs, -1, query_len, -1)
    # selection_masks_expanded_key = selection_masks.unsqueeze(0).unsqueeze(2).expand(bs, -1, seq_len, -1)
    # 修正mask逻辑
    # query_selected = torch.where(selection_masks_expanded_query, query_selected, 0)
    # key_selected = torch.where(selection_masks_expanded_key, key_selected, 0)
    
    
    
    return query_selected,key_selected,key_unselected,  selection_expanded,unselected_expanded      
