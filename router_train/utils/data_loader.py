# coding=utf-8
"""
数据加载器 - 支持预先计算好的hidden states
"""
import json
import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset


class PrecomputedHiddenStatesDataset(Dataset):
    """
    加载预先计算好的hidden states数据集
    
    数据格式:
    1. JSON文件:
    {
        "1": {
            "prompt": "对话prompt",
            "Token": [[token_id, [layer_ids]], ...],
            "output": "完整输出(Prompt+Output)"
        },
        ...
    }
    
    2. PT文件:
    {
        "1": tensor([seq_len, 3*hidden_size]),  # 三层hidden states已拼接
        ...
    }
    """
    
    def __init__(
        self,
        path_dir: str,
        num_layers: int = 36,
        max_len: int = 2048,
        start_id: int = 0,
        end_id: int =160,
    ):
        self.num_layers = num_layers
        self.max_len = max_len
        self.json_data = {}
        self.hidden_states_data = {}
        self.start_idx = start_id
        self.end_idx = end_id
        # 加载JSON
        print(f"Loading JSON and hidden_states from {path_dir}...")
        for i in range(start_id, end_id):
            start_idx = i*100
            end_idx = start_idx + 100
            json_path = f"{path_dir}/sharegpt_common_en_Qwen3-8B_normal_info_{start_idx}_{end_idx}.json"
            pt_path = f"{path_dir}/sharegpt_common_en_Qwen3-8B_last_hidden_states_{start_idx}_{end_idx}.pt"
            last_hidden_states = torch.load(pt_path, map_location='cpu', mmap=True)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                start_ptr = 0
                self.json_data = self.json_data | data
                for id, meta_json_data in data.items():
                    len_O = len(meta_json_data['output'])
                    len_P = len(meta_json_data['Prompt'])
                    len_T = len(meta_json_data['Token'])
                    assert len_P + len_T == len_O - 1, \
                        f"Warning: Question {id} length mismatch! "f"len_P:{len_P}, len_T:{len_T}, len_O:{len_O}"
                    hidden_states = last_hidden_states[:,start_ptr:start_ptr+len_P+len_T,:].cpu()
                    start_ptr += len_P+len_T
                    self.hidden_states_data[id] = hidden_states
        
        print(f"Loaded {len(self.json_data)} samples")
    
    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, idx):
        
        # 获取JSON数据
        idx = self.start_idx*100+idx
        sample = self.json_data[str(idx)]
        hidden_states = self.hidden_states_data[str(idx)][0,:-1,:]
        prompt_ids = sample["Prompt"]
        gen_ids = sample["Token"]  # [[token_id, [layer_ids]], ...]
        output_ids = sample["output"]
        
        len_P = len(prompt_ids)
        len_T = len(gen_ids)
        len_O = len(output_ids)
        
        
        # 构建layer_masks
        # 前prompt_len个位置没有layer标注,后token_len个位置有标注
        seq_len = len_P + len_T - 1
        layer_masks = torch.zeros(seq_len, self.num_layers, dtype=torch.float32)
        
        # 对于prompt位置,默认执行所有层(因为没有标注)
        layer_masks[:len_P-1, :] = 1.0
        
        # 对于token位置,根据标注设置
        for idx, token_info in enumerate(gen_ids):
            layer_masks[len_P-1+idx][token_info['layer_index']] = 1.0
            
        # 构建loss_mask: 只在有标注的token位置计算loss
        loss_mask = torch.zeros(seq_len, dtype=torch.float32)
        loss_mask[len_P-1:] = 1.0  # 只在生成的token上计算loss
        
        # input_ids: 用于获取embedding
        # 使用output的前seq_len个token (因为最后一个token没有下一步)
        input_ids = torch.tensor(output_ids[1:len_P+len_T], dtype=torch.long)
        
        # attention_mask
        attention_mask = torch.ones(seq_len, dtype=torch.float32)
        
        assert input_ids.shape[0] == hidden_states.shape[0], \
            f"Input_ids shape({input_ids.shape[0]}) dismatch with hidden states({hidden_states.shape[0]}) shape"
        
        return {
            "input_ids": input_ids,  # [seq_len]
            "hidden_states": hidden_states,  # [seq_len, 3*hidden_size]
            "layer_masks": layer_masks,  # [seq_len, num_layers]
            "loss_mask": loss_mask,  # [seq_len]
            "attention_mask": attention_mask,  # [seq_len]
        }


class PrecomputedDataCollator:
    """数据整理器,支持hidden_states的padding"""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[0] for item in features)
        
        batch_size = len(features)
        hidden_size = features[0]['hidden_states'].shape[1]
        num_layers = features[0]['layer_masks'].shape[1]
        
        # 初始化batch tensors
        batch_input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        batch_hidden_states = torch.zeros(batch_size, max_length, hidden_size, dtype=torch.float16)
        batch_layer_masks = torch.zeros(batch_size, max_length, num_layers, dtype=torch.float16)
        batch_loss_mask = torch.zeros(batch_size, max_length, dtype=torch.float16)
        batch_attention_mask = torch.zeros(batch_size, max_length, dtype=torch.float16)
        
        # 填充数据
        for i, item in enumerate(features):
            seq_len = item['input_ids'].shape[0]
            batch_input_ids[i, :seq_len] = item['input_ids']
            batch_hidden_states[i, :seq_len] = item['hidden_states']
            batch_layer_masks[i, :seq_len] = item['layer_masks']
            batch_loss_mask[i, :seq_len] = item['loss_mask']
            batch_attention_mask[i, :seq_len] = item['attention_mask']
        
        return {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "layer_masks": batch_layer_masks,
            "loss_mask": batch_loss_mask,
            "attention_mask": batch_attention_mask,
        }
