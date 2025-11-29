import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



MLP_INTERNAL_DIM = 1024 


BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10



TRAIN_SPLIT = 0.9     



import json

datasets = ['alpaca', 'gsm8k', 'math_infini', 'mt-bench', 'sum', 'vicuna-bench']
models = ['Qwen3-8B', 'Qwen3-14B']

for dataset in datasets:
    for model in models:
        spec_hidden_states_path = f'./train_data/{dataset}_{model}_spec_hidden_states.pt'
        cur_hidden_states_path = f'./train_data/{dataset}_{model}_cur_hidden_states_None_None.pt'
        last_hidden_states_path = f'./train_data/{dataset}_{model}_true_last_hidden_states_None_None.pt'
        json_path = f'./train_data/{dataset}_{model}_data_None_None.json'
        spec_hidden_states = torch.load(spec_hidden_states_path).to('cuda')
        cur_hidden_states = torch.load(cur_hidden_states_path).to('cuda')
        last_hidden_states = torch.load(last_hidden_states_path).to('cuda')
        with open(json_path, "r") as f:
            data = json.load(f)
        mask = torch.ones(last_hidden_states.shape[0], dtype=torch.bool)
        idx = 0
        for key, value in data.items():
            idx += len(value['Token'])
            mask[idx] = False
            idx += 1
        select_last_hidden_states = last_hidden_states[mask]
        spec_hidden_states = spec_hidden_states.unsqueeze(1)
        print(select_last_hidden_states.shape)
        print(cur_hidden_states.shape)
        print(spec_hidden_states.shape)
        train_data = torch.cat([select_last_hidden_states, cur_hidden_states, spec_hidden_states], dim = -1)
        torch.save(train_data, f'./train_data/{dataset}_{model}_train_data.pt')
