import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.EAGLE_model import Model as SpecModel
import torch
from typing import Optional, Union
import json
import torch.nn as nn
from model.utils import Global_router, PathPredictorMLP
import os


class Spec_Data(Dataset):
    def __init__(self, spec_model, ori_model):
        datasets = ['alpaca', 'gsm8k', 'mt-bench', 'sum', 'vicuna-bench']
        
        self.inputs = []
        self.labels = []
        with torch.no_grad():
            for dataset in datasets:
                file_path = f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/global_router/0.97/non_thinking/{dataset}_Qwen3-8B_last_hidden_states_None_None.pt"
                if not os.path.exists(file_path):
                    continue
                last_hidden_states = torch.load(file_path)
                with open(f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/global_router/0.97/non_thinking/{dataset}_Qwen3-8B_normal_info_None_None.json", "r") as f:
                    json_data = json.load(f)
                    start_ptr = 0
                    for id, meta_json_data in json_data.items():
                        len_O = len(meta_json_data['output'])
                        len_P = len(meta_json_data['Prompt'])
                        len_T = len(meta_json_data['Token'])
                        input_ids = torch.tensor(meta_json_data['output']).view(1,-1).to(ori_model.device)
                        hidden_states = last_hidden_states[:,start_ptr:start_ptr+len_P+len_T,:].to(ori_model.device)
                        start_ptr += len_P+len_T
                        # print(hidden_states.shape)
                        # print(input_ids.shape)
                        spec_model.reset_kv()
                        spec_hidden_states = spec_model.topK_generate(hidden_states=hidden_states, input_ids = input_ids)
                        spec_hidden_states = spec_hidden_states[0].detach().to('cpu')
                        cur_hidden_states = ori_model.model.embed_tokens(torch.tensor(meta_json_data['output'][1:], device=ori_model.device))
                        cur_hidden_states = cur_hidden_states.detach().to('cpu')
                        # print(cur_hidden_states.shape)
                        # print(spec_hidden_states.shape)
                        input_data = torch.cat([cur_hidden_states, spec_hidden_states], dim = -1)[len_P-1:len_P-1+len_T]
                        self.inputs.append(input_data.to('cpu'))
                        labels = torch.zeros(len_T, ori_model.config.num_hidden_layers).to('cpu')
                        for idx, token_info in enumerate(meta_json_data['Token']):
                            labels[idx][token_info['layer_index']] = 1.0
                        self.labels.append(labels)
                # file_path = f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/global_router/0.94/thinking/{dataset}_Qwen3-8B_last_hidden_states_None_None.pt"
                # if not os.path.exists(file_path):
                #     continue
                # last_hidden_states = torch.load(file_path)
                # with open(f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/global_router/0.94/thinking/{dataset}_Qwen3-8B_normal_info_None_None.json", "r") as f:
                #     json_data = json.load(f)
                #     start_ptr = 0
                #     for id, meta_json_data in json_data.items():
                #         len_O = len(meta_json_data['output'])
                #         len_P = len(meta_json_data['Prompt'])
                #         len_T = len(meta_json_data['Token'])
                #         input_ids = torch.tensor(meta_json_data['output']).view(1,-1).to(ori_model.device)
                #         hidden_states = last_hidden_states[:,start_ptr:start_ptr+len_P+len_T,:].to(ori_model.device)
                #         start_ptr += len_P+len_T
                #         # print(hidden_states.shape)
                #         # print(input_ids.shape)
                #         spec_model.reset_kv()
                #         spec_hidden_states = spec_model.topK_generate(hidden_states=hidden_states, input_ids = input_ids)
                #         spec_hidden_states = spec_hidden_states[0].detach().to('cpu')
                #         cur_hidden_states = ori_model.model.embed_tokens(torch.tensor(meta_json_data['output'][1:], device=ori_model.device))
                #         cur_hidden_states = cur_hidden_states.detach().to('cpu')
                #         # print(cur_hidden_states.shape)
                #         # print(spec_hidden_states.shape)
                #         input_data = torch.cat([cur_hidden_states, spec_hidden_states], dim = -1)[len_P-1:len_P-1+len_T]
                #         self.inputs.append(input_data.to('cpu'))
                #         labels = torch.zeros(len_T, ori_model.config.num_hidden_layers).to('cpu')
                #         for idx, token_info in enumerate(meta_json_data['Token']):
                #             labels[idx][token_info['layer_index']] = 1.0
                #         self.labels.append(labels)

        self.inputs = torch.cat(self.inputs, dim = 0).to('cpu')
        self.labels = torch.cat(self.labels, dim = 0).to('cpu')
        exec_layers = self.labels.sum(dim=-1)
        print(exec_layers.mean())
        num_pos = (self.labels == 1).sum()
        
        num_neg = (self.labels == 0).sum()
        print('num_pos: ', num_pos)
        print('num_neg: ', num_neg)
        
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]      


def train_global_router(
    dataset,
    input_dim,
    output_dim,
    router_dim=64, 
    batch_size=256, 
    lr=1e-3, 
    epochs=10, 
    model_type = 1,
    device='cuda',
):
    print("Training Global Router")
    if model_type == 1:
        router = Global_router(input_dim=input_dim, hidden_dim=router_dim,output_dim=output_dim).to(device)
    elif model_type == 2:
        router = PathPredictorMLP(n_layers=output_dim, llm_hidden_dim=input_dim, mlp_internal_dim=router_dim).to(device)
    num_pos = (dataset.labels == 1).sum()
    num_neg = (dataset.labels == 0).sum()    
    pos_weight = num_neg / num_pos
    # print(pos_weight)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr)
    
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device)) 
    criterion = nn.BCEWithLogitsLoss()
    router.train()
    
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for x, y_true in dataloader:
            x = x.to(device)    # [Batch, Dim]
            x = x.float()
            y_true = y_true.to(device)  # [Batch, Dim]
            y_true = y_true.float()
            optimizer.zero_grad()
            
            logits = router(x)
            loss = criterion(logits, y_true)
            optimizer.zero_grad()
            loss.backward()       # 计算梯度
            optimizer.step()  
            steps += 1
            total_loss += loss.item()

        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")
    return router



if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", "-d", type=int, default=32)
    parser.add_argument("--type", '-t', type=int, default=1)
    args = parser.parse_args()
    Ori_model_path = f"/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B/"
    Spec_model_path = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/models/qwen3_8b_eagle3"
    ori_model = AutoModelForCausalLM.from_pretrained(Ori_model_path, device_map='auto', torch_dtype=torch.float16)
    spec_model = SpecModel.from_pretrained(Spec_model_path=Spec_model_path, Ori_model_path=Ori_model_path, dtype=torch.float16).to(ori_model.device)
    data = Spec_Data(spec_model, ori_model)
    print(data.inputs.shape)
    print(data.labels.shape)
    router = train_global_router(dataset=data, input_dim=data.inputs.shape[-1], output_dim = data.labels.shape[-1], router_dim = args.dim, epochs=100, lr=1e-3, model_type=args.type)
    torch.save(router.state_dict(), f"./checkpoint/global_router/global_router_{args.dim}_Model{args.type}_non_thinking.pt")
