from transformers import AutoTokenizer, AutoModelForCausalLM
from model.EAGLE_model import Model as SpecModel
import torch
from typing import Optional, Union
import json

import torch.nn as nn


def gen_train_data(model_name,dataset, ori_model, spec_model):

    

    
    last_hidden_states_path =f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/router/{dataset}_{model_name}_last_hidden_states_None_None.pt"
    last_hidden_states = torch.load(last_hidden_states_path).to(ori_model.device)
    # print(last_hidden_states.shape)
    json_path = f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/router/{dataset}_{model_name}_data_None_None.json"
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    
    
    N_LAYERS = ori_model.config.num_hidden_layers  
    HIDDEN_DIM = ori_model.config.hidden_size
    
    
    
    
    
    start_idx = 0
    
    
    train_X = []
    
    train_Y = []
    
    for id, json_data in data.items():
        output = json_data["output"]
        tot_len = len(output)
        prompt_len = len(json_data['Prompt'])
        spec_model.reset_kv()
        cur_hidden_states = last_hidden_states[:,start_idx:start_idx+tot_len-1]
        # print(cur_hidden_states.shape)
        input_ids = torch.tensor([output]).to(ori_model.device)
        spec_hidden_states = spec_model.topK_generate(hidden_states=cur_hidden_states,
                                                      input_ids = input_ids)
        
        spec_hidden_states = spec_hidden_states[:, -(tot_len-prompt_len-1):,:].cpu()
        label = torch.zeros(spec_hidden_states.shape[1], N_LAYERS).cpu()
        for idx, token_info in enumerate(json_data['Token']):
            label[idx, token_info['layer_index']] = 1
        train_X.append(spec_hidden_states.squeeze(0))
        train_Y.append(label)
    
    train_X = torch.cat(train_X, dim=0)
    train_Y = torch.cat(train_Y, dim=0)
    return train_X, train_Y

class PathPredictorMLP(nn.Module):
    def __init__(self, n_layers, llm_hidden_dim, mlp_internal_dim):
        super().__init__()
        
        self.input_dim = llm_hidden_dim 
        self.output_dim = n_layers
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, mlp_internal_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_internal_dim), 
            nn.Linear(mlp_internal_dim, self.output_dim)
        )
     
    def forward(self, x):
        return self.net(x)


from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20



TRAIN_SPLIT = 0.9     
MLP_INTERNAL_DIM = 2048



def router_train(train_X, train_Y):
    
    print(f'X: {train_X.shape} {train_X.dtype}, Y: {train_Y.shape} {train_Y.dtype}')
    dataset = TensorDataset(train_X, train_Y)
    NUM_SAMPLES = train_X.shape[0]
    train_size = int(NUM_SAMPLES * TRAIN_SPLIT)
    val_size = NUM_SAMPLES - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {train_size} validation: {val_size}")
    device = torch.device("cuda")
    print(device)
    router = PathPredictorMLP(train_Y.shape[-1], train_X.shape[-1], MLP_INTERNAL_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(router.parameters(), lr=LEARNING_RATE)
    print(f"Para num {sum(p.numel() for p in router.parameters()):,}")
    print("--- Training ---")
    for epoch in range(NUM_EPOCHS):
    
        router.train()
        total_train_loss = 0
        
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            logits = router(batch_X)
            loss = criterion(logits, batch_Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
    
        router.eval()
        total_val_loss = 0
        all_preds_flat = []
        all_targets_flat = []
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                
                logits = router(batch_X)
                loss = criterion(logits, batch_Y)
                total_val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5)
                
                all_preds_flat.append(preds.cpu().numpy().flatten())
                all_targets_flat.append(batch_Y.cpu().numpy().flatten())
        avg_val_loss = total_val_loss / len(val_loader)
        y_pred = np.concatenate(all_preds_flat)
        y_true = np.concatenate(all_targets_flat)
        
        val_f1 = f1_score(y_true, y_pred, zero_division=0)
        val_acc = accuracy_score(y_true, y_pred)
        val_prec = precision_score(y_true, y_pred, zero_division=0)
        val_rec = recall_score(y_true, y_pred, zero_division=0)

        print(f"--- EPOCH {epoch+1}/{NUM_EPOCHS} summary ---")
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Average validation loss: {avg_val_loss:.4f}")
        print(f"  Validation metrics (global):")
        print(f"    Accuracy: {val_acc:.4f}")
        print(f"    F1-Score: {val_f1:.4f}")
        print(f"    Precision: {val_prec:.4f}")
        print(f"    Recall: {val_rec:.4f}")
        print("---------------------------------")
    print("Training complete.")
    return router        
        
if __name__ == "__main__":
    models = ['Qwen3-8B', 'Qwen3-14B']
    datasets = ['alpaca', 'gsm8k', 'sum', 'mt-bench','vicuna-bench', 'math_infini']
    
    model_name = models[0]
    
    train_X = []
    train_Y = []
    
    model_path = f"/inspire/hdd/global_public/public_models/Qwen/{model_name}/"
    if model_name == 'Qwen3-8B':
        Spec_model_path = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/models/qwen3_8b_eagle3"
    elif model_name == 'Qwen3-14B':
        Spec_model_path = '/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/models/Qwen3-14B_eagle3'
    
    ori_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',torch_dtype=torch.float16)
    
    
    spec_model = SpecModel.from_pretrained(Spec_model_path=Spec_model_path, Ori_model_path=model_path, dtype=torch.float16).to(ori_model.device)
    
    for dataset in datasets:
        data = gen_train_data(model_name, dataset, ori_model, spec_model)
        train_X.append(data[0])
        train_Y.append(data[1])
    
    train_X = torch.cat(train_X, dim = 0)
    train_X = train_X.float()
    train_Y = torch.cat(train_Y, dim = 0)
    train_Y = train_Y.float()
    # print(train_X.shape)
    # print(train_Y.shape)
    router = router_train(train_X, train_Y)
    model_save_path = f"./checkpoint/router/{model_name}_layer_router_{MLP_INTERNAL_DIM}.pth"
    torch.save(router.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
        

        
        
        
        
        
        
    