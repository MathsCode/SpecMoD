import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

import numpy as np


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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


BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10



TRAIN_SPLIT = 0.9     
MLP_INTERNAL_DIM = 1024 


models = ['Qwen3-8B', 'Qwen3-14B']
datasets = ['alpaca', 'gsm8k', 'sum', 'mt-bench','vicuna-bench', 'math_infini']

for model_name in models:
    all_train_data = []
    all_train_label = []
    for dataset in datasets:
        train_data_path = f'./train_data/{dataset}_{model_name}_train_data.pt'
        train_label_path = f'./train_data/{dataset}_{model_name}_label_data.pt'
        if os.path.exists(train_data_path):
            train_data = torch.load(train_data_path).to(torch.float32)
            train_label = torch.load(train_label_path).to(torch.float32)
            # train_data = torch.cat([train_data[:,:,:train_data.shape[-1]//3],train_data[:,:,train_data.shape[-1]//3*2:]], dim=-1)
            all_train_data.append(train_data)
            all_train_label.append(train_label)
    all_train_data = torch.cat(all_train_data, dim=0).cpu()
    all_train_label = torch.cat(all_train_label, dim=0).cpu()
    all_train_label = all_train_label.unsqueeze(1)
    print(f'X: {all_train_data.shape}, Y: {all_train_label.shape}')
    dataset = TensorDataset(all_train_data, all_train_label)
    NUM_SAMPLES = all_train_data.shape[0]
    train_size = int(NUM_SAMPLES * TRAIN_SPLIT)
    val_size = NUM_SAMPLES - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {train_size} validation: {val_size}")
    device = torch.device("cuda")
    print(device)
    model = PathPredictorMLP(all_train_label.shape[-1], all_train_data.shape[-1], MLP_INTERNAL_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Para num {sum(p.numel() for p in model.parameters()):,}")
    print("--- Training ---")
    for epoch in range(NUM_EPOCHS):
    
        model.train()
        total_train_loss = 0
        
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            logits = model(batch_X)
            loss = criterion(logits, batch_Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        all_preds_flat = []
        all_targets_flat = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                
                logits = model(batch_X)
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
    model_save_path = f"{model_name}_layer_router_{MLP_INTERNAL_DIM}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
