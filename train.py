import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

from model.qwen3_model_dev import Spec_Qwen3ForCausalLM

model = Spec_Qwen3ForCausalLM.from_pretrained(f"/share/others/public_models/Qwen3-8B/")



N_LAYERS = model.config.num_hidden_layers  
HIDDEN_DIM = model.config.hidden_size


MLP_INTERNAL_DIM = 1024 


BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10



TRAIN_SPLIT = 0.9     



import json

with open("/home/xujiaming/xujiaming/Paper/SpecMoD/data/alpaca_Qwen3-8B_data.json", "r") as f:
    data = json.load(f)
    lst = []
    ground_truth = []
    for q_id, q_re in data.items():
        for token_id, token_value in q_re['Token'].items():
            lst.append(eval(token_id))
            ground_truth.append(token_value['layer_index'])

NUM_SAMPLES = len(ground_truth)


X_data = model.model.embed_tokens(torch.tensor(lst)).detach()

Y_data = torch.zeros((len(ground_truth), N_LAYERS))

for i, layer_idx in enumerate(ground_truth):
    Y_data[i, layer_idx] = 1.0


print(f"X: {X_data.shape}, Y: {Y_data.shape}")


dataset = TensorDataset(X_data, Y_data)

train_size = int(NUM_SAMPLES * TRAIN_SPLIT)
val_size = NUM_SAMPLES - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Train: {train_size} validation:{val_size}")


device = torch.device("cuda")
print(device)

model = PathPredictorMLP(N_LAYERS, HIDDEN_DIM, MLP_INTERNAL_DIM).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Para num {sum(p.numel() for p in model.parameters()):,}")


print("\n--- Training ---")

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

model_save_path = "path_predictor_mlp_baseline.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")