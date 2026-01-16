import torch
import torch.nn as nn
import torch.optim as optim

from model.utils import PathPredictorMLP
from torch.utils.data import Dataset, DataLoader
class LayerData(Dataset):
    def __init__(self, layer_id = 0, model = 'Qwen3-8B'):
        # 模拟你的预处理数据
        # 实际使用时，请加载保存的 .pt 文件
        # datasets = ['alpaca', 'gsm8k', 'mt-bench', 'sum', 'vicuna-bench']
        datasets = ['alpaca', 'sum']
        
        self.inputs = []
        self.targets = []
        for dataset in datasets:
            self.inputs.append(torch.load(f'./train_data/layer_router/thinking/{dataset}_{model}_laye_router_X_idx{layer_id}_None_None.pt').squeeze(0))
            self.inputs.append(torch.load(f'./train_data/layer_router/non_thinking/{dataset}_{model}_laye_router_X_idx{layer_id}_None_None.pt').squeeze(0))
            self.targets.append(torch.load(f'./train_data/layer_router/thinking/{dataset}_{model}_laye_router_Y_idx{layer_id}_None_None.pt').view(-1,1))
            self.targets.append(torch.load(f'./train_data/layer_router/non_thinking/{dataset}_{model}_laye_router_Y_idx{layer_id}_None_None.pt').view(-1,1))
        self.inputs = torch.cat(self.inputs, dim = 0)
        self.targets = torch.cat(self.targets, dim = 0)


    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def train_one_layer_router(
    layer_id, 
    hidden_dim=4096, 
    router_dim=64, 
    batch_size=256, 
    lr=1e-3, 
    epochs=10, 
    device='cuda',
):
    print(f"--- Training Layer router for Layer {layer_id} ---")
    
    layer_router = PathPredictorMLP(n_layers=1, llm_hidden_dim=hidden_dim, mlp_internal_dim=router_dim).to(device)
    dataset = LayerData(layer_id=layer_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(layer_router.parameters(), lr=lr)
    layer_router.train()

    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for x, y_true in dataloader:
            x = x.to(device)    # [Batch, Dim]
            x = x.float()
            y_true = y_true.to(device)  # [Batch, Dim]
            y_true = y_true.float()
            optimizer.zero_grad()
            
            logits = layer_router(x)
            loss = criterion(logits, y_true)
            optimizer.zero_grad()
            loss.backward()       # 计算梯度
            optimizer.step()  
            steps += 1
            total_loss += loss.item()

        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")
    print(f"Layer {layer_id} Layer router trained.")
    return layer_router

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", "-d", type=int, default=32)
    parser.add_argument("--layer_id","-l", type=int, default=0)
    args = parser.parse_args()
    layer_id = args.layer_id
    trained_adapter = train_one_layer_router(layer_id, epochs=100, lr=1e-3, router_dim=args.dim)
    
    torch.save(trained_adapter.state_dict(), f"./checkpoint/layer_router/{args.dim}/router_layer_{layer_id}_{args.dim}.pt")