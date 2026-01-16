import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model.utils import ShadowAdapter3
class ShadowAdapter1(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=64):
        """
        Args:
            hidden_dim: 原模型 Hidden Size (e.g., Llama-3-8B is 4096)
            bottleneck_dim: 压缩后的维度 (e.g., 64 or 128), 越小越快
        """
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim)
        # 下投影：把维度压下去
        self.gate_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        # 激活函数：保持和 Llama 一致 (SiLU)
        self.act = nn.SiLU()
        # 上投影：把维度升回来
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        
        # 【关键技巧】零初始化 (Zero Initialization)
        # 让 up_proj 的权重一开始全是 0。
        # 这样初始时: Adapter(x) = 0, Output = x + 0 = x (完美透传)
        # 训练开始后，它会慢慢学到非 0 的修正值。
        
        nn.init.zeros_(self.up_proj.weight)
        
        # down_proj 正常随机初始化即可
        nn.init.kaiming_normal_(self.down_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.gate_proj.weight, nonlinearity='linear')
    def forward(self, x):
        # 注意：这里我们只计算 Delta (残差部分)
        # 最终输出应该是 x + adapter(x)，在训练循环里加
        x_norm = self.norm(x)
        
        # 2. SwiGLU
        gate = F.silu(self.gate_proj(x_norm))
        down = self.down_proj(x_norm)
        fused = gate * down
        output = self.up_proj(fused)
        return output

class ShadowAdapter2(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=64):
        """
        Args:
            hidden_dim: 原模型 Hidden Size (e.g., Llama-3-8B is 4096)
            bottleneck_dim: 压缩后的维度 (e.g., 64 or 128), 越小越快
        """
        super().__init__()
        # 下投影：把维度压下去
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        # 激活函数：保持和 Llama 一致 (SiLU)
        self.act = nn.SiLU()
        # 上投影：把维度升回来
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        
        # 【关键技巧】零初始化 (Zero Initialization)
        # 让 up_proj 的权重一开始全是 0。
        # 这样初始时: Adapter(x) = 0, Output = x + 0 = x (完美透传)
        # 训练开始后，它会慢慢学到非 0 的修正值。
        
        nn.init.zeros_(self.up_proj.weight)
        
        # down_proj 正常随机初始化即可
        nn.init.kaiming_normal_(self.down_proj.weight, nonlinearity='linear')

    def forward(self, x):
        # 注意：这里我们只计算 Delta (残差部分)
        # 最终输出应该是 x + adapter(x)，在训练循环里加
        return self.up_proj(self.act(self.down_proj(x)))
    
class LayerData(Dataset):
    def __init__(self, layer_id = 0, model = 'Qwen3-8B'):
        # 模拟你的预处理数据
        # 实际使用时，请加载保存的 .pt 文件
        datasets = ['alpaca', 'gsm8k', 'math_infini', 'mt-bench', 'sum', 'vicuna-bench']
        self.inputs = []
        self.targets = []
        for dataset in datasets:
            self.inputs.append(torch.load(f'./train_data/adaptor/{args.threshold}/{dataset}_{model}_X_idx{layer_id}_None_None.pt').squeeze(1).to('cpu'))
            self.targets.append(torch.load(f'./train_data/adaptor/{args.threshold}/{dataset}_{model}_Y_idx{layer_id}_None_None.pt').squeeze(1).to('cpu'))
        
        self.inputs = torch.cat(self.inputs, dim = 0)
        self.targets = torch.cat(self.targets, dim = 0)


    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
        

def train_one_layer_adapter(
    layer_id, 
    hidden_dim=4096, 
    adapter_dim=64, 
    batch_size=256, 
    lr=1e-3, 
    epochs=10, 
    device='cuda',
    model_type = 1,
):
    print(f"--- Training Adapter for Layer {layer_id} ---")
    
    hidden_dim = 4096
    if model_type == 1:
        adapter = ShadowAdapter1(hidden_dim, bottleneck_dim=adapter_dim).to(device)
    elif model_type == 2:
        adapter = ShadowAdapter2(hidden_dim, bottleneck_dim=adapter_dim).to(device)
    elif model_type == 3:
        adapter = ShadowAdapter3(hidden_dim, bottleneck_dim=adapter_dim).to(device)
    dataset = LayerData(layer_id=layer_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    mse_loss_fn = nn.MSELoss()
    cos_loss_fn = nn.CosineEmbeddingLoss()

    adapter.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_cos_identity = 0
        total_mse_identity = 0
        total_mse_adaptor = 0
        total_cos_adaptor = 0
        steps = 0
        for x, y_true in dataloader:
            x = x.to(device)    # [Batch, Dim]
            x = x.float()
            y_true = y_true.to(device)  # [Batch, Dim]
            y_true = y_true.float()
            

            optimizer.zero_grad()
            
            delta = adapter(x)
            y_pred = x + delta
            
            # --- 计算 Loss (只针对 Mask=1 的样本) ---
            
            # Loss 1: MSE (让数值接近)
            # 形状: [Batch, Dim] -> mean over dim -> [Batch]
            loss_mse = mse_loss_fn(y_pred, y_true)
            
            # Loss 2: Cosine (让向量方向一致)
            # target 设为 1，表示希望相似度为 1
            target_ones = torch.ones(x.size(0)).to(device)
            loss_cos = cos_loss_fn(y_pred, y_true, target_ones) # Output: [Batch]

            loss = loss_mse + loss_cos
            
            mse_identity = F.mse_loss(x,y_true)
            cos_identity = F.cosine_similarity(x, y_true).mean().item()
            mse_adaptor = F.mse_loss(y_pred, y_true)
            cos_adaptor = F.cosine_similarity(y_pred, y_true, dim=-1).mean().item()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            steps += 1
            total_loss += loss.item()
            total_cos_identity += cos_identity
            total_mse_identity += mse_identity
            total_mse_adaptor += mse_adaptor
            total_cos_adaptor += cos_adaptor
            
        avg_loss = total_loss / steps
        avg_cos_adaptor = total_cos_adaptor / steps
        avg_mse_adaptor = total_mse_adaptor/steps
        avg_cos_identity = total_cos_identity/steps
        avg_mse_identity = total_mse_identity/steps
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f} | Avg MSE Indentity: {avg_mse_identity:.5f} | Avg MSE Adaptor: {avg_mse_adaptor:.5f} | Avg Cos Indentity: {avg_cos_identity:.5f} | Avg Cos Adaptor: {avg_cos_adaptor:.5f}")

    print(f"Layer {layer_id} Adapter trained.")
    return adapter

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", "-d", type=int, default=32)
    parser.add_argument("--model","-m", type=int, default=1)
    parser.add_argument("--layer_id","-l", type=int, default=0)
    parser.add_argument("--threshold", "-t", type=float, default=0.95)
    args = parser.parse_args()
    layer_id = args.layer_id
    trained_adapter = train_one_layer_adapter(layer_id, epochs=20, lr=1e-3, adapter_dim=args.dim, model_type = args.model)
    
    torch.save(trained_adapter.state_dict(), f"./checkpoint/adaptor/{args.dim}/adapter_layer_{layer_id}_{args.dim}_Model{args.model}_{args.threshold}.pt")