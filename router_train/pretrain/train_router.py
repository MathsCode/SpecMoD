# coding=utf-8
"""
训练脚本 - 方案A: 从头训练Layer Router (使用预计算hidden states)
"""
import argparse
import deepspeed
import json
import os
import sys
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate.utils import set_seed

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import PrecomputedHiddenStatesDataset, PrecomputedDataCollator
from utils.metrics import compute_metrics
from pretrain.router_model import LayerRouterModel

from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

torch.backends.cuda.matmul.allow_tf32 = True
set_seed(42)

from typing import Dict

class RouterConfig(Dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value
    


def parse_args():
    parser = argparse.ArgumentParser(description='Train Layer Router - Approach A (Precomputed)')
    parser.add_argument('--target_model_path', type=str, required=True, default="/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B/",
                        help='Path to Qwen3-8B model (for loading embeddings)')
    parser.add_argument('--train_data_path_dir', type=str, required=True, default=f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/global_router/sharegpt",
                        help='Path to training file dir')
    parser.add_argument('--savedir', type=str, default='./checkpoints/global_router_pretrain',
                        help='Directory to save checkpoints')
    parser.add_argument('--config', type=str, default='/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/router_train/pretrain/router_config.json',
                        help='Path to router config')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 加载DeepSpeed配置
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)
    
    # 训练配置
    train_config = {
        "bs": ds_config["train_micro_batch_size_per_gpu"],
        "num_epochs": 100,
        "num_workers": 2,
        "max_len": 2048,
        "num_target_layers": 36,
        "gradient_checkpoint": True
    }
    
    # 加载router配置
    with open(args.config) as f:
        router_config = json.load(f)
        router_config = RouterConfig(router_config)
    # 初始化tokenizer
    print(f"Loading tokenizer from {args.target_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    
    # 构建数据集
    print("Building datasets...")
    traindataset = PrecomputedHiddenStatesDataset(
        path_dir=args.train_data_path_dir,
        num_layers=train_config["num_target_layers"],
        max_len=train_config["max_len"],
        start_id=0,
        end_id=180,
    )
    testdataset = PrecomputedHiddenStatesDataset(
        path_dir=args.train_data_path_dir,
        num_layers=train_config["num_target_layers"],
        max_len=train_config["max_len"],
        start_id=180,
        end_id=184
    )
    
    print(f"Train dataset size: {len(traindataset)}")
    print(f"Test dataset size: {len(testdataset)}")
    
    # 初始化模型
    print("Initializing LayerRouterModel...")
    model = LayerRouterModel(
        config=router_config,
        training_config=train_config,
        target_model_path=args.target_model_path,
        load_emb=True
    )
    
    # DeepSpeed初始化
    print("Initializing DeepSpeed...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )
    
    # 获取分布式训练信息
    global_rank = deepspeed.comm.get_rank()
    local_rank = deepspeed.comm.get_local_rank()
    world_size = deepspeed.comm.get_world_size()
    
    # Wandb初始化
    if global_rank == 0:
        try:
            import wandb
            # 设置离线模式 - 数据保存到本地,之后可以上传
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project="layer-router-approach-a-precomputed",
                config={**train_config, **router_config, **ds_config},
                dir=args.savedir  # 指定wandb文件保存位置
            )
            print(f"\n{'='*60}")
            print(f"Wandb OFFLINE mode enabled")
            print(f"Logs will be saved to: {args.savedir}/wandb")
            print(f"To upload later, run: wandb sync {args.savedir}/wandb/latest-run")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Wandb initialization failed: {e}")
            wandb = None
    else:
        wandb = None
    
    # 创建保存目录
    os.makedirs(args.savedir, exist_ok=True)
    
    # 数据加载器
    test_sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    test_loader = DataLoader(
        testdataset,
        batch_size=train_config["bs"],
        sampler=test_sampler,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        collate_fn=PrecomputedDataCollator(pad_token_id=tokenizer.pad_token_id)
    )
    
    train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(
        traindataset,
        batch_size=train_config["bs"],
        sampler=train_sampler,
        num_workers=train_config["num_workers"],
        pin_memory=True,
        collate_fn=PrecomputedDataCollator(pad_token_id=tokenizer.pad_token_id)
    )
    
    # 训练循环
    num_epochs = train_config["num_epochs"]
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch + 1)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}\n")
        
        # ===== 训练阶段 =====
        model.train()
        epoch_losses = []
        epoch_metrics = {
            "layer_accuracy": [],
            "exact_match": [],
            "f1": []
        }
        
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}") if global_rank == 0 else train_loader
        
        for batch_idx, data in enumerate(train_bar):
            model_engine.zero_grad()
            
            # Forward pass - 使用预计算的hidden states
            loss, logits = model_engine(
                input_ids=data["input_ids"].to(local_rank),
                hidden_states=data["hidden_states"].to(local_rank),  # 关键修改!
                attention_mask=data["attention_mask"].to(local_rank),
                loss_mask=data["loss_mask"].to(local_rank),
                layer_masks=data["layer_masks"].to(local_rank),
            )
            
            # Backward
            model_engine.backward(loss)
            model_engine.step()
            
            # 计算指标
            with torch.no_grad():
                metrics = compute_metrics(
                    logits.detach(),
                    data["layer_masks"].to(local_rank),
                    data["loss_mask"].to(local_rank)
                )
            
            epoch_losses.append(loss.item())
            epoch_metrics["layer_accuracy"].append(metrics["layer_accuracy"])
            epoch_metrics["exact_match"].append(metrics["exact_match"])
            epoch_metrics["f1"].append(metrics["f1"])
            
            # 日志记录
            if global_rank == 0 and batch_idx % 100 == 0:
                if wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/layer_accuracy": metrics["layer_accuracy"],
                        "train/exact_match": metrics["exact_match"],
                        "train/f1": metrics["f1"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": epoch * len(train_loader) + batch_idx
                    })
                
                if isinstance(train_bar, tqdm):
                    train_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{metrics["layer_accuracy"]:.4f}',
                        'f1': f'{metrics["f1"]:.4f}'
                    })
        
        # 聚合训练指标
        avg_train_loss = torch.tensor(epoch_losses).cuda().mean()
        deepspeed.comm.all_reduce(avg_train_loss, op=deepspeed.comm.ReduceOp.AVG)
        
        if global_rank == 0:
            train_acc = sum(epoch_metrics["layer_accuracy"]) / len(epoch_metrics["layer_accuracy"])
            train_f1 = sum(epoch_metrics["f1"]) / len(epoch_metrics["f1"])
            print(f"\n[Train] Loss: {avg_train_loss.item():.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            
            if wandb:
                wandb.log({
                    "train/epoch_loss": avg_train_loss.item(),
                    "train/epoch_acc": train_acc,
                    "train/epoch_f1": train_f1,
                    "epoch": epoch + 1
                })
        
        # ===== 验证阶段 =====
        model.eval()
        val_losses = []
        val_metrics = {
            "layer_accuracy": [],
            "exact_match": [],
            "f1": [],
            "precision": [],
            "recall": []
        }
        
        test_bar = tqdm(test_loader, desc=f"Validation Epoch {epoch+1}") if global_rank == 0 else test_loader
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_bar):
                loss, logits = model_engine(
                    input_ids=data["input_ids"].to(local_rank),
                    hidden_states=data["hidden_states"].to(local_rank),
                    attention_mask=data["attention_mask"].to(local_rank),
                    loss_mask=data["loss_mask"].to(local_rank),
                    layer_masks=data["layer_masks"].to(local_rank),
                )
                
                metrics = compute_metrics(
                    logits,
                    data["layer_masks"].to(local_rank),
                    data["loss_mask"].to(local_rank)
                )
                
                val_losses.append(loss.item())
                val_metrics["layer_accuracy"].append(metrics["layer_accuracy"])
                val_metrics["exact_match"].append(metrics["exact_match"])
                val_metrics["f1"].append(metrics["f1"])
                val_metrics["precision"].append(metrics["precision"])
                val_metrics["recall"].append(metrics["recall"])
        
        # 聚合验证指标
        avg_val_loss = torch.tensor(val_losses).cuda().mean()
        deepspeed.comm.all_reduce(avg_val_loss, op=deepspeed.comm.ReduceOp.AVG)
        
        avg_val_f1 = torch.tensor(val_metrics["f1"]).cuda().mean()
        deepspeed.comm.all_reduce(avg_val_f1, op=deepspeed.comm.ReduceOp.AVG)
        
        if global_rank == 0:
            val_acc = sum(val_metrics["layer_accuracy"]) / len(val_metrics["layer_accuracy"])
            val_f1 = avg_val_f1.item()
            val_precision = sum(val_metrics["precision"]) / len(val_metrics["precision"])
            val_recall = sum(val_metrics["recall"]) / len(val_metrics["recall"])
            val_exact_match = sum(val_metrics["exact_match"]) / len(val_metrics["exact_match"])
            
            print(f"[Valid] Loss: {avg_val_loss.item():.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            print(f"        Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | Exact Match: {val_exact_match:.4f}")
            
            if wandb:
                wandb.log({
                    "val/loss": avg_val_loss.item(),
                    "val/acc": val_acc,
                    "val/f1": val_f1,
                    "val/precision": val_precision,
                    "val/recall": val_recall,
                    "val/exact_match": val_exact_match,
                    "epoch": epoch + 1
                })
        
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 保存checkpoint
        if global_rank == 0:
            save_path = f"{args.savedir}/epoch_{epoch+1}"
            model_engine.save_16bit_model(save_path, exclude_frozen_parameters=True)
            print(f"Saved checkpoint to {save_path}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_save_path = f"{args.savedir}/best_model"
                model_engine.save_16bit_model(best_save_path, exclude_frozen_parameters=True)
                print(f"Saved BEST checkpoint (F1={val_f1:.4f}) to {best_save_path}")
        
        if (epoch + 1) % 10 == 0:
            deepspeed.DeepSpeedEngine.save_checkpoint(
                model_engine,
                save_dir=f"{args.savedir}/full_checkpoint_epoch_{epoch+1}"
            )
    
    if global_rank == 0:
        print(f"\n{'='*60}")
        print(f"Training completed! Best F1: {best_f1:.4f}")
        print(f"{'='*60}\n")
        if wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
