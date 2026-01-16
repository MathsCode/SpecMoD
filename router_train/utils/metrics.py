# coding=utf-8
"""
评估指标 - Layer Router性能评估
"""
import torch
import numpy as np
from typing import Dict


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算Layer Router的各项评估指标
    
    Args:
        predictions: [B, seq_len, num_layers] 模型输出logits
        targets: [B, seq_len, num_layers] ground truth (0/1)
        mask: [B, seq_len, 1] 有效位置掩码
        threshold: 二分类阈值
    
    Returns:
        metrics字典
    """
    # 应用sigmoid将logits转为概率
    probs = torch.sigmoid(predictions)
    # 二值化预测
    preds = (probs > threshold).float()
    
    if mask is not None:
        # 只计算有效位置
        mask = mask.squeeze(-1)  # [B, seq_len]
        mask_3d = mask.unsqueeze(-1)  # [B, seq_len, 1]
        
        preds = preds * mask_3d
        targets = targets * mask_3d
        valid_count = mask.sum().item()
    else:
        valid_count = predictions.shape[0] * predictions.shape[1]
    
    # 1. Per-layer Accuracy: 每个位置预测正确的层数占比
    correct_layers = (preds == targets).float()  # [B, seq_len, num_layers]
    per_position_acc = correct_layers.mean(dim=-1)  # [B, seq_len]
    
    if mask is not None:
        layer_accuracy = (per_position_acc * mask).sum() / (valid_count + 1e-6)
    else:
        layer_accuracy = per_position_acc.mean()
    
    # 2. Exact Match: 整个36维向量完全匹配的位置比例
    exact_match = (correct_layers.sum(dim=-1) == predictions.shape[-1]).float()  # [B, seq_len]
    
    if mask is not None:
        exact_match_ratio = (exact_match * mask).sum() / (valid_count + 1e-6)
    else:
        exact_match_ratio = exact_match.mean()
    
    # 3. Hamming Loss: 预测错误的层占比
    hamming_loss = 1 - layer_accuracy
    
    # 4. Precision, Recall, F1 (宏平均)
    # 将所有有效位置展平计算
    if mask is not None:
        valid_positions = mask.bool()
        preds_flat = preds[valid_positions.unsqueeze(-1).expand_as(preds)].reshape(-1, predictions.shape[-1])
        targets_flat = targets[valid_positions.unsqueeze(-1).expand_as(targets)].reshape(-1, targets.shape[-1])
    else:
        preds_flat = preds.reshape(-1, predictions.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])
    
    # 计算每一层的precision/recall
    precisions = []
    recalls = []
    f1s = []
    
    num_layers = predictions.shape[-1]
    for layer_idx in range(num_layers):
        pred_layer = preds_flat[:, layer_idx]
        target_layer = targets_flat[:, layer_idx]
        
        tp = (pred_layer * target_layer).sum().item()
        fp = (pred_layer * (1 - target_layer)).sum().item()
        fn = ((1 - pred_layer) * target_layer).sum().item()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # 5. Sparsity: 预测执行的层占比 (越小越好,说明跳过更多层)
    pred_sparsity = preds.sum(dim=-1).mean().item() / num_layers
    target_sparsity = targets.sum(dim=-1).mean().item() / num_layers
    
    metrics = {
        "layer_accuracy": layer_accuracy.item(),
        "exact_match": exact_match_ratio.item(),
        "hamming_loss": hamming_loss.item(),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
        "pred_sparsity": pred_sparsity,
        "target_sparsity": target_sparsity,
    }
    
    return metrics


def compute_per_layer_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor = None
) -> Dict[int, float]:
    """
    计算每一层的单独准确率
    
    Returns:
        {layer_id: accuracy}
    """
    probs = torch.sigmoid(predictions)
    preds = (probs > 0.5).float()
    
    num_layers = predictions.shape[-1]
    per_layer_acc = {}
    
    for layer_idx in range(num_layers):
        pred_layer = preds[..., layer_idx]
        target_layer = targets[..., layer_idx]
        
        correct = (pred_layer == target_layer).float()
        
        if mask is not None:
            mask_2d = mask.squeeze(-1)
            acc = (correct * mask_2d).sum() / (mask_2d.sum() + 1e-6)
        else:
            acc = correct.mean()
        
        per_layer_acc[layer_idx] = acc.item()
    
    return per_layer_acc
