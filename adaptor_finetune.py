import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedTokenizerBase
)
import logging
import sys

from model.utils import ShadowAdapter2, ShadowAdapter3
torch.multiprocessing.set_sharing_strategy('file_system')
# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class JointFinetuningModel(nn.Module):
    """
    包装器模型：包含冻结的 Teacher 和带有 Adapter 的 Student Backbone。
    负责执行"混合前向传播"逻辑。
    """
    def __init__(self, model_name_or_path, adapter_dim=1024):
        super().__init__()
        logger.info(f"正在加载基础模型: {model_name_or_path} ...")
        
        # 加载 HuggingFace 模型
        # 注意：这里加载一次即可，Teacher 和 Backbone 共享除了 Adapter 外的权重内存
        self.teacher = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.float16, # 建议半精度
            trust_remote_code=True,
        )
        
        self.config = self.teacher.config
        num_layers = self.config.num_hidden_layers
        
        # 初始化 Adapters
        logger.info("初始化 Adapters...")
        self.adapters = nn.ModuleList([
            ShadowAdapter3(self.config.hidden_size, adapter_dim) for _ in range(num_layers)
        ])
        self.adapters.to(torch.float32)
        # --- 冻结策略 (关键) ---
        self.teacher.eval() # Teacher 永远是 eval 模式
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 只有 Adapter 需要梯度
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True
    @property
    def backbone(self):
        return self.teacher.model
    def forward(self, input_ids, attention_mask, oracle_mask):
        """
        input_ids: [Batch, SeqLen]
        oracle_mask: [Batch, SeqLen, NumLayers] -> 由 Collator 构造好的全量 Mask
        """
        if self.backbone.layers[0].self_attn.q_proj.weight.device == 'cpu':
            print("警告：模型还在 CPU 上！")
        else:
            # 只在第0号进程打印，避免刷屏
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                pass
        batch_size, seqlen = input_ids.shape
        
        # 1. Teacher Forward (用于蒸馏的目标)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # 2. Student Forward (混合路径)
        hidden_states = self.backbone.embed_tokens(input_ids)
        
        cache_position = torch.arange(seqlen, dtype=torch.long, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = self.backbone.rotary_emb(hidden_states, position_ids)
        causal_mask = self.backbone._update_causal_mask(
            attention_mask, hidden_states, cache_position, None
        )
        
        # 遍历每一层
        for i, (layer, adapter) in enumerate(zip(self.backbone.layers, self.adapters)):
            
            # 准备 Mask
            # oracle_mask: [Batch, SeqLen, NumLayers] -> 取出当前层 -> [Batch, SeqLen, 1]
            # 必须转成 hidden_states 的 dtype (fp16/bf16)，否则会报错
            layer_mask = oracle_mask[:, :, i].unsqueeze(-1).to(hidden_states.dtype)
            
            # --- 路径 A: 原始 Layer ---
            # 即使被 Mask 掉也要计算，因为要保持 Batch 并行。
            # 这里的计算量在 Teacher Forward 里其实算过一次，但为了代码解耦和反向传播图的简单性，
            # 这里重算一次通常是可接受的。如果极致优化可以 Cache Teacher 的中间层。
            with torch.no_grad():
                layer_out = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids, use_cache=False, position_embeddings = position_embeddings)[0]
            
            # --- 路径 B: Adapter ---
            adapter_out = hidden_states + adapter(hidden_states)
            
            # --- 混合 ---
            # mask=1 -> layer_out (Prompt部分 或 Keep部分)
            # mask=0 -> adapter_out (Response部分 且 Skip部分)
            hidden_states = layer_mask * layer_out + (1 - layer_mask) * adapter_out

        hidden_states = self.backbone.norm(hidden_states)
        student_logits = self.teacher.lm_head(hidden_states)

        # 返回字典，包含 Student 和 Teacher 的 logits
        return {
            "logits": student_logits,
            "teacher_logits": teacher_logits
        }

class finetuneDataset(Dataset):
    """
    你需要根据你的实际数据格式修改这个类。
    假设每条数据包含: input_ids, prompt_len, token_skip_masks
    """
    def __init__(self, data_path, num_layers):
        import json
        self.data = []
        with open(data_path, 'r') as f:
            data = json.load(f)
            for meta_data in data:
                train_meta_data = {}
                train_meta_data['input_ids'] = meta_data['input_ids']
                response_mask = torch.zeros((len(meta_data['layer_index']), num_layers))
                train_meta_data['prompt_len'] = len(meta_data['input_ids']) - len(meta_data['layer_index'])
                for idx, layer_index in enumerate(meta_data['layer_index']):
                    for layer_id in layer_index:
                        response_mask[idx, layer_id] = 1.0
                train_meta_data['response_mask'] = response_mask
                self.data.append(train_meta_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class TokenLevelOracleCollator:
    tokenizer: PreTrainedTokenizerBase
    num_layers: int
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        features: list of dict {'input_ids':[], 'prompt_len':int, 'token_skip_masks':[[...]]}
        """
        input_ids = [f['input_ids'] for f in features]
        prompt_lens = [f['prompt_len'] for f in features]
        # token_skip_masks 是 Response 部分的 masks
        response_masks = [f['response_mask'] for f in features] 
        
        # 1. 自动 Padding Input IDs
        # padding=True 会 pad 到当前 Batch 中最长的序列
        batch = self.tokenizer.pad(
            {'input_ids': input_ids}, 
            padding=True, 
            return_tensors='pt'
        )
        
        bsz, max_seq_len = batch['input_ids'].shape
        
        # 2. 构建全局 Oracle Mask [Batch, MaxSeqLen, NumLayers]
        # 初始化为 1.0 (保留/不跳)。
        # 这涵盖了 Prompt 区域 (不跳) 和 Padding 区域 (不跳/无意义)
        final_mask = torch.ones((bsz, max_seq_len, self.num_layers), dtype=torch.float32)
        
        for i in range(bsz):
            p_len = prompt_lens[i]
            r_mask = response_masks[i] # [ResponseLen, NumLayers]
            r_len = r_mask.shape[0]
            
            # 计算 Response 在 Padded Sequence 中的有效区间
            # 从 p_len 开始，到 p_len + r_len 结束 (但不能超过 max_seq_len)
            valid_end = min(p_len + r_len, max_seq_len)
            valid_len = valid_end - p_len
            
            if valid_len > 0:
                # 填入 Response 的实际跳层策略 (0/1)
                final_mask[i, p_len:valid_end, :] = r_mask[:valid_len, :]
        
        batch['oracle_mask'] = final_mask
        
        # 补全 attention_mask (如果 tokenizer 没生成)
        if 'attention_mask' not in batch:
            batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).float16()
            
        return batch


class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            oracle_mask=inputs["oracle_mask"]
        )
        
        student_logits = outputs["logits"]
        teacher_logits = outputs["teacher_logits"]
        mask = inputs["attention_mask"].float()
        # 蒸馏温度
        T = 2.0 
        loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='none'
        ) * (T ** 2)
        loss = loss.sum(dim=-1)
        masked_loss = loss * mask
        valid_token_count = mask.sum()
        if valid_token_count > 0:
            loss = masked_loss.sum() / valid_token_count
        else:
            loss = masked_loss.sum()
        return (loss, outputs) if return_outputs else loss


def main(args):
    # --- 配置区域 ---
    
    MODEL_PATH = f"/inspire/hdd/global_public/public_models/Qwen/{args.model}/"   # 替换模型路径
    DATA_PATH = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/train_data/adaptor_finetune/finetune_data.json"            # 替换数据路径
    PRETRAINED_ADAPTER_DIR = f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/checkpoint/adaptor/{args.adaptor_dim}/" # 你的 Adapter 预训练权重文件夹
    OUTPUT_DIR = f"/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/checkpoint/adaptor/{args.adaptor_dim}_finetune/"
    
    
    # 显存富裕时的配置
    BATCH_SIZE = args.batchsize       # 调大直到 OOM
    GRAD_ACCUM = 1       # 设为 1 最快
    LR = 1e-4

    # 1. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token 

    # 2. 准备数据 (这里使用假数据生成器，你需要换成 MyJsonlDataset)
    
    # 3. 初始化模型
    model = JointFinetuningModel(MODEL_PATH, args.adaptor_dim)
    
    NUM_LAYERS = model.config.num_hidden_layers
    dataset = finetuneDataset(DATA_PATH, NUM_LAYERS)

    
    # 4. 【关键】加载预训练 Adapter 权重
    logger.info(f"Loading pretrained adapters from {PRETRAINED_ADAPTER_DIR}...")
    
    for i in range(NUM_LAYERS):
        adapter_path = os.path.join(PRETRAINED_ADAPTER_DIR, f"adapter_layer_{i}_{args.adaptor_dim}_Model3.pt")
        # 简单检查文件是否存在 (演示时跳过，实际请取消注释)
        if os.path.exists(adapter_path):
             state_dict = torch.load(adapter_path, map_location="cpu")
             model.adapters[i].load_state_dict(state_dict)
             logger.info(f"Loaded Adapter Layer {i}")
        else:
             logger.warning(f"Adapter {i} not found, using random init.")

    # 5. 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=3,
        learning_rate=LR,
        warmup_ratio=0.1,
        
        # 精度设置 (A100/3090/4090 用 bf16)
        # bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=True,
        
        # DDP 关键设置
        ddp_find_unused_parameters=False, # 必须 False (Teacher冻结)
        
        # 数据处理
        remove_unused_columns=False,      # 必须 False (保留oracle_mask)
        dataloader_num_workers=4,
        
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none", # 或 tensorboard
        lr_scheduler_type="cosine",
        # 禁用 DeepSpeed
        deepspeed=None
    )

    # 6. 初始化 Trainer
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=TokenLevelOracleCollator(tokenizer, NUM_LAYERS),
    )

    # 7. 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    from datetime import datetime
    save_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 8. 保存最终的 Adapters
    if trainer.is_world_process_zero():
        save_path = os.path.join(OUTPUT_DIR, f"final_adapters_{args.adaptor_dim}_{save_timestamp}.pt")
        logger.info(f"Saving adapters to {save_path}")
        # 只保存 Adapter 权重，不保存整个大模型
        torch.save(model.adapters.state_dict(), save_path)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument("--adaptor_dim","-d", type=int, default=2048)
    parser.add_argument("--batchsize","-b", type=int, default=1)
    args = parser.parse_args()
    main(args)