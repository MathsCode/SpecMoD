# coding=utf-8
"""
Router模型架构 - 基于EAGLE设计,用于预测层执行掩码
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import json
import os
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for GQA"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed





class Qwen3Attention(nn.Module):
    """Router的自注意力层,支持KV Cache"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config["max_position_embeddings"]

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 输入是concat(input_emb, hidden_states),所以是hidden_size * 2
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0]) if cache_hidden else 0

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 更新cache
        if cache_hidden is None:
            local_cache_k = []
            local_cache_v = []
        else:
            local_cache_k = list(cache_hidden[0])
            local_cache_v = list(cache_hidden[1])

        local_cache_k.append(key_states)
        local_cache_v.append(value_states)

        cache_k = local_cache_k
        cache_v = local_cache_v

        k0 = cache_k[0]
        v0 = cache_v[0]

        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        lck = len(cache_k)

        attn_weights = attn_weights + attention_mask

        for i in range(1, lck):
            ki = cache_k[i]
            qi = query_states
            kiq = ki
            attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights0 = attn_weights[..., :q_len]

        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, lck):
            vi = cache_v[i]
            attn_weightsi = attn_weights[..., q_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        new_past_key_value = [local_cache_k, local_cache_v]
        return attn_output, new_past_key_value


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        if config["hidden_act"] == "silu":
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3DecoderLayer(nn.Module):
    """Router的Decoder层,类似EAGLE的midlayer"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.self_attn = Qwen3Attention(config=config)
        self.mlp = Qwen3MLP(config)
        self.hidden_norm = Qwen3RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.input_layernorm = Qwen3RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = Qwen3RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: List[torch.Tensor] = [],
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        # Concat input_emb and hidden_states
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        return_hidden = hidden_states

        # Self Attention
        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, return_hidden)

        return outputs, latest_hidden_cache


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)




class LayerRouterModel(nn.Module):
    """
    Layer Router模型 - 预测每个token需要执行哪些层
    
    架构:
    1. 从target_model获取前3层hidden states
    2. FC融合为single hidden representation
    3. RouterDecoderLayer处理
    4. 输出层预测36层的执行概率
    """
    
    def __init__(self, config, training_config, target_model_path, load_emb=True):
        super().__init__()
        self.train_config = training_config
        self.config = config
        
        # DeepSpeed ZeRO-3兼容性
        # if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        #     dschf = HfDeepSpeedConfig(ds_config)
        # else:
        #     dschf = None
        
        self.gradient_checkpointing = self.train_config.get("gradient_checkpoint", False)
        self.padding_idx = 0
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_target_layers = config["num_target_layers"]
        
        # Router主体层
        self.rotary_emb = Qwen3RotaryEmbedding(
            config=config
        )
        self.midlayer = Qwen3DecoderLayer(config)
        self.norm = Qwen3RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])


        # FC融合层: 3*hidden_size -> hidden_size
        self.fc_fusion = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
        
        # 加载embedding (冻结)
        if load_emb:
            print("Loading embeddings from target model...")
            self.target_model = self.target_model = AutoModelForCausalLM.from_pretrained(
                target_model_path, 
                dtype=torch.float16,
            )
            embed_weight = self.target_model.get_input_embeddings().weight.data.clone().float()
            self.embed_tokens = nn.Embedding(
                config["vocab_size"], 
                config["hidden_size"], 
                self.padding_idx, 
                _weight=embed_weight
            )
            for param in self.embed_tokens.parameters():
                param.requires_grad = False
        else:
            self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"], self.padding_idx)
        
        # 输出层: hidden_size -> num_target_layers
        self.layer_head = nn.Linear(config["hidden_size"], config["num_target_layers"], bias=False)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


    def forward(
        self,
        input_ids,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        layer_masks: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass
        
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            loss_mask: [B, seq_len] - 有效位置掩码
            layer_masks: [B, seq_len, num_target_layers] - ground truth
        
        Returns:
            loss, logits, metrics
        """
        
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        # 2. FC融合
        hidden_states = self.fc_fusion(hidden_states)  # [B, seq_len, hidden_size]

        # 3. 准备position_ids
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 4. 准备attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # 5. 获取input embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad = True
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        # 6. Router Decoder Layer
        cache_hidden = [[], []]
        
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.midlayer),
                inputs_embeds,
                hidden_states,
                cache_hidden,
                attention_mask,
                position_embeddings,
            )
        else:
            layer_outputs, cache_hidden = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states_out = layer_outputs[0]  # [B, seq_len, hidden_size]

        # 7. Output layer
        hidden_states_out = self.norm(hidden_states_out)
        logits = self.layer_head(hidden_states_out)  # [B, seq_len, num_target_layers]

        # 8. 计算损失
        loss = None
        # if layer_masks is not None:
        #     # Binary Cross-Entropy Loss
        #     loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        #     loss = loss_fn(logits, layer_masks.float())  # [B, seq_len, num_target_layers]
            
        #     # 应用loss_mask
        #     if loss_mask is not None:
        #         loss_mask = loss_mask.unsqueeze(-1)  # [B, seq_len, 1]
        #         loss = loss * loss_mask
        #         loss = loss.sum() / (loss_mask.sum() * self.num_target_layers + 1e-6)
        #     else:
        #         loss = loss.mean()
        if layer_masks is not None:
            alpha = 0.25
            gamma = 2.0
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, layer_masks.float(), reduction='none'
            )
            probs =torch.sigmoid(logits)
            
            p_t = probs * layer_masks + (1 - probs) * (1 - layer_masks)
            focal_weight = (1 - p_t) ** gamma
            alpha_weight = alpha * layer_masks + (1 - alpha) * (1 - layer_masks)
        
            loss = focal_weight * alpha_weight * bce_loss
            
            if loss_mask is not None:
                loss_mask = loss_mask.unsqueeze(-1)
                loss = loss * loss_mask
                loss = loss.sum() / (loss_mask.sum() * self.num_target_layers + 1e-6)
            else:
                loss = loss.mean()

        return loss, logits
