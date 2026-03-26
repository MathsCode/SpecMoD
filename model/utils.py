from transformers.cache_utils import Cache, DynamicCache
import torch
from typing import List, Tuple
from torch import nn as nn
class DataStorage:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._json_data = []
        self._total_length = 0
        self._total_tokens = 0
        self._last_hidden_states = []
        self._layer_hidden_states = []
        self._label = []
    
    def add_normal_info(self, json_item, length):
        self._json_data.append(json_item)
        self._total_length += length
        self._total_tokens += 1
    def get_normal_info(self):
        return self._json_data.copy(), self._total_length, self._total_tokens
    
    
    def add_last_hidden_states(self, last_hidden_states, ):
        self._last_hidden_states.append(last_hidden_states)
    def get_last_hidden_states(self):
        return self._last_hidden_states

    
    def add_layer_hidden_states(self, layer_hidden_states, label, layer_id):
        while len(self._layer_hidden_states) <= layer_id:
            self._layer_hidden_states.append([])
            self._label.append([])
        self._layer_hidden_states[layer_id].append(layer_hidden_states)
        self._label[layer_id].append(label)
    def get_layer_hidden_states(self):
        return  self._layer_hidden_states, self._label
    
    
class Record:
    def __init__(self):
        self.reset()
    def reset(self):
        self.exec_layer_list = []
    def add(self, layer_id):
        self.exec_layer_list.append(layer_id)
    def get_average_len(self):
        return sum(self.exec_layer_list)/len(self.exec_layer_list)

class DynamicBuffer(Cache):
    def __init__(self):
        super().__init__()
        self._seen_tokens = []
        self.buffer : List[torch.Tensor] = []
        self.position_embeddings : List[Tuple[torch.Tensor]] = []
        
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.buffer[layer_idx], self.position[layer_idx])
        else:
            raise KeyError(f"Buffer only has {len(self)} layers, attempted to access layer with index {layer_idx}")
    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.buffer[layer_idx], self.position_embeddings[layer_idx])
    def __len__(self) -> int:
        return len(self.buffer)
    
    
    def update(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: List[Tuple[torch.Tensor]],
        layer_idx: int,
    ):
        if len(self.buffer) <= layer_idx:
            for _ in range(len(self.buffer), layer_idx):
                self._seen_tokens[layer_idx] = 0
                self.buffer.append(torch.tensor([]))
                self.position_embeddings.append((torch.tensor([]),torch.tensor([])))
            self.buffer.append(hidden_states)
            self._seen_tokens.append(hidden_states.shape[-2])
            self.position_embeddings.append(position_embeddings)
        elif not self.buffer[layer_idx].numel():
            self.buffer[layer_idx] = hidden_states
            self.position_embeddings[layer_idx] = position_embeddings
        else:
            self.buffer[layer_idx] = torch.cat(
                [self.buffer[layer_idx], hidden_states], dim=-2
            )
            self.position_embeddings[layer_idx] = (
                torch.cat(
                    [self.position_embeddings[layer_idx][0], position_embeddings[0]],
                    dim=-2,
                ),
                torch.cat(
                    [self.position_embeddings[layer_idx][1], position_embeddings[1]],
                    dim=-2,
                ),
            )
    def get_data(self, layer_idx: int):
        return self.buffer[layer_idx], self.position_embeddings[layer_idx]
    def reset(
        self,
        layer_idx
    ):
        self.buffer[layer_idx] = torch.tensor([])
        self.position_embeddings[layer_idx] = (torch.tensor([]), torch.tensor([]))
        
    def get_length(self, layer_idx: int) -> int:
        is_empty_layer = (
            len(self.buffer) <= layer_idx
        )
        length = self.buffer[layer_idx].shape[-2] if not is_empty_layer else 0
        return length
    
    def clear_buffer(self):
        self.buffer = []
        self.position_embeddings = []



from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple

@dataclass
class Spec_CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None

@dataclass
class Spec_BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

@dataclass
class Spec_SequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


from transformers.generation.utils import ModelOutput, ALL_CACHE_NAMES
from typing import Any, Dict


def Spec_update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)

        if 'last_hidden_state' in model_kwargs:
            model_kwargs['last_hidden_state'] = getattr(outputs, 'last_hidden_state')
        
        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs
    
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

from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


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
    
    
class ShadowAdapter3(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=64):
        """
        Args:
            hidden_dim: 原模型 Hidden Size (e.g., Llama-3-8B is 4096)
            bottleneck_dim: 压缩后的维度 (e.g., 64 or 128), 越小越快
        """
        super().__init__()
        self.norm = Qwen3RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.down_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.act_fn = nn.SiLU()
        self.gate_scale = nn.Parameter(torch.zeros(1))
        # 【关键技巧】零初始化 (Zero Initialization)
        # 让 up_proj 的权重一开始全是 0。
        # 这样初始时: Adapter(x) = 0, Output = x + 0 = x (完美透传)
        # 训练开始后，它会慢慢学到非 0 的修正值。
        
        # nn.init.zeros_(self.up_proj.weight)
        nn.init.kaiming_normal_(self.gate_proj.weight, a=0.2)
        nn.init.kaiming_normal_(self.up_proj.weight, a=0.2)
        nn.init.xavier_normal_(self.down_proj.weight)

    def forward(self, x):
        x_norm = self.norm(x)
        
        # 2. SwiGLU 计算
        # 公式: (SiLU(Gate) * Up) -> Down
        gate = self.act_fn(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        inter = gate * up
        out = self.down_proj(inter)
        return self.gate_scale*out

class Global_router(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=36):

        super().__init__()
        self.norm = Qwen3RMSNorm(input_dim)
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=True)
        self.act_fn = nn.SiLU()

        
        nn.init.kaiming_normal_(self.gate_proj.weight, a=0.2)
        nn.init.kaiming_normal_(self.up_proj.weight, a=0.2)
        nn.init.xavier_normal_(self.down_proj.weight, gain=0.1)
        nn.init.constant_(self.down_proj.bias, -2.0)

    def forward(self, x):
        x_norm = self.norm(x)
        
        # 2. SwiGLU 计算
        # 公式: (SiLU(Gate) * Up) -> Down
        gate = self.act_fn(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        inter = gate * up
        out = self.down_proj(inter)
        return out

class Cos_Sim_storage():
    def __init__(self):
        self.data = []
    def add(self, data, layer_id):
        while len(self.data) <= layer_id:
            self.data.append([])
        self.data[layer_id].append(data)
    def get(self,layer_id):
        return self.data[layer_id]
        
import torch
import torch.nn.functional as F
import numpy as np

def generate_perturbed_hidden_states(clean_hidden_states, num_samples=50):
    """
    生成一系列受到不同程度扰动的 hidden_states。
    
    Args:
        clean_hidden_states: 原始的 hidden state, shape [1, seq_len, hidden_dim]
        num_samples: 采样的点数
        
    Returns:
        List of (lambda, perturbed_hidden_states)
    """
    
    # 获取原始信号的模长 (Norm)，保持量级一致
    # shape: [1, seq_len, 1]
    device = clean_hidden_states.device
    signal_norm = torch.norm(clean_hidden_states, p=2, dim=-1, keepdim=True)
    
    # 设定 lambda (噪声强度) 的范围
    # 0.0 表示无噪声 (Sim=1.0)
    # 10.0 通常足够让 Sim 降到接近 0
    lambdas = np.linspace(0, 10.0, num_samples)
    
    perturbed_list = []
    
    # 固定一个随机噪声方向，还是每个 lambda 随机一个方向？
    # 建议：每个 lambda 都随机生成一个方向，这样更具一般性（蒙特卡洛采样的感觉）
    
    for lam in lambdas:
        # 1. 生成高斯白噪声
        noise = torch.randn_like(clean_hidden_states)
        
        # 2. 对噪声进行归一化 (单位向量化)
        noise_norm = torch.norm(noise, p=2, dim=-1, keepdim=True)
        normalized_noise = noise / (noise_norm + 1e-8)
        
        # 3. 施加扰动
        # 公式: h_new = h + lambda * |h| * noise_unit
        # 这样 lambda=0.1 就意味着噪声幅度是信号幅度的 10%
        perturbation = lam * signal_norm * normalized_noise
        perturbation = perturbation.to(device)
        h_perturbed = clean_hidden_states + perturbation
        input_sim = torch.nn.functional.cosine_similarity(h_perturbed, clean_hidden_states, dim=-1).mean()
        perturbed_list.append((input_sim.item(), h_perturbed))
        
    return perturbed_list

# --- 在主循环中调用 ---
# 假设你已经拿到了某一层干净的 layer_output
# perturbed_data = generate_perturbed_hidden_states(clean_layer_output, 100)
# for lam, h_tilde in perturbed_data:
#     1. 计算输入相似度 Input Sim (h_tilde vs clean_layer_output)
#     2. 把 h_tilde 喂给模型后续层，得到 final_output
#     3. 计算输出相似度 Output Sim
#     4. 画点 (Input Sim, Output Sim)
       
        


storage = DataStorage()
record = Record()
cos_sim_storage = Cos_Sim_storage()
