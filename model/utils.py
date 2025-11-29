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
        self._true_last_hidden_states = []
        self._fake_last_hidden_states = []
        self._cur_hidden_states = []
    
    def add(self, json_item, length, cur_hidden_states, fake_last_hidden_states):
        self._json_data.append(json_item)
        self._total_length += length
        self._total_tokens += 1
        self._cur_hidden_states.append(cur_hidden_states)
        self._fake_last_hidden_states.append(fake_last_hidden_states)
    def add_true_last_hidden_states(self, true_last_hidden_states, ):
        self._true_last_hidden_states.append(true_last_hidden_states)
    def get_data(self):
        return self._json_data.copy(), self._cur_hidden_states, self._fake_last_hidden_states, self._true_last_hidden_states, self._total_length, self._total_tokens

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

storage = DataStorage()