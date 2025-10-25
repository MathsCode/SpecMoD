from transformers.cache_utils import Cache, DynamicCache
import torch
from typing import List, Tuple
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
        
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Buffer only has {len(self)} layers, attempted to access layer with index {layer_idx}")
    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.buffer[layer_idx])
    def __len__(self) -> int:
        return len(self.buffer)
    
    
    def update(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor]:
        self._seen_tokens[layer_idx] += hidden_states.shape[-2]
        if hidden_states is not None:
            if len(self.buffer) <= layer_idx:
                for _ in range(len(self.buffer), layer_idx):
                    self.buffer.append(torch.tensor([]))
            self.buffer.append(hidden_states)
        elif{
            not self.buffer[layer_idx].numel()
        }:
            self.buffer[layer_idx] = hidden_states
        else:
            self.buffer[layer_idx] = torch.cat(
                [self.buffer[layer_idx], hidden_states], dim=-2
            )
        return self.buffer[layer_idx]
    def reset(
        self,
        layer_idx
    ):
        self.buffer[layer_idx] = torch.tensor([])
        
    def get_length(self, layer_idx: int) -> int:
        is_empty_layer = (
            len(self.buffer) <= layer_idx
        )
        length = self.buffer[layer_idx].shape[-2] if not is_empty_layer else 0
        return length
    
    def clear_buffer(self):
        self.buffer = []







storage = DataStorage()