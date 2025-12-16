from typing import Any, Dict, List, Optional, Tuple
from numpy import dtype
import torch
import math
from collections import OrderedDict

class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")


############## Single GPU Cache ###############
class FlashSimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)

        self.scores = []

    def print_status(self):
        print("[Full Cache] Cached:", self.seq_len, "| Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states

        key = self.key_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]

        return key, value

class OffloadingFlashSimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype
        self.device = model.device

        self.key_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu').pin_memory()
        self.value_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu').pin_memory()

        # init layer cache buffer on chip
        self.key_cache_buffer = torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=self.device)
        self.value_cache_buffer = torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=self.device)

        self.load_stream = torch.cuda.Stream(device=self.device)

    def print_status(self):
        print("[Offloading Flash Simple Cache] Cached Size:", self.seq_len, "| Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # copy incoming k v cache to cpu
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states.cpu()
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states.cpu()

        # copy k v cache to buffer
        self.key_cache_buffer.copy_(self.key_cache[layer_idx], non_blocking=True)
        self.value_cache_buffer.copy_(self.value_cache[layer_idx], non_blocking=True)
        
        key = self.key_cache_buffer[:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache_buffer[:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]

        return key, value

class RetrievalCache(Cache):
    def __init__(self, model, max_budget=1024, prefill=1024, chunk_size=8, gamma=6) -> None:
        
        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.gamma = gamma
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"

        self.real_budget = max_budget + gamma + 1

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)

        self.init_graph = False

    def print_status(self):
        print("[Retrieval Cache] Budget:", self.max_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        # query_states: (bsz, 1, 32, head_dim) --> (bsz, 32, 1, head_dim)
        # key_cache: (bsz, seq_len, 32, head_dim) --> (bsz, 32, head_dim, seq_len)
        # print(query_states.shape, self.chunk_k[layer_idx].shape)

        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        chunk_k = kv_cache.key_cache[layer_idx,:,:self.prefill].cuda().view(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        
        # (bsz, 32, chunks)
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2)
        # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)
        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = kv_cache.key_cache[layer_idx][:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        self.key_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        value_ = kv_cache.value_cache[layer_idx][:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        self.value_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        if layer_idx == self.layers-1:
            self.init_graph = True

    def update_graph_cache(self, kv_cache=None):
        self.value_cache[:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.value_cache[:,:, self.prefill:kv_cache.seq_len].clone()
        self.key_cache[:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.key_cache[:,:, self.prefill:kv_cache.seq_len].clone()

    def update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int):

        self.key_cache[layer_idx][:, self.real_budget-self.gamma-1:] = new_k_cache.clone()
        self.value_cache[layer_idx][:, self.real_budget-self.gamma-1:] = new_v_cache.clone()

        return self.key_cache[layer_idx][:,:self.real_budget], self.value_cache[layer_idx][:,:self.real_budget]

    def update_graph_cache_retrieval(self, kv_cache, query_states, layer_idx):
        self.init_graph_cache(kv_cache, query_states, layer_idx)
        self.value_cache[layer_idx,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.value_cache[layer_idx,:, self.prefill:kv_cache.seq_len].clone()
        self.key_cache[layer_idx,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.key_cache[layer_idx,:, self.prefill:kv_cache.seq_len].clone()

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()

class StreamingLLMEvictionCache(Cache):

    def __init__(self, model, gamma=6, start_size=16, recent_size=496) -> None:

        self.gamma = gamma
        self.start_size = start_size
        self.recent_size = recent_size
        self.real_budget = self.start_size + self.recent_size + self.gamma + 1 + 1 + 1

        self.seq_len = 0 # just for prefill usage

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        self.key_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
    
    def print_status(self):
        print("[StreamingLLM Cache] Start Size:", self.start_size, "| Recent Size:", self.recent_size, "| Gamma:", self.gamma, "| Real Budget:", self.real_budget, "| Cached:", self.seq_len)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        
        incoming = key_states.shape[-3]

        assert self.seq_len + incoming <= self.start_size + self.recent_size
        self.key_cache[layer_idx][:, self.seq_len:self.seq_len + incoming] = key_states.clone()
        self.value_cache[layer_idx][:, self.seq_len:self.seq_len + incoming] = value_states.clone()

        key = self.key_cache[layer_idx][:, :self.seq_len + incoming]
        value = self.value_cache[layer_idx][:, :self.seq_len + incoming]

        if layer_idx == self.layers-1:
            self.seq_len += incoming
        return key, value

    def spec_update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int, gamma_offset=0):

        start = self.real_budget-self.gamma-3
        end = self.real_budget-self.gamma-3+new_k_cache.shape[-3]

        self.key_cache[layer_idx][:, start:end] = new_k_cache.clone()
        self.value_cache[layer_idx][:, start:end] = new_v_cache.clone()

        return self.key_cache[layer_idx][:,:end], self.value_cache[layer_idx][:,:end]

    def reset(self):
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def evict_prefill(self, incoming):
        # evict
        if self.seq_len + incoming <= self.start_size + self.recent_size:
            return
        for layer_idx in range(self.layers):
            size_keep = self.recent_size - incoming
            self.key_cache[layer_idx][:, self.start_size:self.start_size+size_keep] = self.key_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()
            self.value_cache[layer_idx][:, self.start_size:self.start_size+size_keep] = self.value_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()

        self.seq_len = self.start_size + self.recent_size - incoming

    def evict_for_spec(self, current_seq_len):
        self.key_cache[:,:,self.start_size:self.start_size+self.recent_size] = self.key_cache[:,:, current_seq_len-self.recent_size:current_seq_len].clone()
        self.value_cache[:,:, self.start_size:self.start_size+self.recent_size] = self.value_cache[:,:, current_seq_len-self.recent_size:current_seq_len].clone()

############## Dist Cache ###############
class DistributedSimpleCache(Cache):
    def __init__(
        self,
        config,
        max_budget=1024,
        device=None,
        on_chip_layers=0,
        ssl=0,
        kv_cache_quant: bool = False,
        kv_cache_quant_bits: int = 8,
        kv_fp16_tail: int = 0,
        kv_resident_gpu: bool = False,
        kv_resident_max_layers: int = 0,
    ):
        self.config = config
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.device  = device
        self.ssl = ssl
        self.ssl_cur = 0
        self.max_budget = max_budget
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_key_value_heads // self.world_size
        
        self.head_dim = self.hidden_size // self.config.num_attention_heads
        self.layers = self.config.num_hidden_layers
        
        self.seq_len = 0
        dtype=torch.float16
        self.on_chip_layers = on_chip_layers
        self.kv_quant_enabled = bool(kv_cache_quant)
        self.kv_quant_bits = int(kv_cache_quant_bits)
        self.kv_fp16_tail = int(kv_fp16_tail)
        self.kv_resident_gpu = bool(kv_resident_gpu)
        self.kv_resident_max_layers = int(kv_resident_max_layers)
        if self.kv_quant_bits != 8:
            raise ValueError("Only int8 KV cache quantization is supported (kv_cache_quant_bits=8).")
        self._qmax = 127
        self._eps = 1e-6
        self._k_block = 128
        self._resident: "OrderedDict[int, tuple[torch.Tensor, torch.Tensor, int]]" = OrderedDict()

        self.key_cache = torch.zeros([self.on_chip_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=device)
        self.value_cache = torch.zeros([self.on_chip_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=device)

        cpu_layers = self.layers - self.on_chip_layers
        if not self.kv_quant_enabled:
            self.cpu_key_cache=torch.zeros([cpu_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu', pin_memory=True)
            self.cpu_value_cache=torch.zeros([cpu_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu', pin_memory=True)
        else:
            self.cpu_key_cache = None
            self.cpu_value_cache = None
            self.cpu_key_q = torch.zeros([cpu_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=torch.int8, device='cpu', pin_memory=True)
            self.cpu_value_q = torch.zeros([cpu_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=torch.int8, device='cpu', pin_memory=True)
            num_blocks = (self.max_budget + self._k_block - 1) // self._k_block
            self.cpu_k_scale_blocks = torch.zeros([cpu_layers, num_blocks, self.num_heads, self.head_dim], dtype=torch.float16, device='cpu', pin_memory=True)
            self.cpu_v_scale = torch.zeros([cpu_layers, 1, self.max_budget, self.num_heads, 1], dtype=torch.float16, device='cpu', pin_memory=True)

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget)

    def reset(self):
        self.seq_len = 0
        if self.cpu_key_cache is not None:
            self.cpu_key_cache.zero_()
        if self.cpu_value_cache is not None:
            self.cpu_value_cache.zero_()
        self.key_cache.zero_()
        self.value_cache.zero_()
        self._resident.clear()
        if self.kv_quant_enabled:
            self.cpu_key_q.zero_()
            self.cpu_value_q.zero_()
            self.cpu_k_scale_blocks.zero_()
            self.cpu_v_scale.zero_()

    def normal_(self, seq_len=1024*127):
        self.seq_len = seq_len
        if self.cpu_key_cache is not None:
            self.cpu_key_cache.normal_()
        if self.cpu_value_cache is not None:
            self.cpu_value_cache.normal_()
        self.key_cache.normal_()
        self.value_cache.normal_()

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx + 1 <= self.on_chip_layers, (layer_idx, self.on_chip_layers)
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[1]] = key_states.clone()
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[1]] = value_states.clone()

        key = self.key_cache[layer_idx][:, :self.seq_len + value_states.shape[1]]
        value = self.value_cache[layer_idx][:, :self.seq_len + value_states.shape[1]]

        return key, value

    def ssl_update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx + 1 <= self.ssl, (layer_idx, self.ssl)
        self.key_cache[layer_idx][:, self.seq_len+self.ssl_cur : self.seq_len+self.ssl_cur + key_states.shape[1]] = key_states.clone()
        self.value_cache[layer_idx][:, self.seq_len+self.ssl_cur : self.seq_len+self.ssl_cur + value_states.shape[1]] = value_states.clone()

        key = self.key_cache[layer_idx][:, :self.seq_len+self.ssl_cur + value_states.shape[1]]
        value = self.value_cache[layer_idx][:, :self.seq_len+self.ssl_cur + value_states.shape[1]]

        if layer_idx == self.ssl - 1:
            self.ssl_cur += key_states.shape[1]

        return key, value

    def gather_kv_incremental(self, indices: list[int], offset:int):
        if self.kv_quant_enabled:
            raise NotImplementedError("gather_kv_incremental is not supported with kv_cache_quant enabled.")
        indices = [i + offset for i in indices]

        self.key_cache[:,:, offset:offset + len(indices)].copy_(self.key_cache[:,:, indices].clone(), non_blocking=True)
        self.value_cache[:,:, offset:offset + len(indices)].copy_(self.value_cache[:,:, indices].clone(), non_blocking=True)

        if self.cpu_key_cache is not None:
            self.cpu_key_cache[:, :, offset:offset + len(indices)].copy_(self.cpu_key_cache[:, :, indices].clone(), non_blocking=True)
        if self.cpu_value_cache is not None:
            self.cpu_value_cache[:, :, offset:offset + len(indices)].copy_(self.cpu_value_cache[:, :, indices].clone(), non_blocking=True)
        if self.kv_quant_enabled:
            self.cpu_key_q[:, :, offset:offset + len(indices)].copy_(self.cpu_key_q[:, :, indices].clone(), non_blocking=True)
            self.cpu_value_q[:, :, offset:offset + len(indices)].copy_(self.cpu_value_q[:, :, indices].clone(), non_blocking=True)
            self.cpu_v_scale[:, :, offset:offset + len(indices)].copy_(self.cpu_v_scale[:, :, indices].clone(), non_blocking=True)

        self.seq_len = offset + len(indices)
        self.ssl_cur = 0

    def get_cpu_kv_fp16(self, layer_idx: int, start: int, end: int, device=None):
        layer_off = layer_idx - self.on_chip_layers
        if device is None:
            device = self.device
        if not self.kv_quant_enabled:
            if self.cpu_key_cache is None or self.cpu_value_cache is None:
                raise RuntimeError("cpu_key_cache/cpu_value_cache are not allocated (kv_quant_enabled=True).")
            k = self.cpu_key_cache[layer_off][:, start:end].to(device, non_blocking=True)
            v = self.cpu_value_cache[layer_off][:, start:end].to(device, non_blocking=True)
            return k, v

        qk = self.cpu_key_q[layer_off][:, start:end].to(device, non_blocking=True)
        qv = self.cpu_value_q[layer_off][:, start:end].to(device, non_blocking=True)
        v_scale = self.cpu_v_scale[layer_off][:, start:end].to(device, non_blocking=True)  # (1,T,H,1)
        k = torch.empty_like(qk, dtype=torch.float16, device=device)
        b0 = start // self._k_block
        b1 = (end + self._k_block - 1) // self._k_block
        scales = self.cpu_k_scale_blocks[layer_off, b0:b1].to(device, non_blocking=True)  # (B,H,D)
        for bi in range(b0, b1):
            s = max(start, bi * self._k_block) - start
            e = min(end, (bi + 1) * self._k_block) - start
            k[:, s:e].copy_(qk[:, s:e].to(torch.float16) * scales[bi - b0].unsqueeze(0).unsqueeze(0))
        v = qv.to(torch.float16) * v_scale
        return k, v

    def get_cpu_kv_fp16_all(self, start: int, end: int, device=None):
        if device is None:
            device = self.device
        if not self.kv_quant_enabled:
            if self.cpu_key_cache is None or self.cpu_value_cache is None:
                raise RuntimeError("cpu_key_cache/cpu_value_cache are not allocated (kv_quant_enabled=True).")
            k = self.cpu_key_cache[:, :, start:end].to(device, non_blocking=True)
            v = self.cpu_value_cache[:, :, start:end].to(device, non_blocking=True)
            return k, v

        qk = self.cpu_key_q[:, :, start:end].to(device, non_blocking=True)
        qv = self.cpu_value_q[:, :, start:end].to(device, non_blocking=True)
        v_scale = self.cpu_v_scale[:, :, start:end].to(device, non_blocking=True)  # (L,1,T,H,1)
        k = torch.empty_like(qk, dtype=torch.float16, device=device)
        b0 = start // self._k_block
        b1 = (end + self._k_block - 1) // self._k_block
        scales = self.cpu_k_scale_blocks[:, b0:b1].to(device, non_blocking=True)  # (L,B,H,D)
        for bi in range(b0, b1):
            s = max(start, bi * self._k_block) - start
            e = min(end, (bi + 1) * self._k_block) - start
            k[:, :, s:e].copy_(qk[:, :, s:e].to(torch.float16) * scales[:, bi - b0].unsqueeze(1).unsqueeze(1))
        v = qv.to(torch.float16) * v_scale
        return k, v

    def _resident_get_or_create(self, layer_off: int):
        if not self.kv_resident_gpu or self.kv_resident_max_layers <= 0:
            return None
        if layer_off in self._resident:
            k, v, valid = self._resident.pop(layer_off)
            self._resident[layer_off] = (k, v, valid)
            return k, v, valid
        while len(self._resident) >= self.kv_resident_max_layers:
            _, (k_old, v_old, _) = self._resident.popitem(last=False)
            del k_old
            del v_old
        k = torch.zeros((1, self.max_budget, self.num_heads, self.head_dim), device=self.device, dtype=torch.float16)
        v = torch.zeros((1, self.max_budget, self.num_heads, self.head_dim), device=self.device, dtype=torch.float16)
        self._resident[layer_off] = (k, v, 0)
        return k, v, 0

    def resident_dequant_into(self, layer_idx: int, target_len: int):
        if not self.kv_quant_enabled:
            return None
        if not self.kv_resident_gpu or self.kv_resident_max_layers <= 0:
            return None
        if layer_idx < self.on_chip_layers:
            return None
        layer_off = layer_idx - self.on_chip_layers
        res = self._resident_get_or_create(layer_off)
        if res is None:
            return None
        k, v, valid = res

        if target_len <= valid:
            self._resident[layer_off] = (k, v, int(target_len))
            return k, v, int(target_len)

        start = valid
        end = int(target_len)
        qk = self.cpu_key_q[layer_off][:, start:end].to(self.device, non_blocking=True)
        qv = self.cpu_value_q[layer_off][:, start:end].to(self.device, non_blocking=True)

        b0 = start // self._k_block
        b1 = (end + self._k_block - 1) // self._k_block
        scales = self.cpu_k_scale_blocks[layer_off, b0:b1].to(self.device, non_blocking=True)  # (B,H,D)
        for bi in range(b0, b1):
            s = max(start, bi * self._k_block)
            e = min(end, (bi + 1) * self._k_block)
            ss = s - start
            ee = e - start
            k[:, s:e].copy_(qk[:, ss:ee].to(torch.float16) * scales[bi - b0].unsqueeze(0).unsqueeze(0))

        v_scale = self.cpu_v_scale[layer_off][:, start:end].to(self.device, non_blocking=True)  # (1,T,H,1)
        v[:, start:end].copy_(qv.to(torch.float16) * v_scale.to(torch.float16))

        self._resident[layer_off] = (k, v, end)
        return k, v, end

    def copy_back_from_buffer(self, kv_buffer, layer_idx:int):
        layer_off = layer_idx - self.on_chip_layers
        start = self.seq_len
        end = kv_buffer.seq_len
        if end <= start:
            return

        if not self.kv_quant_enabled:
            self.cpu_key_cache[layer_off][:, start:end].copy_(kv_buffer.key_cache[:, start:end], non_blocking=True)
            self.cpu_value_cache[layer_off][:, start:end].copy_(kv_buffer.value_cache[:, start:end], non_blocking=True)
        else:
            k = kv_buffer.key_cache[:, start:end]
            v = kv_buffer.value_cache[:, start:end]

            k_q = torch.empty_like(k, dtype=torch.int8, device=k.device)
            b0 = start // self._k_block
            b1 = (end + self._k_block - 1) // self._k_block
            for bi in range(b0, b1):
                s = max(start, bi * self._k_block) - start
                e = min(end, (bi + 1) * self._k_block) - start
                scale_cpu = self.cpu_k_scale_blocks[layer_off, bi]  # (H,D)
                if torch.all(scale_cpu == 0):
                    k_max = k[:, s:e].abs().amax(dim=1).squeeze(0)
                    scale = (k_max / self._qmax).clamp_min(self._eps)
                    self.cpu_k_scale_blocks[layer_off, bi].copy_(scale.to('cpu', dtype=torch.float16))
                    scale_gpu = scale
                else:
                    scale_gpu = scale_cpu.to(k.device, non_blocking=True).to(k.dtype)
                k_q[:, s:e].copy_(torch.clamp(torch.round(k[:, s:e] / scale_gpu.unsqueeze(0).unsqueeze(0)), -self._qmax, self._qmax).to(torch.int8))

            v_max = v.abs().amax(dim=-1, keepdim=True)
            v_scale = (v_max / self._qmax).clamp_min(self._eps)
            v_q = torch.clamp(torch.round(v / v_scale), -self._qmax, self._qmax).to(torch.int8)

            self.cpu_key_q[layer_off][:, start:end].copy_(k_q, non_blocking=True)
            self.cpu_value_q[layer_off][:, start:end].copy_(v_q, non_blocking=True)
            self.cpu_v_scale[layer_off][:, start:end].copy_(v_scale.to(dtype=torch.float16), non_blocking=True)

        if layer_idx == self.layers - 1:
            self.seq_len = kv_buffer.seq_len
            self.ssl_cur = 0

class DistributedKVCacheBuffer:
    def __init__(self, config, max_budget=1024, device=None) -> None:

        self.config = config
        self.max_budget = max_budget
        self.device = device
        self.dtype = torch.float16

        self.world_size = config.world_size
        self.local_rank = config.local_rank

        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_key_value_heads // self.world_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.key_cache = torch.zeros(1, self.max_budget, self.num_heads, self.head_dim, device=self.device,dtype=self.dtype)
        self.value_cache = torch.zeros(1, self.max_budget, self.num_heads, self.head_dim, device=self.device,dtype=self.dtype)
        self.seq_len = 0
        self.layer_idx = None

    def copy_kv(self, kv_cache, layer_idx):
        on_chip_layers = kv_cache.on_chip_layers
        if self.layer_idx is None or self.layer_idx != layer_idx:
            self.layer_idx = layer_idx
            self.seq_len = 0

        new_len = kv_cache.seq_len
        if new_len <= self.seq_len:
            self.seq_len = new_len
            return

        start = self.seq_len
        end = new_len
        if getattr(kv_cache, "kv_quant_enabled", False) and getattr(kv_cache, "kv_resident_gpu", False):
            if layer_idx >= on_chip_layers:
                res = kv_cache.resident_dequant_into(layer_idx, new_len)
                if res is not None:
                    k, v, valid = res
                    self.key_cache = k
                    self.value_cache = v
                    self.seq_len = valid
                    return
        if not getattr(kv_cache, "kv_quant_enabled", False):
            self.key_cache[:, start:end].copy_(kv_cache.cpu_key_cache[layer_idx-on_chip_layers][:, start:end], non_blocking=True)
            self.value_cache[:, start:end].copy_(kv_cache.cpu_value_cache[layer_idx-on_chip_layers][:, start:end], non_blocking=True)
        else:
            layer_off = layer_idx - on_chip_layers
            qk = kv_cache.cpu_key_q[layer_off][:, start:end].to(self.device, non_blocking=True)
            qv = kv_cache.cpu_value_q[layer_off][:, start:end].to(self.device, non_blocking=True)

            b0 = start // kv_cache._k_block
            b1 = (end + kv_cache._k_block - 1) // kv_cache._k_block
            scales = kv_cache.cpu_k_scale_blocks[layer_off, b0:b1].to(self.device, non_blocking=True).to(self.dtype)  # (B,H,D)
            for bi in range(b0, b1):
                s = max(start, bi * kv_cache._k_block)
                e = min(end, (bi + 1) * kv_cache._k_block)
                ss = s - start
                ee = e - start
                self.key_cache[:, s:e].copy_(qk[:, ss:ee].to(self.dtype) * scales[bi - b0].unsqueeze(0).unsqueeze(0))

            v_scale = kv_cache.cpu_v_scale[layer_off][:, start:end].to(self.device, non_blocking=True).to(self.dtype)  # (1,T,H,1)
            self.value_cache[:, start:end].copy_(qv.to(self.dtype) * v_scale)

        self.seq_len = new_len

    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int):
        input_length = key_states.shape[1]
        self.key_cache[:,self.seq_len:self.seq_len + input_length] = key_states
        self.value_cache[:,self.seq_len:self.seq_len + input_length] = value_states
        self.seq_len += input_length
        return self.key_cache[:,:self.seq_len], self.value_cache[:,:self.seq_len]

class DistributedRetrievalCache_Seqouia:

    def __init__(self, config, max_budget=1024, device=None, prefill=1024, chunk_size=8, tree_size=128) -> None:

        self.config = config
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.device  = device
        
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_key_value_heads // self.world_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.layers = self.config.num_hidden_layers

        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.tree_size = tree_size
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"
        self.real_budget = max_budget + tree_size
        self.init_graph = False
        self.device=device
        dtype=torch.float16
        self.key_cache=torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
        self.value_cache=torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)

    def print_status(self):
        print("Budget:", self.max_budget, " | Real Budget:", self.real_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        if self.init_graph == True:
            raise ValueError("Graph is already initialized")
        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        if hasattr(kv_cache, 'cpu_key_cache'):
            key_cache = kv_cache.key_cache[layer_idx]
            value_cache = kv_cache.value_cache[layer_idx]
        else:
            key_cache = kv_cache.key_cache
            value_cache = kv_cache.value_cache

        # chunk_k: (bsz, chunks, chunk_size, kv_heads, head_dim) --> (bsz, chunks, kv_heads, head_dim)
        chunk_k = key_cache[:,:self.prefill].view(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2) # (bsz, 32, chunks)
        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1) # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)
        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = key_cache[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        self.key_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        value_ = value_cache[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        self.value_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        if layer_idx == self.layers-1:
            self.init_graph = True


    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int, storage_ids):
        input_length = len(storage_ids)
        assert input_length == key_states.shape[1]
        assert input_length == value_states.shape[1]

        self.key_cache[layer_idx].index_copy_(dim=1, index=storage_ids, source=key_states)
        self.value_cache[layer_idx].index_copy_(dim=1, index=storage_ids, source=value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_graph_cache(self, kv_cache=None):

        # on-chip layers
        on_chip_layers = kv_cache.on_chip_layers
        self.value_cache[:on_chip_layers,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.value_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)
        self.key_cache[:on_chip_layers,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.key_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)

        # cpu layers
        self.value_cache[on_chip_layers:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.cpu_value_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)
        self.key_cache[on_chip_layers:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.cpu_key_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.init_graph = False

    def normal_(self):
        self.key_cache.normal_()
        self.value_cache.normal_()

class DistributedRetrievalCache:
    def __init__(self, config, max_budget=1024, device=None, prefill=1024, chunk_size=8, gamma=6) -> None:

        self.config = config
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.device  = device
        
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_key_value_heads // self.world_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.layers = self.config.num_hidden_layers

        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.gamma = gamma
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"
        self.real_budget = max_budget + gamma + 1
        self.init_graph = False
        self.graph_postfill_len = 0
        self.device=device
        dtype=torch.float16
        self.key_cache=torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
        self.value_cache=torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)

    def print_status(self):
        print("Budget:", self.max_budget, " | Real Budget:", self.real_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        if self.init_graph == True:
            raise ValueError("Graph is already initialized")
        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        if hasattr(kv_cache, 'cpu_key_cache'):
            key_cache = kv_cache.key_cache[layer_idx]
            value_cache = kv_cache.value_cache[layer_idx]
        else:
            key_cache = kv_cache.key_cache
            value_cache = kv_cache.value_cache

        # chunk_k: (bsz, chunks, chunk_size, kv_heads, head_dim) --> (bsz, chunks, kv_heads, head_dim)
        chunk_k = key_cache[:,:self.prefill].view(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2) # (bsz, 32, chunks)
        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1) # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)
        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = key_cache[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.key_cache[layer_idx][:,:self.max_budget].copy_(result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim))

        value_ = value_cache[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.value_cache[layer_idx][:,:self.max_budget].copy_(result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim))

        if layer_idx == self.layers-1:
            self.init_graph = True
            self.graph_postfill_len = 0


    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int):

        self.key_cache[layer_idx][:, self.real_budget-self.gamma-1:] = key_states.clone()
        self.value_cache[layer_idx][:, self.real_budget-self.gamma-1:] = value_states.clone()

        return self.key_cache[layer_idx][:,:self.real_budget], self.value_cache[layer_idx][:,:self.real_budget]

    def update_graph_cache(self, kv_cache=None):
        on_chip_layers = kv_cache.on_chip_layers
        new_post = max(0, kv_cache.seq_len - self.prefill)
        if new_post == self.graph_postfill_len:
            return
        if new_post > self.max_budget:
            new_post = self.max_budget

        old_post = self.graph_postfill_len
        old_start = self.max_budget - old_post
        new_start = self.max_budget - new_post

        if old_post > 0 and new_post > 0:
            moved = min(old_post, new_post)
            tmp_k = self.key_cache[:, :, old_start:self.max_budget].clone()
            tmp_v = self.value_cache[:, :, old_start:self.max_budget].clone()
            self.key_cache[:, :, new_start:new_start + moved].copy_(tmp_k[:, :, :moved])
            self.value_cache[:, :, new_start:new_start + moved].copy_(tmp_v[:, :, :moved])

        if new_post > old_post:
            delta = new_post - old_post
            src_s = self.prefill + old_post
            src_e = self.prefill + new_post
            dst_s = self.max_budget - delta
            dst_e = self.max_budget

            # on-chip layers
            self.key_cache[:on_chip_layers, :, dst_s:dst_e].copy_(kv_cache.key_cache[:, :, src_s:src_e], non_blocking=True)
            self.value_cache[:on_chip_layers, :, dst_s:dst_e].copy_(kv_cache.value_cache[:, :, src_s:src_e], non_blocking=True)

            # cpu layers
            if on_chip_layers < self.layers:
                k_cpu, v_cpu = kv_cache.get_cpu_kv_fp16_all(src_s, src_e, device=self.device)
                self.key_cache[on_chip_layers:, :, dst_s:dst_e].copy_(k_cpu, non_blocking=True)
                self.value_cache[on_chip_layers:, :, dst_s:dst_e].copy_(v_cpu, non_blocking=True)

        self.graph_postfill_len = new_post

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.init_graph = False
        self.graph_postfill_len = 0

    def normal_(self):
        self.key_cache.normal_()
        self.value_cache.normal_()