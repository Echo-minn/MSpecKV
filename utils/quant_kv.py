import torch
from typing import List, Tuple

PastKeyValues = List[Tuple[torch.Tensor, torch.Tensor]]  # per layer (K, V)

def quantize_tensor(x: torch.Tensor, num_bits: int = 8):
    """
    Very simple per-tensor symmetric quantization.
    x: (..., dim)
    Returns:
        q: int8 tensor
        scale: broadcastable scale tensor
    """
    # max over last dim (or everything) – you can refine this
    max_val = x.abs().amax(dim=-1, keepdim=True) + 1e-6
    qmax = 2**(num_bits - 1) - 1
    scale = max_val / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int8)
    return q, scale


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = None):
    """Dequantize tensor, optionally converting to specified dtype."""
    result = q.float() * scale
    if dtype is not None:
        result = result.to(dtype)
    return result


def quantize_past_kv(
    past_key_values: PastKeyValues,
    num_bits: int = 8
):
    """
    past_key_values: list of (K, V) per layer
        K, V: [batch, num_heads, seq_len, head_dim]
    Returns:
        quantized_kv: list of ((K_q, K_scale), (V_q, V_scale))
    """
    quantized = []
    for K, V in past_key_values:
        K_q, K_scale = quantize_tensor(K, num_bits=num_bits)
        V_q, V_scale = quantize_tensor(V, num_bits=num_bits)
        quantized.append(((K_q, K_scale), (V_q, V_scale)))
    return quantized


def dequantize_past_kv(
    quantized_kv,
    dtype: torch.dtype = None
) -> PastKeyValues:
    """
    quantized_kv: list of ((K_q, K_scale), (V_q, V_scale))
    dtype: target dtype for dequantized tensors (e.g., torch.float16)
    """
    deq = []
    for (K_q, K_scale), (V_q, V_scale) in quantized_kv:
        K_fp = dequantize_tensor(K_q, K_scale, dtype=dtype)
        V_fp = dequantize_tensor(V_q, V_scale, dtype=dtype)
        deq.append((K_fp, V_fp))
    return deq