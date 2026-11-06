"""Single model-parallel rank adapters; architecture stays in official modules.

The PyTorch chunk recurrence implements the released equations without a
Volta-incompatible Triton kernel. Accumulated gates/states use FP32 in mixed
precision. This is an explicit numerical backend deviation, not original logs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ColumnParallelLinear(nn.Linear):
    def __init__(self, input_size, output_size, bias=True, init_method=None, **kwargs):
        super().__init__(input_size, output_size, bias=bias)
        if init_method:
            init_method(self.weight)


RowParallelLinear = ColumnParallelLinear


class VocabParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, init_method=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        if init_method:
            init_method(self.weight)


def copy_to_model_parallel_region(x):
    return x


def checkpoint_wrapper(layer):
    original = layer.forward
    def forward(*args, **kwargs):
        if layer.training and torch.is_grad_enabled():
            return checkpoint(original, *args, use_reentrant=False, **kwargs)
        return original(*args, **kwargs)
    layer.forward = forward
    return layer


def swiglu(x, y):
    return F.silu(x) * y


def apply_rotary_emb(x, cos, sin, interleaved=True, **kwargs):
    if not interleaved:
        raise NotImplementedError('Only the official interleaved YOCO path is supported')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), **kwargs):
    # SDPA memory-efficient backend supports Volta; math is a portable fallback.
    if q.shape[2] != k.shape[2]:
        repeats = q.shape[2] // k.shape[2]
        k, v = k.repeat_interleave(repeats, 2), v.repeat_interleave(repeats, 2)
    q, k, v = [a.transpose(1, 2).contiguous() for a in (q, k, v)]
    mask = None
    simple_causal = causal and q.shape[-2] == k.shape[-2] and window_size == (-1, -1)
    if not simple_causal:
        # Bottom-right alignment also handles incremental single-token queries.
        qi = torch.arange(q.shape[-2], device=q.device) + k.shape[-2] - q.shape[-2]
        ki = torch.arange(k.shape[-2], device=q.device)
        allowed = ki[None, :] <= qi[:, None] if causal else torch.ones((len(qi), len(ki)), device=q.device, dtype=torch.bool)
        if window_size[0] >= 0:
            allowed = allowed & (ki[None, :] >= qi[:, None] - window_size[0])
        mask = allowed
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=simple_causal).transpose(1, 2).contiguous()


def chunk_gate_retention(q, k, v, g, chunk_size=256, last_hidden_state=None):
    b, h, length, dk = q.shape
    if length % chunk_size:
        raise ValueError('Official chunk retention requires padded chunk boundaries')
    n = length // chunk_size
    qr = q.reshape(b, h, n, chunk_size, dk)
    kr = k.reshape(b, h, n, chunk_size, dk) * dk ** -0.5
    vr = v.reshape(b, h, n, chunk_size, v.shape[-1])
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    gs = g.reshape(b, h, n, chunk_size).to(acc_dtype).cumsum(-1)
    lower = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool).tril()
    decay = (gs[..., :, None] - gs[..., None, :]).masked_fill(~lower, -torch.inf).exp()
    inner = ((qr @ kr.transpose(-1, -2)) * decay.to(q.dtype)) @ vr
    contributions = kr.transpose(-1, -2) @ (vr * (gs[..., -1, None] - gs).exp().to(v.dtype)[..., None])
    state = (torch.zeros((b, h, dk, v.shape[-1]), device=q.device, dtype=acc_dtype)
             if last_hidden_state is None else last_hidden_state.to(acc_dtype))
    states = []
    for i in range(n):
        states.append(state.to(q.dtype))
        state = state * gs[:, :, i, -1].exp()[..., None, None] + contributions[:, :, i].to(acc_dtype)
    # The recurrent states are FP32 already. AMP must not narrow their
    # cross-chunk result to FP16 before RMSNorm: finite values above 65504
    # otherwise become Inf and normalize to NaN. Keep this accumulation and
    # the final sum in the existing state dtype, then normalize before casting.
    with torch.autocast(device_type=q.device.type, enabled=False):
        cross = (qr.to(acc_dtype) * gs.exp()[..., None]) @ torch.stack(states, dim=2).to(acc_dtype)
        output = inner.to(acc_dtype) + cross
    return output.reshape(b, h, length, v.shape[-1])


def recurrent_gate_retention(q, k, v, g, incremental_state):
    state = incremental_state.get('last_hidden_state')
    kv = k.transpose(-1, -2).float() * (q.shape[-1] ** -0.5) * v.float()
    if state is not None:
        kv = kv + state.float() * g.float().exp()[..., None]
    incremental_state['last_hidden_state'] = kv
    return q @ kv.to(q.dtype)
