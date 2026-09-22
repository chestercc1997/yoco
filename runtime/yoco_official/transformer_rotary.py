"""PyTorch equivalent of the official Transformer's non-interleaved RoPE."""
import torch


def apply_rotary_emb(x, cos, sin):
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin,
                      first * sin + second * cos), dim=-1)
