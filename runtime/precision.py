"""Explicit training and evaluation arithmetic; legacy recipes default to AMP."""
import torch


def configure(precision):
    if precision not in {'fp16_mixed_fp32_params', 'fp32', 'fp32_tf32'}:
        raise ValueError(f'Unsupported precision: {precision}')
    if precision != 'fp16_mixed_fp32_params':
        torch.backends.cuda.matmul.allow_tf32 = precision == 'fp32_tf32'
        torch.backends.cudnn.allow_tf32 = precision == 'fp32_tf32'


def autocast(precision):
    return torch.autocast('cuda', dtype=torch.float16,
                          enabled=precision == 'fp16_mixed_fp32_params')
