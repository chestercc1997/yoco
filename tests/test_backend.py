from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from runtime.yoco_official.backend import chunk_gate_retention, flash_attn_func, apply_rotary_emb


def main():
    torch.manual_seed(7)
    # Compare an independent token-by-token recurrence including gradients.
    q, k, v = [torch.randn(2, 3, 12, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    g = (-torch.rand(2, 3, 12, dtype=torch.float64)).requires_grad_()
    initial = torch.randn(2, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    fast = chunk_gate_retention(q, k, v, g, chunk_size=4, last_hidden_state=initial)
    state, parts = initial, []
    for i in range(12):
        state = state * g[:, :, i].exp()[..., None, None] + k[:, :, i, :, None] * v[:, :, i, None, :] / 2
        parts.append((q[:, :, i, None, :] @ state).squeeze(-2))
    slow = torch.stack(parts, 2)
    torch.testing.assert_close(fast, slow, atol=1e-10, rtol=1e-10)
    a = torch.autograd.grad(fast.square().sum(), (q,k,v,g,initial), retain_graph=True)
    b = torch.autograd.grad(slow.square().sum(), (q,k,v,g,initial))
    for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-9)
    q = torch.randn(1, 7, 4, 8); k = torch.randn(1, 7, 2, 8); v = torch.randn_like(k)
    out = flash_attn_func(q,k,v)
    changed = v.clone(); changed[:,4:] += 100
    torch.testing.assert_close(out[:,:4],flash_attn_func(q,k,changed)[:,:4])
    torch.testing.assert_close(flash_attn_func(q[:,-1:],k,v),out[:,-1:])
    cos=torch.ones(7,4); sin=torch.zeros(7,4)
    torch.testing.assert_close(apply_rotary_emb(q,cos,sin),q)
    print('PASS: chunk recurrence and gradients, causal GQA, decode alignment, rotary identity')

if __name__ == '__main__': main()
