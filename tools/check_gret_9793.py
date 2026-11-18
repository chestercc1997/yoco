"""Validate the 830M recurrent-state write using the captured failure and a healthy control."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.model import build_model
from runtime.yoco_official import backend, gate_retention


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def first_write(q, k, v, g, chunk_size=256):
    b, h, t, d = k.shape
    keys = k.reshape(b, h, -1, chunk_size, d) * d ** -0.5
    values = v.reshape(b, h, -1, chunk_size, v.size(-1))
    gates = g.reshape(b, h, -1, chunk_size).float().cumsum(-1)
    weighted = values * (gates[..., -1, None] - gates).exp().to(v.dtype)[..., None]
    result = keys.transpose(-1, -2) @ weighted
    assert all(torch.isfinite(x).all() for x in (q, k, v, g, weighted))
    bad = (~torch.isfinite(result)).nonzero()
    assert len(bad) == 1
    batch, head, chunk, key, value = bad[0].tolist()
    left = keys[batch, head, chunk, :, key].to(v.dtype).double()
    right = weighted[batch, head, chunk, :, value].double()
    oracle = float(torch.dot(left, right))
    repaired = backend._retention_sum(keys.transpose(-1, -2), weighted, torch.float32)
    actual = float(repaired[batch, head, chunk, key, value])
    mask = torch.isfinite(result)
    assert torch.equal(repaired[mask], result[mask].float())
    assert abs(oracle) > torch.finfo(torch.float16).max and abs(actual - oracle) < 1
    return dict(bad_elements=len(bad), index=bad[0].tolist(), fp64_dot=oracle,
                repaired_dot=actual, finite_elements_unchanged=True)


def forward(model, x, target):
    logits = model(x)[0]
    return logits, F.cross_entropy(logits.flatten(0, 1), target.flatten())


def backward(model, x, target, multiplier):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.float16):
        logits, loss = forward(model, x, target)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    (loss * multiplier).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return float(loss.detach())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    folder = args.run / "baseline"
    snapshot = folder / "nonfinite_forward_00009793_rank1.pt"
    saved = torch.load(snapshot, map_location="cpu", mmap=True, weights_only=False)
    config = saved["config"]
    assert config["training"]["precision"] == "fp16_mixed_fp32_params"
    torch.backends.cuda.matmul.allow_tf32 = config["execution"]["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["execution"]["allow_tf32"]
    old_path = args.run / "release/runtime/yoco_official/backend.py"
    spec = importlib.util.spec_from_file_location("frozen_backend", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    model = build_model(config).cuda().train()
    model.load_state_dict(saved["model"])
    x, target = saved["input_tokens"].cuda(), saved["targets"].cuda()
    scale = saved["scaler"]["scale"]
    multiplier = scale / config["training"]["gradient_accumulation"]
    first = {}

    def observed(*inputs, **kwargs):
        out = old.chunk_gate_retention(*inputs, **kwargs)
        if not first:
            first.update(first_write(*inputs, **kwargs))
            first["old_output_nonfinite"] = int((~torch.isfinite(out)).sum())
        return out

    with patch.object(gate_retention, "chunk_gate_retention", observed):
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            old_logits, old_loss = forward(model, x, target)
            assert not torch.isfinite(old_loss)
            del old_logits
    fixed_loss = backward(model, x, target, multiplier)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        ref_logits, ref_loss = forward(model, x, target)
        ref_loss = float(ref_loss)
        del ref_logits
    assert abs(fixed_loss - ref_loss) < 0.01
    source = folder / "checks/resumed/checkpoint_00005004.pt"
    source_hash = sha(source)
    assert source_hash == json.loads(source.with_suffix(".json").read_text())["sha256"]
    healthy = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(healthy["model"])
    control_multiplier = healthy["scaler"]["scale"] / config["training"]["gradient_accumulation"]
    with patch.object(gate_retention, "chunk_gate_retention", old.chunk_gate_retention):
        old_control = backward(model, x, target, control_multiplier)
        old_grads = [p.grad.detach().cpu() for p in model.parameters()]
    control = backward(model, x, target, control_multiplier)
    assert control == old_control
    assert all(torch.equal(p.grad.detach().cpu(), ref) for p, ref in zip(model.parameters(), old_grads))
    report = dict(passed=True, snapshot=str(snapshot), snapshot_sha256=sha(snapshot),
                  failure_update=saved["failure_update"], microbatch=saved["microbatch"],
                  first_layer_state_write=first, scale=scale, frozen_loss=str(float(old_loss)),
                  fixed_loss=fixed_loss, fp32_tf32_reference_loss=ref_loss,
                  scaled_gradient_tensors_finite=len(old_grads), training_updates=0,
                  healthy_control=dict(step=healthy["step"], loss=control,
                                       all_gradients_bitwise_equal=True),
                  source_checkpoint=str(source), source_checkpoint_sha256=source_hash,
                  model_source_changes={"yoco_official/backend.py": {
                      "before":sha(old_path), "after":sha(ROOT/"runtime/yoco_official/backend.py")}})
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
