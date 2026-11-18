"""Compare one plain and fused AdamW update on copies of a 400M checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.model import build_model
from runtime.precision import autocast
from runtime.train import Tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    resolved = json.loads((args.checkpoint.parent / "resolved_config.json").read_text())
    config = resolved["config"]
    training = config["training"]
    precision = training["precision"]
    assert precision == "fp16_mixed_fp32_params"
    torch.backends.cuda.matmul.allow_tf32 = config["execution"]["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["execution"]["allow_tf32"]
    with args.checkpoint.open("rb") as file:
        sha = hashlib.file_digest(file, "sha256").hexdigest()
    assert sha == json.loads(args.checkpoint.with_suffix(".json").read_text())["sha256"]
    saved = torch.load(args.checkpoint, mmap=True, map_location="cpu", weights_only=False)
    model = build_model(config).cuda().train()
    model.load_state_dict(saved["model"])
    tokens = Tokens(resolved["data"], "train", training["sequence_length"], config["seed"])
    micro = training["microbatch"]
    accum = training["gradient_accumulation"]
    world = training["world_size"]
    count = micro * accum * world
    first = saved["step"] * count
    update = saved["step"] + 1
    assert training["warmup_updates"] < update <= training["max_updates"]
    lr = training["end_lr"] + (training["lr"] - training["end_lr"]) * (
        training["max_updates"] - update) / (training["max_updates"] - training["warmup_updates"])

    def loss_at(index):
        batch = torch.stack([tokens.sample(i) for i in range(index, index + micro)]).cuda()
        with autocast(precision):
            logits = model(batch[:, :-1])[0]
            return F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].reshape(-1))

    # Hold the gradient fixed between both optimizer implementations. This
    # one-device sum covers all samples but has a different order from DDP.
    scale = saved["scaler"]["scale"]
    for index in range(first, first + count, micro):
        (loss_at(index) / accum * scale).backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(scale * world)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"],
                                                error_if_nonfinite=True))
    ref = {}
    differences = {}
    results = {}
    for fused in (False, True):
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.AdamW(model.parameters(), fused=fused)
        # Loading a checkpoint normally restores the old 'fused' and LR
        # flags too. Set them on a private parameter-group copy before load.
        state = saved["optimizer"]
        groups = [dict(group, fused=fused, lr=lr) for group in state["param_groups"]]
        # The plain kernel can keep its step counter on CPU. Clone it
        # so the first update cannot mutate the mmap-backed source counter.
        moments = {key: dict(values, step=values["step"].clone())
                   for key, values in state["state"].items()}
        optimizer.load_state_dict(dict(state=moments, param_groups=groups))
        assert all(group["fused"] is fused for group in optimizer.param_groups)
        optimizer.step()
        with torch.no_grad():
            results["fused" if fused else "plain"] = float(loss_at(first + count))
            for name, p in model.named_parameters():
                tensors = {"param": p, **optimizer.state[p]}
                for field, value in tensors.items():
                    if not torch.isfinite(value).all():
                        raise RuntimeError(f"Nonfinite {name}.{field} with fused={fused}")
                    if not fused:
                        ref[name, field] = value.detach().cpu().clone()
                    else:
                        delta = float((value - ref.pop((name, field)).to(value.device)).abs().max())
                        differences[field] = max(differences.get(field, 0.0), delta)
        del optimizer
    assert not ref and all(math.isfinite(loss) for loss in results.values())
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha, update=update,
                  global_samples=count, single_gpu_accumulation=True, precision=precision,
                  scale=scale, learning_rate=lr, preclip_grad_norm=norm,
                  next_microbatch_loss=results, max_absolute_differences=differences,
                  all_parameters_and_optimizer_states_finite=True, checkpoints_written=0)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
