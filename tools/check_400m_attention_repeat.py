"""Repeat one captured attention backward at the first step after the historical 10k fork."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str((args.run / "release").resolve()))
    from runtime.model import build_model
    from runtime.train import Tokens
    from runtime.yoco_official import cross_attention

    torch.set_num_threads(2)
    resolved = json.loads((args.run / "resolved_config.json").read_text())
    config = resolved["config"]
    training = config["training"]
    torch.backends.cuda.matmul.allow_tf32 = config["execution"]["allow_tf32"]
    saved = torch.load(args.run / "checkpoint_00010000.pt", mmap=True, map_location="cpu", weights_only=False)
    model = build_model(config).cuda().train()
    model.load_state_dict(saved["model"])
    tokens = Tokens(resolved["data"], "train", training["sequence_length"], config["seed"])
    rank, micro = 3, training["microbatch"]
    start = (saved["step"] * training["gradient_accumulation"] * training["world_size"] + rank) * micro
    batch = torch.stack([tokens.sample(start + i) for i in range(micro)]).cuda()
    frozen = cross_attention.flash_attn_func
    captured = {}

    def capture(*operands, **kwargs):
        result = frozen(*operands, **kwargs)
        if not captured:
            captured.update(inputs=[x.detach() for x in operands], kwargs=kwargs)
            def save_grad(gradient):
                captured["upstream"] = gradient.detach()
            result.register_hook(save_grad)
        return result

    with patch.object(cross_attention, "flash_attn_func", capture):
        with torch.autocast("cuda", torch.float16):
            logits = model(batch[:, :-1])[0]
            loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].reshape(-1))
        (loss * (saved["scaler"]["scale"] / training["gradient_accumulation"])).backward()
    del model, logits, loss
    inputs = [x.requires_grad_() for x in captured["inputs"]]
    report = dict(update=saved["step"] + 1, rank=rank, microbatch=0, optimizer_updates=0,
                  layer="cross_decoder.layers.0.mixer", frozen_native_operator=True, cases=[])
    for deterministic in [False, True]:
        torch.use_deterministic_algorithms(deterministic)
        differences = [0.0, 0.0, 0.0]
        output_difference = 0.0
        for repeat in range(3):
            with torch.autocast("cuda", torch.float16):
                output = frozen(*inputs, **captured["kwargs"])
            grads = torch.autograd.grad(output, inputs, captured["upstream"])
            if repeat == 0:
                first = grads
                first_output = output.detach()
            else:
                output_difference = max(output_difference, float((output.detach() - first_output).abs().max()))
                for i, (a, b) in enumerate(zip(first, grads)):
                    differences[i] = max(differences[i], float((a - b).abs().max()))
        report["cases"].append(dict(deterministic=deterministic, repeats=3,
            max_output_difference=output_difference, max_grad_differences_qkv=differences))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
