"""Check the captured 400M failure with activation checkpointing on and off.

Run once per runtime root so imports cannot mix the frozen and current code.
The snapshot is diagnostic only. No optimizer is constructed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


def replay(build_model, saved, checkpoint_activations):
    config = saved["config"]
    model = build_model(config, checkpoint_activations=checkpoint_activations).cuda().train()
    model.load_state_dict(saved["model"])
    result = {"checkpoint_activations": checkpoint_activations}

    def inspect_norm(module, inputs):
        tensor = inputs[0].detach()
        result["first_norm_input_nonfinite"] = int((~torch.isfinite(tensor)).sum())

    hook = model.self_decoder.layers[0].mixer.subln.register_forward_pre_hook(inspect_norm)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(saved["input_tokens"].cuda())[0]
        loss = F.cross_entropy(logits.flatten(0, 1), saved["targets"].cuda().flatten())
    hook.remove()
    result["logits_dtype"] = str(logits.dtype)
    result["logits_nonfinite"] = int((~torch.isfinite(logits.detach())).sum())
    result["loss"] = str(float(loss.detach()))
    if torch.isfinite(loss):
        scale = saved["scaler"]["scale"]
        (loss / config["training"]["gradient_accumulation"] * scale).backward()
        norms = []
        bad = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                norm = torch.linalg.vector_norm(parameter.grad.detach() / scale)
                norms.append(norm)
                if not torch.isfinite(norm):
                    bad.append(name)
        result["gradient_tensors"] = len(norms)
        result["nonfinite_gradient_norms"] = bad
        result["unscaled_microbatch_grad_norm"] = float(torch.stack(norms).norm())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expect", choices=["reproduce", "fixed"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_root.resolve()))
    from runtime.model import build_model

    torch.set_num_threads(2)
    saved = torch.load(args.snapshot, mmap=True, map_location="cpu", weights_only=False)
    config = saved["config"]
    if config["training"]["precision"] != "fp16_mixed_fp32_params":
        raise ValueError("Expected the frozen mixed-FP16 configuration")
    torch.backends.cuda.matmul.allow_tf32 = config["execution"]["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["execution"]["allow_tf32"]
    backend = args.runtime_root / "runtime/yoco_official/backend.py"
    with args.snapshot.open("rb") as f:
        snapshot_sha = hashlib.file_digest(f, "sha256").hexdigest()
    report = {
        "snapshot_sha256": snapshot_sha,
        "backend_sha256": hashlib.sha256(backend.read_bytes()).hexdigest(),
        "snapshot": str(args.snapshot), "runtime_root": str(args.runtime_root),
        "update": saved["failure_update"], "rank": saved["rank"],
        "microbatch": saved["microbatch"], "loss_scale": saved["scaler"]["scale"],
        "precision": config["training"]["precision"], "compile": False,
        "optimizer_steps": 0, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "cases": [replay(build_model, saved, flag) for flag in (False, True)],
    }
    if args.expect == "fixed":
        passed = all(row["first_norm_input_nonfinite"] == row["logits_nonfinite"] == 0
                     and row["logits_dtype"] == "torch.float16"
                     and row.get("gradient_tensors", 0) > 0 and not row["nonfinite_gradient_norms"]
                     for row in report["cases"])
    else:
        passed = all(row["first_norm_input_nonfinite"] > 0 and row["logits_nonfinite"] > 0
                     and row["loss"] == "nan" for row in report["cases"])
    report["passed"] = passed
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise SystemExit("Snapshot replay differed from expectation")


if __name__ == "__main__":
    main()
