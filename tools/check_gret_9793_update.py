"""Replay the entire captured update across eight ranks without taking an optimizer step."""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.model import build_model
from runtime.precision import autocast
from runtime.train import Tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    try:
        folder = args.run / "baseline"
        saved = torch.load(folder / "nonfinite_forward_00009793_rank1.pt", map_location="cpu",
                           weights_only=False, mmap=True)
        config = saved["config"]
        t, e = config["training"], config["execution"]
        world, rank = dist.get_world_size(), dist.get_rank()
        assert world == t["world_size"] and not e["compile"] and e["deterministic_algorithms"]
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = e["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = e["allow_tf32"]
        model = build_model(config, checkpoint_activations=e["checkpoint_activations"]).cuda().train()
        model.load_state_dict(saved["model"])
        ddp = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False,
                                      find_unused_parameters=e["ddp_find_unused_parameters"],
                                      gradient_as_bucket_view=True)
        manifest = json.loads((folder / "resolved_config.json").read_text())["data"]
        tokens = Tokens(manifest, "train", t["sequence_length"], config["seed"])
        scale = saved["scaler"]["scale"]
        accum, micro = t["gradient_accumulation"], t["microbatch"]
        losses = []
        matched = False
        for j in range(accum):
            index = ((saved["step"] * accum + j) * world + rank) * micro
            batch = torch.stack([tokens.sample(index + i) for i in range(micro)])
            if (rank, j) == (saved["rank"], saved["microbatch"]):
                matched = (torch.equal(batch[:, :-1], saved["input_tokens"])
                           and torch.equal(batch[:, 1:], saved["targets"]))
                assert matched
            batch = batch.cuda()
            with ddp.no_sync() if j < accum - 1 else contextlib.nullcontext():
                with autocast(t["precision"]):
                    logits = ddp(batch[:, :-1])[0]
                    loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].reshape(-1))
                assert torch.isfinite(logits).all() and torch.isfinite(loss)
                (loss / accum * scale).backward()
                losses.append(float(loss.detach()))
                del logits, loss
        norms = []
        for parameter in model.parameters():
            assert parameter.grad is not None
            gradient = parameter.grad.detach() / scale
            norms.append(torch.linalg.vector_norm(gradient))
        norm = float(torch.linalg.vector_norm(torch.stack(norms)))
        assert torch.isfinite(torch.tensor(norm))
        reports = [None] * world
        dist.all_gather_object(reports, dict(rank=rank, microbatch_losses=losses, preclip_norm=norm,
                                             tensors=len(norms), captured_tokens_matched=matched))
        if rank == 0:
            assert sum(row["captured_tokens_matched"] for row in reports) == 1
            assert all(row["preclip_norm"] == norm for row in reports)
            mean = sum(sum(row["microbatch_losses"]) for row in reports) / (world * accum)
            report = dict(update=saved["failure_update"], scale=scale, global_samples=world * accum * micro,
                          global_loss=mean, ranks=reports, optimizer_steps=0, passed=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
