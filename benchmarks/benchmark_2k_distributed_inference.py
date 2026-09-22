#!/usr/bin/env python3
"""Measure replicated 2K autoregressive inference from 1 to 32 GPUs.

Each rank owns one complete model replica and an independent request batch.  The
reported distributed latency is the slowest rank and aggregate throughput is
the sum of useful tokens across all ranks divided by that latency.  This is a
serving/data-parallel benchmark; it does not claim that 32 GPUs shorten one
request's latency.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--batch-per-gpu", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_bytes(value, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        storage = value.untyped_storage()
        pointer = storage.data_ptr()
        if pointer in seen:
            return 0
        seen.add(pointer)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item, seen) for item in value)
    return 0


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def global_max(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def load_model(config_path: Path, checkpoint: Path, max_seq_len: int, batch_size: int, device: torch.device):
    from runtime.model import architecture, build_model

    config = json.loads(config_path.read_text())
    config["model"]["max_seq_len"] = max_seq_len
    config["model"]["max_batch_size"] = batch_size
    model = build_model(config, checkpoint_activations=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.half().to(device).eval(), config, architecture(config)


@torch.inference_mode()
def run_once(kind: str, model, prompt: torch.Tensor, decode_tokens: int, device: torch.device) -> dict:
    state: dict = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)

    barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if kind == "yoco":
        model(
            prompt[:, :-1], incremental_state=state, start_pos=0,
            skip_cross_decoder=True, is_prefilling=True,
        )
    else:
        model.forward_partial(prompt[:, :-1], start_pos=0, incremental_state=state)
    torch.cuda.synchronize(device)
    local_prefill = time.perf_counter() - started
    cache_after_prefill = tensor_bytes(state)
    prefill_peak = torch.cuda.max_memory_allocated(device)

    current = prompt[:, -1:]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index in range(decode_tokens):
        position = prompt.shape[1] - 1 + index
        if kind == "yoco":
            logits, _ = model(
                current, incremental_state=state, start_pos=position,
                skip_cross_decoder=False, is_prefilling=False,
            )
        else:
            logits, _ = model(current, start_pos=position, incremental_state=state)
        current = logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    local_decode = time.perf_counter() - started

    # A distributed serving batch completes when its slowest replica completes.
    prefill_seconds = global_max(local_prefill, device)
    decode_seconds = global_max(local_decode, device)
    peak_bytes = global_max(float(torch.cuda.max_memory_allocated(device)), device)
    baseline_bytes = global_max(float(baseline_allocated), device)
    prefill_peak_bytes = global_max(float(prefill_peak), device)
    cache_prefill_bytes = global_max(float(cache_after_prefill), device)
    cache_decode_bytes = global_max(float(tensor_bytes(state)), device)
    return {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
        "baseline_allocated_bytes_per_gpu_max": int(baseline_bytes),
        "prefill_peak_allocated_bytes_per_gpu_max": int(prefill_peak_bytes),
        "total_peak_allocated_bytes_per_gpu_max": int(peak_bytes),
        "cache_bytes_after_prefill_per_gpu_max": int(cache_prefill_bytes),
        "cache_bytes_after_decode_per_gpu_max": int(cache_decode_bytes),
    }


def median_record(records: list[dict]) -> dict:
    return {
        key: sorted(record[key] for record in records)[len(records) // 2]
        for key in records[0]
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(42 + rank)
    model, config, kind = load_model(
        args.config, args.checkpoint, args.context_length + args.decode_tokens, args.batch_per_gpu, device
    )
    vocab_size = int(config["model"].get("vocab_size", 100_288))
    prompt = torch.randint(
        0, vocab_size, (args.batch_per_gpu, args.context_length), device=device
    )

    # Warm both kernels and the exact attention shape before measurement.
    warmup = torch.randint(0, vocab_size, (args.batch_per_gpu, 256), device=device)
    run_once(kind, model, warmup, min(4, args.decode_tokens), device)
    del warmup
    gc.collect()
    torch.cuda.empty_cache()
    run_once(kind, model, prompt, min(8, args.decode_tokens), device)
    gc.collect()
    torch.cuda.empty_cache()

    records = [
        run_once(kind, model, prompt, args.decode_tokens, device)
        for _ in range(args.repeats)
    ]
    median = median_record(records)
    total_requests = world_size * args.batch_per_gpu
    prefill_tokens = total_requests * (args.context_length - 1)
    decode_tokens = total_requests * args.decode_tokens
    median.update({
        "aggregate_prefill_tokens_per_second": prefill_tokens / median["prefill_seconds"],
        "aggregate_decode_tokens_per_second": decode_tokens / median["decode_seconds"],
        "aggregate_end_to_end_tokens_per_second": (
            prefill_tokens + decode_tokens
        ) / median["total_seconds"],
        "requests_per_second": total_requests / median["total_seconds"],
        "cache_bytes_after_prefill_per_request": (
            median["cache_bytes_after_prefill_per_gpu_max"] / args.batch_per_gpu
        ),
        "cache_bytes_after_decode_per_request": (
            median["cache_bytes_after_decode_per_gpu_max"] / args.batch_per_gpu
        ),
    })

    if rank == 0:
        output = {
            "model": kind,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "dtype": "float16",
            "world_size": world_size,
            "batch_per_gpu": args.batch_per_gpu,
            "concurrent_requests": total_requests,
            "context_length": args.context_length,
            "decode_tokens": args.decode_tokens,
            "host": socket.gethostname(),
            "parallelism": "independent full-model replicas (data-parallel serving)",
            "latency_aggregation": "maximum across ranks",
            "result": median,
            "repeats": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output), flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
