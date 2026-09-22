#!/usr/bin/env python3
"""Deterministic dynamic-shape stress test for YOCO and its paired baseline."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist

from benchmark_2k_distributed_inference import (
    barrier,
    load_model,
    run_once,
    setup_distributed,
)


# Covers short/medium/near-limit prompts and short/long generation bursts.  The
# order is intentionally non-monotonic to exercise allocator and kernel paths.
DEFAULT_TRACE = (
    (128, 16),
    (1024, 32),
    (256, 64),
    (1792, 16),
    (512, 32),
    (2048, 64),
    (768, 16),
    (1536, 64),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-per-gpu", type=int, default=1)
    parser.add_argument("--steady-passes", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_rates(record: dict, world_size: int, batch_size: int, context: int, decode: int) -> dict:
    requests = world_size * batch_size
    output = dict(record)
    output.update({
        "context_length": context,
        "decode_tokens": decode,
        "aggregate_prefill_tokens_per_second": requests * (context - 1) / record["prefill_seconds"],
        "aggregate_decode_tokens_per_second": requests * decode / record["decode_seconds"],
        "aggregate_end_to_end_tokens_per_second": requests * (context - 1 + decode) / record["total_seconds"],
    })
    return output


def workload_summary(records: list[dict], world_size: int, batch_size: int) -> dict:
    requests = world_size * batch_size
    prefill_tokens = requests * sum(item["context_length"] - 1 for item in records)
    decode_tokens = requests * sum(item["decode_tokens"] for item in records)
    prefill_seconds = sum(item["prefill_seconds"] for item in records)
    decode_seconds = sum(item["decode_seconds"] for item in records)
    return {
        "trace_requests": requests * len(records),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
        "aggregate_prefill_tokens_per_second": prefill_tokens / prefill_seconds,
        "aggregate_decode_tokens_per_second": decode_tokens / decode_seconds,
        "aggregate_end_to_end_tokens_per_second": (prefill_tokens + decode_tokens) / (prefill_seconds + decode_seconds),
        "max_cache_bytes_per_request": max(
            item["cache_bytes_after_decode_per_gpu_max"] / batch_size for item in records
        ),
        "max_peak_allocated_bytes_per_gpu": max(
            item["total_peak_allocated_bytes_per_gpu_max"] for item in records
        ),
    }


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260922 + rank)
    max_seq_len = max(context + decode for context, decode in DEFAULT_TRACE)
    model, config, kind = load_model(args.config, args.checkpoint, max_seq_len, args.batch_per_gpu, device)
    vocab_size = int(config["model"].get("vocab_size", 100_288))
    prompts = [
        torch.randint(0, vocab_size, (args.batch_per_gpu, context), device=device)
        for context, _ in DEFAULT_TRACE
    ]

    # First encounter of each shape, in the published deterministic order.
    cold = []
    for prompt, (context, decode) in zip(prompts, DEFAULT_TRACE):
        cold.append(add_rates(
            run_once(kind, model, prompt, decode, device),
            world_size, args.batch_per_gpu, context, decode,
        ))

    # Revisit the identical trace and take each shape's median steady result.
    passes: list[list[dict]] = []
    for _ in range(args.steady_passes):
        current_pass = []
        for prompt, (context, decode) in zip(prompts, DEFAULT_TRACE):
            current_pass.append(run_once(kind, model, prompt, decode, device))
        passes.append(current_pass)
    steady = []
    for shape_index, (context, decode) in enumerate(DEFAULT_TRACE):
        samples = [pass_records[shape_index] for pass_records in passes]
        median = {
            key: sorted(sample[key] for sample in samples)[len(samples) // 2]
            for key in samples[0]
        }
        steady.append(add_rates(
            median, world_size, args.batch_per_gpu, context, decode,
        ))

    cold_summary = workload_summary(cold, world_size, args.batch_per_gpu)
    steady_summary = workload_summary(steady, world_size, args.batch_per_gpu)
    steady_summary["cold_to_steady_time_ratio"] = (
        cold_summary["total_seconds"] / steady_summary["total_seconds"]
    )
    if rank == 0:
        output = {
            "model": kind,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "device": torch.cuda.get_device_name(device),
            "dtype": "float16",
            "world_size": world_size,
            "batch_per_gpu": args.batch_per_gpu,
            "parallelism": "independent full-model replicas (data-parallel serving)",
            "trace_seed": 20260922,
            "trace": [
                {"context_length": context, "decode_tokens": decode}
                for context, decode in DEFAULT_TRACE
            ],
            "cold_first_pass": {"summary": cold_summary, "shapes": cold},
            "steady_state": {"summary": steady_summary, "shapes": steady, "passes": args.steady_passes},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output), flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    del model, prompts
    gc.collect()


if __name__ == "__main__":
    main()
