#!/usr/bin/env python3
"""Single-GPU long-context inference benchmark for paired YOCO checkpoints.

This follows the released YOCO generation flow: prefill all but the last prompt
token, let YOCO skip its cross-decoder during prefill, then decode recurrently.
The benchmark is intended for architecture profiling. Contexts above the 2K
training length are synthetic efficiency tests, not quality measurements.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(2048, 4096, 8192, 16384, 32768)
    )
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
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


def load_model(config_path: Path, checkpoint: Path, max_seq_len: int, device: torch.device):
    from runtime.model import architecture, build_model

    config = json.loads(config_path.read_text())
    config["model"]["max_seq_len"] = max_seq_len
    config["model"]["max_batch_size"] = 1
    model = build_model(config, checkpoint_activations=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.half().to(device).eval(), architecture(config), config["model"]["vocab_size"]


@torch.inference_mode()
def run_once(
    kind: str,
    model,
    prompt: torch.Tensor,
    decode_tokens: int,
    device: torch.device,
) -> dict:
    state: dict = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)

    torch.cuda.synchronize(device)
    prefill_start = time.perf_counter()
    if kind == "yoco":
        model(
            prompt[:, :-1],
            incremental_state=state,
            start_pos=0,
            skip_cross_decoder=True,
            is_prefilling=True,
        )
    else:
        model.forward_partial(prompt[:, :-1], start_pos=0, incremental_state=state)
    torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start
    prefill_peak = torch.cuda.max_memory_allocated(device)
    cache_bytes_after_prefill = tensor_bytes(state)

    current = prompt[:, -1:]
    torch.cuda.synchronize(device)
    decode_start = time.perf_counter()
    for index in range(decode_tokens):
        position = prompt.shape[1] - 1 + index
        if kind == "yoco":
            logits, _ = model(
                current,
                incremental_state=state,
                start_pos=position,
                skip_cross_decoder=False,
                is_prefilling=False,
            )
        else:
            logits, _ = model(current, start_pos=position, incremental_state=state)
        current = logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    total_peak = torch.cuda.max_memory_allocated(device)
    cache_bytes_after_decode = tensor_bytes(state)

    return {
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": (prompt.shape[1] - 1) / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_tokens / decode_seconds,
        "end_to_end_tokens_per_second": (prompt.shape[1] - 1 + decode_tokens)
        / (prefill_seconds + decode_seconds),
        "baseline_allocated_bytes": baseline_allocated,
        "prefill_peak_allocated_bytes": prefill_peak,
        "total_peak_allocated_bytes": total_peak,
        "cache_bytes_after_prefill": cache_bytes_after_prefill,
        "cache_bytes_after_decode": cache_bytes_after_decode,
    }


def median_record(records: list[dict]) -> dict:
    result = {}
    for key in records[0]:
        values = sorted(record[key] for record in records)
        result[key] = values[len(values) // 2]
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    results = []

    for length in args.lengths:
        max_seq_len = length + args.decode_tokens
        try:
            model, kind, vocab_size = load_model(args.config, args.checkpoint, max_seq_len, device)
            prompt = torch.randint(0, vocab_size, (1, length), device=device)

            # Untimed backend and GPU-clock warmup at a small shape.
            warmup = torch.randint(0, vocab_size, (1, 256), device=device)
            for _ in range(3):
                run_once(kind, model, warmup, 4, device)
            del warmup
            gc.collect()
            torch.cuda.empty_cache()

            # Warm the exact attention shape before collecting steady-state timings.
            run_once(kind, model, prompt, min(8, args.decode_tokens), device)
            gc.collect()
            torch.cuda.empty_cache()

            repeats = [
                run_once(kind, model, prompt, args.decode_tokens, device)
                for _ in range(args.repeats)
            ]
            record = {
                "model": kind,
                "context_length": length,
                "decode_tokens": args.decode_tokens,
                "status": "ok",
                **median_record(repeats),
                "repeats": repeats,
            }
        except torch.cuda.OutOfMemoryError as exc:
            record = {
                "model": kind,
                "context_length": length,
                "decode_tokens": args.decode_tokens,
                "status": "oom",
                "error": str(exc),
            }
        results.append(record)
        print(json.dumps(record), flush=True)
        if "model" in locals():
            del model
        if "prompt" in locals():
            del prompt
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "model": kind,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": "float16",
        "batch_size": 1,
        "trained_context_length": 2048,
        "context_above_training_length_is_efficiency_only": True,
        "prefill_definition": "all prompt tokens except the last; YOCO skips cross-decoder as in official generation code",
        "decode_definition": "last prompt token followed by recurrent greedy decoding",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
