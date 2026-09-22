#!/usr/bin/env python3
"""Create a quality/runtime/cache dashboard from completed YOCO experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/figures"
MODELS = ("transformer", "yoco")
COLORS = {"transformer": "#667085", "yoco": "#2E90FA"}
LABELS = {"transformer": "Transformer", "yoco": "YOCO"}
TASK_METRICS = {
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
    "sciq": "acc_norm,none",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def quality() -> dict[str, list[float]]:
    paths = {
        "transformer": ROOT / "evaluation/results/transformer_downstream_summary.json",
        "yoco": ROOT / "evaluation/results/yoco_downstream_summary.json",
    }
    result = {}
    for model, path in paths.items():
        raw = load(path)
        result[model] = [100 * raw[task][task][metric] for task, metric in TASK_METRICS.items()]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="directory containing *_w{1,8,32}_b1.json results")
    args = parser.parse_args()
    raw_dir = args.raw_dir
    OUT.mkdir(parents=True, exist_ok=True)
    fixed = {
        model: {world: load(raw_dir / f"{model}_w{world}_b1.json") for world in (1, 8, 32)}
        for model in MODELS
    }
    dynamic = {
        model: {world: load(raw_dir / f"{model}_dynamic_w{world}_b1.json") for world in (1, 32)}
        for model in MODELS
    }
    scores = quality()

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.dpi": 150})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle("YOCO vs Transformer: 2K Quality and V100 Inference Efficiency", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    x = np.arange(len(TASK_METRICS)); width = 0.38
    for offset, model in zip((-width / 2, width / 2), MODELS):
        ax.bar(x + offset, scores[model], width, color=COLORS[model], label=LABELS[model])
    ax.set_xticks(x, [name.replace("_", "\n") for name in TASK_METRICS], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Official lm-eval tasks (zero-shot, full test sets)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25)

    ax = axes[0, 1]
    worlds = np.array([1, 8, 32])
    for model in MODELS:
        values = [fixed[model][w]["result"]["aggregate_end_to_end_tokens_per_second"] for w in worlds]
        ax.plot(worlds, values, marker="o", linewidth=2.4, color=COLORS[model], label=LABELS[model])
    ax.set_xticks(worlds)
    ax.set_xlabel("V100 GPUs / concurrent requests")
    ax.set_ylabel("Aggregate tokens/s")
    ax.set_title("Fixed 2048 prompt + 64-token generation")
    ax.grid(alpha=.25); ax.legend(frameon=False)

    ax = axes[0, 2]
    categories = ["Prefill", "Decode", "End-to-end"]
    tr = fixed["transformer"][32]["result"]; yo = fixed["yoco"][32]["result"]
    speedups = [
        yo["aggregate_prefill_tokens_per_second"] / tr["aggregate_prefill_tokens_per_second"],
        yo["aggregate_decode_tokens_per_second"] / tr["aggregate_decode_tokens_per_second"],
        yo["aggregate_end_to_end_tokens_per_second"] / tr["aggregate_end_to_end_tokens_per_second"],
    ]
    bars = ax.bar(categories, speedups, color=["#12B76A", "#F79009", "#2E90FA"])
    ax.axhline(1, color="black", linewidth=1)
    ax.bar_label(bars, labels=[f"{v:.2f}×" for v in speedups], padding=3)
    ax.set_ylabel("YOCO / Transformer throughput")
    ax.set_title("32-GPU fixed-shape speedup")
    ax.grid(axis="y", alpha=.25)

    ax = axes[1, 0]
    x = np.arange(2); width = .36
    for offset, model in zip((-width / 2, width / 2), MODELS):
        values = [dynamic[model][w]["steady_state"]["summary"]["aggregate_end_to_end_tokens_per_second"] for w in (1, 32)]
        ax.bar(x + offset, values, width, color=COLORS[model], label=LABELS[model])
    ax.set_xticks(x, ["1 GPU", "32 GPUs"])
    ax.set_ylabel("Aggregate tokens/s")
    ax.set_title("Dynamic-shape trace: 8 prompt/generation patterns")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)

    ax = axes[1, 1]
    shapes = dynamic["yoco"][32]["steady_state"]["shapes"]
    tr_shapes = dynamic["transformer"][32]["steady_state"]["shapes"]
    labels = [f"{s['context_length']}+{s['decode_tokens']}" for s in shapes]
    values = [s["aggregate_end_to_end_tokens_per_second"] / t["aggregate_end_to_end_tokens_per_second"] for s, t in zip(shapes, tr_shapes)]
    bars = ax.bar(np.arange(len(labels)), values, color=["#12B76A" if v >= 1 else "#F79009" for v in values])
    ax.axhline(1, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("YOCO / Transformer")
    ax.set_title("32-GPU dynamic-shape end-to-end speedup")
    ax.grid(axis="y", alpha=.25)

    ax = axes[1, 2]
    cache = [fixed[m][32]["result"]["cache_bytes_after_decode_per_request"] / 2**20 for m in MODELS]
    peak = [fixed[m][32]["result"]["total_peak_allocated_bytes_per_gpu_max"] / 2**20 for m in MODELS]
    x = np.arange(2); width = .34
    b1 = ax.bar(x - width/2, cache, width, color="#7F56D9", label="KV/state cache MiB/request")
    b2 = ax.bar(x + width/2, peak, width, color="#98A2B3", label="Peak allocated MiB/GPU")
    ax.set_xticks(x, [LABELS[m] for m in MODELS])
    ax.bar_label(b1, fmt="%.1f", padding=3); ax.bar_label(b2, fmt="%.1f", padding=3)
    ax.set_ylabel("MiB")
    ax.set_title("Fixed 2K+64 memory at 32 GPUs")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.25)

    fig.text(.5, .012, "FP16, batch/GPU=1; latency uses slowest rank. 32 GPUs = 4 nodes × 8 V100, replicated serving.", ha="center", color="#475467")
    fig.tight_layout(rect=(0, .035, 1, .95))
    stem = OUT / "04_yoco_transformer_2k_32gpu_inference"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")

    rows = []
    for model in MODELS:
        for world in (1, 8, 32):
            d=fixed[model][world]; r=d["result"]
            rows.append({"workload":"fixed_2048_64","model":model,"gpus":world,"aggregate_end_to_end_tokens_s":r["aggregate_end_to_end_tokens_per_second"],"aggregate_prefill_tokens_s":r["aggregate_prefill_tokens_per_second"],"aggregate_decode_tokens_s":r["aggregate_decode_tokens_per_second"],"runtime_s":r["total_seconds"],"cache_mib_request":r["cache_bytes_after_decode_per_request"]/2**20,"peak_mib_gpu":r["total_peak_allocated_bytes_per_gpu_max"]/2**20})
        for world in (1, 32):
            r=dynamic[model][world]["steady_state"]["summary"]
            rows.append({"workload":"dynamic_trace","model":model,"gpus":world,"aggregate_end_to_end_tokens_s":r["aggregate_end_to_end_tokens_per_second"],"aggregate_prefill_tokens_s":r["aggregate_prefill_tokens_per_second"],"aggregate_decode_tokens_s":r["aggregate_decode_tokens_per_second"],"runtime_s":r["total_seconds"],"cache_mib_request":r["max_cache_bytes_per_request"]/2**20,"peak_mib_gpu":r["max_peak_allocated_bytes_per_gpu"]/2**20})
    with (OUT / "inference_2k_32gpu_comparison.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

    summary={
        "quality_primary_metric_macro_percent": {m: sum(scores[m])/len(scores[m]) for m in MODELS},
        "fixed_2k_32gpu": {m: fixed[m][32]["result"] for m in MODELS},
        "dynamic_32gpu_steady": {m: dynamic[m][32]["steady_state"]["summary"] for m in MODELS},
    }
    (OUT / "inference_2k_32gpu_summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
