#!/usr/bin/env python3
"""Plot measured long-context inference results for YOCO and Transformer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "evaluation" / "figures"
INPUTS = {
    "Transformer": FIGURES / "long_context_transformer_v100.json",
    "YOCO": FIGURES / "long_context_yoco_v100.json",
}
COLORS = {"Transformer": "#1f77b4", "YOCO": "#ff7f0e"}


def load() -> dict[str, list[dict]]:
    runs = {}
    for name, path in INPUTS.items():
        payload = json.loads(path.read_text())
        rows = payload["results"]
        assert all(row["status"] == "ok" for row in rows)
        lengths = [row["context_length"] for row in rows]
        assert lengths == sorted(lengths) and len(lengths) == len(set(lengths))
        assert [row["cache_bytes_after_prefill"] for row in rows] == sorted(
            row["cache_bytes_after_prefill"] for row in rows
        )
        runs[name] = rows
    assert [row["context_length"] for row in runs["Transformer"]] == [
        row["context_length"] for row in runs["YOCO"]
    ]
    return runs


def axis_style(ax: plt.Axes, lengths: np.ndarray) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, [f"{length // 1024}K" for length in lengths])
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def annotate_ratio(
    ax: plt.Axes, x: float, y: float, label: str, color: str, y_offset: int = 12
) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(-8, y_offset),
        textcoords="offset points",
        ha="right",
        color=color,
        fontsize=9,
        fontweight="bold",
        va="bottom" if y_offset >= 0 else "top",
    )


def plot(runs: dict[str, list[dict]]) -> None:
    lengths = np.asarray([row["context_length"] for row in runs["YOCO"]])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    memory_ax, prefill_ax, decode_ax, total_ax = axes.flat

    for name, rows in runs.items():
        color = COLORS[name]
        peak = np.asarray([row["total_peak_allocated_bytes"] / 2**30 for row in rows])
        cache = np.asarray([row["cache_bytes_after_prefill"] / 2**30 for row in rows])
        prefill = np.asarray([row["prefill_seconds"] for row in rows])
        decode = np.asarray([row["decode_tokens_per_second"] for row in rows])
        total = np.asarray([row["end_to_end_tokens_per_second"] for row in rows])

        memory_ax.plot(
            lengths, peak, color=color, marker="o", linewidth=2.2, label=f"{name} peak"
        )
        memory_ax.plot(
            lengths,
            cache,
            color=color,
            marker="s",
            linewidth=1.7,
            linestyle="--",
            label=f"{name} incremental cache",
        )
        prefill_ax.plot(lengths, prefill, color=color, marker="o", linewidth=2.2, label=name)
        decode_ax.plot(lengths, decode, color=color, marker="o", linewidth=2.2, label=name)
        total_ax.plot(lengths, total, color=color, marker="o", linewidth=2.2, label=name)

    transformer = runs["Transformer"][-1]
    yoco = runs["YOCO"][-1]
    prefill_speedup = transformer["prefill_seconds"] / yoco["prefill_seconds"]
    decode_speedup = yoco["decode_tokens_per_second"] / transformer["decode_tokens_per_second"]
    total_speedup = (
        yoco["end_to_end_tokens_per_second"]
        / transformer["end_to_end_tokens_per_second"]
    )
    cache_reduction = (
        transformer["cache_bytes_after_prefill"] / yoco["cache_bytes_after_prefill"]
    )
    peak_reduction = (
        transformer["total_peak_allocated_bytes"] / yoco["total_peak_allocated_bytes"]
    )

    memory_ax.set(
        title="Measured GPU memory",
        xlabel="Context length",
        ylabel="Allocated memory (GiB)",
    )
    prefill_ax.set(
        title="Prefill latency (lower is better)",
        xlabel="Context length",
        ylabel="Seconds",
        yscale="log",
    )
    decode_ax.set(
        title="Autoregressive decode throughput",
        xlabel="Context length",
        ylabel="Generated tokens/s",
    )
    total_ax.set(
        title="End-to-end throughput: prompt + 64 generated tokens",
        xlabel="Context length",
        ylabel="Total processed tokens/s",
    )

    for ax in axes.flat:
        axis_style(ax, lengths)
        ax.legend(frameon=True, fontsize=8)

    annotate_ratio(
        prefill_ax,
        lengths[-1],
        yoco["prefill_seconds"],
        f"{prefill_speedup:.2f}\u00d7 faster",
        COLORS["YOCO"],
    )
    annotate_ratio(
        decode_ax,
        lengths[-1],
        yoco["decode_tokens_per_second"],
        f"{decode_speedup:.2f}\u00d7",
        COLORS["YOCO"],
    )
    annotate_ratio(
        total_ax,
        lengths[-1],
        yoco["end_to_end_tokens_per_second"],
        f"{total_speedup:.2f}\u00d7",
        COLORS["YOCO"],
        y_offset=-16,
    )

    fig.suptitle(
        "YOCO 166.7M vs Transformer 162.0M: measured long-context inference",
        fontsize=16,
        y=0.985,
    )
    footer = (
        "Single Tesla V100 32GB, FP16, batch=1, steady-state median of 5 runs. "
        f"At 32K: cache {cache_reduction:.2f}\u00d7 smaller, peak allocation {peak_reduction:.2f}\u00d7 lower. "
        "Contexts >2K profile efficiency only; these checkpoints were trained at 2K."
    )
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    stem = FIGURES / "03_yoco_transformer_long_context_inference_v100"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_comparison(runs: dict[str, list[dict]]) -> None:
    transformer = {row["context_length"]: row for row in runs["Transformer"]}
    yoco = {row["context_length"]: row for row in runs["YOCO"]}
    rows = []
    for length in transformer:
        base, candidate = transformer[length], yoco[length]
        rows.append(
            {
                "context_length": length,
                "transformer_prefill_seconds": base["prefill_seconds"],
                "yoco_prefill_seconds": candidate["prefill_seconds"],
                "yoco_prefill_speedup": base["prefill_seconds"]
                / candidate["prefill_seconds"],
                "transformer_decode_tokens_s": base["decode_tokens_per_second"],
                "yoco_decode_tokens_s": candidate["decode_tokens_per_second"],
                "yoco_decode_speedup": candidate["decode_tokens_per_second"]
                / base["decode_tokens_per_second"],
                "transformer_end_to_end_tokens_s": base[
                    "end_to_end_tokens_per_second"
                ],
                "yoco_end_to_end_tokens_s": candidate[
                    "end_to_end_tokens_per_second"
                ],
                "yoco_end_to_end_speedup": candidate[
                    "end_to_end_tokens_per_second"
                ]
                / base["end_to_end_tokens_per_second"],
                "transformer_cache_mib": base["cache_bytes_after_prefill"] / 2**20,
                "yoco_cache_mib": candidate["cache_bytes_after_prefill"] / 2**20,
                "cache_reduction": base["cache_bytes_after_prefill"]
                / candidate["cache_bytes_after_prefill"],
                "transformer_peak_gib": base["total_peak_allocated_bytes"] / 2**30,
                "yoco_peak_gib": candidate["total_peak_allocated_bytes"] / 2**30,
                "peak_memory_reduction": base["total_peak_allocated_bytes"]
                / candidate["total_peak_allocated_bytes"],
            }
        )
    path = FIGURES / "long_context_inference_comparison.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "scope": "Measured architecture profiling; quality is not claimed above the 2K training length",
        "hardware": "single Tesla V100-PCIE-32GB",
        "precision": "FP16",
        "batch_size": 1,
        "decode_tokens": 64,
        "steady_state_repeats": 5,
        "comparison": rows,
    }
    (FIGURES / "long_context_inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main() -> None:
    runs = load()
    plot(runs)
    write_comparison(runs)
    print("Wrote measured long-context inference figure and comparison tables")


if __name__ == "__main__":
    main()
