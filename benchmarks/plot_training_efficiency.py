#!/usr/bin/env python3
"""Plot YOCO and paired Transformer training efficiency from formal-run logs.

The measured time axis sums the per-update ``seconds`` values recorded by rank
zero.  It therefore represents active training-step time and excludes
validation/checkpoint overhead.  The 6ND axis is a parameter-token compute
proxy, not measured hardware FLOPs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation" / "figures"
TOKENS_PER_UPDATE = 262_144
WORLD_SIZE = 32
PFLOP_DAY = 1e15 * 86_400
ROLLING_WINDOW = 500

COLORS = {"Transformer": "#1f77b4", "YOCO": "#ff7f0e"}


def load_run(name: str, path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train = [row for row in rows if row.get("event") == "train"]
    validation = [row for row in rows if row.get("event") == "validation"]
    started = next(row for row in rows if row.get("event") == "started")

    assert len(train) == 40_000, f"{name}: expected 40,000 train rows"
    assert len(validation) == 160, f"{name}: expected 160 validation rows"
    assert [row["step"] for row in train] == list(range(1, 40_001))
    assert train[-1]["tokens"] == 10_485_760_000
    assert started["world_size"] == WORLD_SIZE
    assert all(row["tokens"] == row["step"] * TOKENS_PER_UPDATE for row in train)

    seconds = np.asarray([row["seconds"] for row in train], dtype=float)
    cumulative_seconds = np.cumsum(seconds)
    validation_steps = np.asarray([row["step"] for row in validation], dtype=int)
    parameters = int(started["parameters"])
    total_tokens = int(train[-1]["tokens"])
    start_time = datetime.fromisoformat(started["time"])
    validation_elapsed_hours = np.asarray(
        [
            (datetime.fromisoformat(row["time"]) - start_time).total_seconds() / 3600
            for row in validation
        ]
    )
    rolling_seconds = np.convolve(
        seconds, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid"
    )

    return {
        "name": name,
        "path": path,
        "parameters": parameters,
        "train": train,
        "validation": validation,
        "validation_steps": validation_steps,
        "validation_loss": np.asarray([row["loss"] for row in validation]),
        "validation_tokens_b": validation_steps * TOKENS_PER_UPDATE / 1e9,
        "validation_gpu_hours": cumulative_seconds[validation_steps - 1]
        * WORLD_SIZE
        / 3600,
        "validation_elapsed_hours": validation_elapsed_hours,
        "validation_compute_pflop_days": (
            6.0 * parameters * validation_steps.astype(float) * TOKENS_PER_UPDATE
            / PFLOP_DAY
        ),
        "rolling_step": np.arange(ROLLING_WINDOW, len(train) + 1),
        "rolling_tokens_b": (
            np.arange(ROLLING_WINDOW, len(train) + 1) * TOKENS_PER_UPDATE / 1e9
        ),
        "rolling_throughput_k": TOKENS_PER_UPDATE / rolling_seconds / 1e3,
        "total_tokens": total_tokens,
        "active_seconds": float(cumulative_seconds[-1]),
        "active_gpu_hours": float(cumulative_seconds[-1] * WORLD_SIZE / 3600),
        "end_to_end_hours": float(validation_elapsed_hours[-1]),
        "compute_pflop_days": float(6 * parameters * total_tokens / PFLOP_DAY),
        "mean_throughput_tokens_s": float(total_tokens / cumulative_seconds[-1]),
        "median_step_seconds": float(np.median(seconds)),
        "peak_memory_gib": float(max(row["peak_memory_bytes"] for row in train) / 2**30),
        "final_validation_loss": float(validation[-1]["loss"]),
        "final_perplexity": float(validation[-1]["perplexity"]),
    }


def style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def plot(runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    tokens_ax, compute_ax, time_ax, speed_ax = axes.flat

    for run in runs:
        name = run["name"]
        color = COLORS[name]
        common = dict(color=color, linewidth=2.2, label=name)
        tokens_ax.plot(
            run["validation_tokens_b"],
            run["validation_loss"],
            marker="o",
            markevery=8,
            markersize=4,
            **common,
        )
        compute_ax.plot(
            run["validation_compute_pflop_days"],
            run["validation_loss"],
            marker="o",
            markevery=8,
            markersize=4,
            **common,
        )
        time_ax.plot(
            run["validation_gpu_hours"],
            run["validation_loss"],
            marker="o",
            markevery=8,
            markersize=4,
            **common,
        )
        speed_ax.plot(
            run["rolling_tokens_b"], run["rolling_throughput_k"], **common
        )

    tokens_ax.set(
        title="Validation loss vs training tokens",
        xlabel="Processed tokens (billions)",
        ylabel="Validation loss",
        ylim=(3.05, 7.0),
    )
    compute_ax.set(
        title="Validation loss vs estimated compute",
        xlabel="Estimated training compute (6ND PF-days)",
        ylabel="Validation loss",
        ylim=(3.05, 7.0),
    )
    time_ax.set(
        title="Validation loss vs measured training time",
        xlabel="Active train-step GPU hours (32 GPUs)",
        ylabel="Validation loss",
        ylim=(3.05, 7.0),
    )
    speed_ax.set(
        title=f"Training throughput ({ROLLING_WINDOW}-step rolling mean)",
        xlabel="Processed tokens (billions)",
        ylabel="Global throughput (thousand tokens/s)",
    )

    for ax in axes.flat:
        style_axis(ax)
        ax.legend(frameon=True, fontsize=9)

    transformer, yoco = runs
    speed_ratio = (
        yoco["mean_throughput_tokens_s"] / transformer["mean_throughput_tokens_s"]
    )
    memory_ratio = yoco["peak_memory_gib"] / transformer["peak_memory_gib"]
    footer = (
        "Formal 32\u00d7V100 runs, same data/order and 10.49B-token budget. "
        f"YOCO mean throughput = {speed_ratio:.3f}\u00d7 Transformer; "
        f"peak memory = {memory_ratio:.3f}\u00d7.  "
        "6ND is a parameter-token proxy; GPU hours sum measured update time and exclude eval/checkpoint overhead."
    )
    fig.suptitle(
        "YOCO 166.7M vs Transformer 162.0M: training quality and efficiency",
        fontsize=16,
        y=0.985,
    )
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "01_yoco_transformer_training_efficiency"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def interpolate_time_to_loss(run: dict, targets: np.ndarray) -> np.ndarray:
    """Interpolate observed end-to-end time at monotonically decreasing loss."""
    losses = run["validation_loss"]
    elapsed = run["validation_elapsed_hours"]
    assert np.all(np.diff(losses) < 0), f'{run["name"]}: validation loss not monotonic'
    return np.interp(targets, losses[::-1], elapsed[::-1])


def plot_measured_hardware(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    full_ax, late_ax, iso_ax, speed_ax = axes.flat

    for run in runs:
        name = run["name"]
        color = COLORS[name]
        common = dict(color=color, linewidth=2.2, label=name)
        for ax in (full_ax, late_ax):
            ax.plot(
                run["validation_elapsed_hours"],
                run["validation_loss"],
                marker="o",
                markevery=8,
                markersize=4,
                **common,
            )
        speed_ax.plot(
            run["rolling_tokens_b"], run["rolling_throughput_k"], **common
        )
        speed_ax.axhline(
            run["mean_throughput_tokens_s"] / 1e3,
            color=color,
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
        )

    common_low = max(run["final_validation_loss"] for run in runs)
    common_high = min(run["validation_loss"][0] for run in runs)
    targets = np.linspace(common_low, common_high, 240)
    transformer_time = interpolate_time_to_loss(runs[0], targets)
    yoco_time = interpolate_time_to_loss(runs[1], targets)
    advantage = 100 * (transformer_time / yoco_time - 1)
    iso_ax.plot(targets, advantage, color="#7b3294", linewidth=2.2)
    iso_ax.axhline(0, color="#333333", linestyle="--", linewidth=1.2)
    iso_ax.fill_between(
        targets, 0, advantage, where=advantage >= 0, color="#2ca02c", alpha=0.18
    )
    iso_ax.fill_between(
        targets, 0, advantage, where=advantage < 0, color="#d62728", alpha=0.14
    )

    full_ax.set(
        title="Validation loss vs observed wall-clock time",
        xlabel="Elapsed wall-clock hours",
        ylabel="Validation loss",
        ylim=(3.05, 7.0),
    )
    late_ax.set(
        title="Late-training wall-clock view",
        xlabel="Elapsed wall-clock hours",
        ylabel="Validation loss",
        ylim=(3.14, 3.45),
    )
    iso_ax.set(
        title="Time to equal validation loss",
        xlabel="Target validation loss",
        ylabel="YOCO wall-time advantage (%)",
    )
    iso_ax.invert_xaxis()
    speed_ax.set(
        title=f"Measured update throughput ({ROLLING_WINDOW}-step mean)",
        xlabel="Processed tokens (billions)",
        ylabel="Global throughput (thousand tokens/s)",
    )

    for ax in axes.flat:
        style_axis(ax)
    full_ax.legend(frameon=True, fontsize=9)
    late_ax.legend(frameon=True, fontsize=9)
    speed_ax.legend(frameon=True, fontsize=9)
    iso_ax.text(
        0.02,
        0.04,
        "Above 0: YOCO reaches the target sooner\nBelow 0: Transformer reaches it sooner",
        transform=iso_ax.transAxes,
        fontsize=8.5,
        va="bottom",
    )

    transformer, yoco = runs
    elapsed_ratio = yoco["end_to_end_hours"] / transformer["end_to_end_hours"]
    memory_saving = 1 - yoco["peak_memory_gib"] / transformer["peak_memory_gib"]
    footer = (
        "Directly observed on the paired 32\u00d7V100 runs; no 6ND axis. "
        f"Train start\u2192final validation: {transformer['end_to_end_hours']:.2f} h Transformer, "
        f"{yoco['end_to_end_hours']:.2f} h YOCO ({elapsed_ratio:.3f}\u00d7). "
        f"YOCO peak memory is {memory_saving:.1%} lower. Dotted throughput lines show full-run means."
    )
    fig.suptitle(
        "YOCO 166.7M vs Transformer 162.0M: measured V100 training performance",
        fontsize=16,
        y=0.985,
    )
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))

    stem = OUTPUT / "02_yoco_transformer_measured_hardware_efficiency"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return targets, advantage


def write_artifacts(
    runs: list[dict], iso_loss_targets: np.ndarray, iso_loss_advantage: np.ndarray
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "Formal paired 32xV100 training runs",
        "definitions": {
            "estimated_compute_pflop_days": "6 * parameters * tokens / (1e15 * 86400)",
            "active_gpu_hours": "sum(rank0 per-update seconds) * 32 / 3600; excludes validation and checkpoint overhead",
            "mean_throughput_tokens_s": "total tokens / sum(rank0 per-update seconds)",
            "end_to_end_hours": "ISO log timestamp from formal train start to final validation; includes validation/checkpoint overhead",
            "rolling_throughput_window_steps": ROLLING_WINDOW,
            "peak_memory_gib": "maximum rank0 logged peak_memory_bytes / 2^30",
        },
        "runs": {
            run["name"]: {
                key: run[key]
                for key in (
                    "parameters",
                    "total_tokens",
                    "active_seconds",
                    "active_gpu_hours",
                    "end_to_end_hours",
                    "compute_pflop_days",
                    "mean_throughput_tokens_s",
                    "median_step_seconds",
                    "peak_memory_gib",
                    "final_validation_loss",
                    "final_perplexity",
                )
            }
            for run in runs
        },
        "source_sha256": {
            str(run["path"]): hashlib.sha256(run["path"].read_bytes()).hexdigest()
            for run in runs
        },
    }
    baseline = summary["runs"]["Transformer"]
    yoco = summary["runs"]["YOCO"]
    summary["relative"] = {
        "yoco_over_transformer_mean_throughput": yoco["mean_throughput_tokens_s"]
        / baseline["mean_throughput_tokens_s"],
        "yoco_over_transformer_active_gpu_hours": yoco["active_gpu_hours"]
        / baseline["active_gpu_hours"],
        "yoco_over_transformer_peak_memory": yoco["peak_memory_gib"]
        / baseline["peak_memory_gib"],
        "yoco_minus_transformer_final_validation_loss": yoco[
            "final_validation_loss"
        ]
        - baseline["final_validation_loss"],
    }
    (OUTPUT / "training_efficiency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with (OUTPUT / "training_efficiency_validation_points.csv").open(
        "w", newline=""
    ) as stream:
        fields = [
            "model",
            "step",
            "tokens",
            "validation_loss",
            "perplexity",
            "estimated_compute_pflop_days",
            "active_gpu_hours",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for index, row in enumerate(run["validation"]):
                writer.writerow(
                    {
                        "model": run["name"],
                        "step": row["step"],
                        "tokens": row["step"] * TOKENS_PER_UPDATE,
                        "validation_loss": row["loss"],
                        "perplexity": row["perplexity"],
                        "estimated_compute_pflop_days": run[
                            "validation_compute_pflop_days"
                        ][index],
                        "active_gpu_hours": run["validation_gpu_hours"][index],
                    }
                )

    with (OUTPUT / "measured_time_to_loss.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target_validation_loss", "yoco_wall_time_advantage_percent"])
        writer.writerows(zip(iso_loss_targets, iso_loss_advantage))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-log", type=Path, required=True)
    parser.add_argument("--yoco-log", type=Path, required=True)
    args = parser.parse_args()
    runs = [
        load_run("Transformer", args.transformer_log),
        load_run("YOCO", args.yoco_log),
    ]
    plot(runs)
    targets, advantage = plot_measured_hardware(runs)
    write_artifacts(runs, targets, advantage)
    print(f"Wrote training-efficiency artifacts to {OUTPUT}")


if __name__ == "__main__":
    main()
