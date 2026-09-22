# YOCO 160M reproduction and Transformer comparison

This repository combines the previously separate YOCO and Transformer baseline
trees into one reproducible project. It contains model and training code,
evaluation and inference benchmarks, pinned dataset metadata, measured tables,
and figures. Checkpoints and training data are intentionally excluded.

![2K quality and inference summary](evaluation/figures/04_yoco_transformer_2k_32gpu_inference.png)

## Main result

The paired 40,000-step runs have nearly identical validation loss and aggregate
downstream accuracy. YOCO uses much less inference cache and has substantially
faster prefill. On the measured dynamic-shape workload it reaches 1.18x the
32-GPU end-to-end throughput; on fixed 2K prompt + 64-token generation it is
0.97x because decode dominates and the current V100 eager decode is slightly
slower.

See [the concise report](evaluation/YOCO_BASELINE_REPORT.md) for the results and
[the detailed 2K report](evaluation/INFERENCE_2K_32GPU.md) for methodology.

## Repository layout

```text
runtime/                 unified YOCO and Transformer model/training code
configs/                 paired 160M configurations
tools/                   dataset preparation and verification
evaluation/              lm-eval adapter, reports, summaries, raw metrics, figures
benchmarks/              fixed, dynamic-shape, long-context, and plotting scripts
tests/                   architecture and recovery checks
provenance/              upstream commit and pinned public-data metadata
```

## Environment

Python 3.12 and CUDA are recommended. Install the CUDA-compatible PyTorch build
for the target machine, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The measured runs used PyTorch 2.7.1, FP16, and 32 V100 32GB GPUs.

## Data and training

Prepare the pinned FineWeb-Edu sample-10BT corpus once. Both configurations
refer to the same relative manifest.

```bash
python tools/prepare_data.py --config configs/yoco_160m.json
python tools/verify_data.py --config configs/yoco_160m.json
```

Single-GPU synthetic smoke tests do not require the corpus:

```bash
python -m runtime.train \
  --config configs/yoco_160m.json \
  --run-dir runs/yoco-smoke --smoke --stop-after 2

python -m runtime.train \
  --config configs/transformer_160m.json \
  --run-dir runs/transformer-smoke --smoke --stop-after 2
```

For the formal run, invoke the generic launcher on every node. Set a different
`NODE_RANK` from 0 to 3 and the same `MASTER_ADDR` on all nodes:

```bash
NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
  bash scripts/run_4node_training.sh configs/yoco_160m.json runs/yoco
```

The configuration requires exactly 32 ranks. The same launcher accepts the
Transformer configuration.

## Downstream evaluation

The eight task dataset revisions are pinned in `evaluation/data_locks/`.

```bash
python evaluation/run.py \
  --config configs/yoco_160m.json \
  --checkpoint checkpoints/yoco/checkpoint_00040000.pt \
  --task arc_easy --output runs/eval/yoco/arc_easy --device cuda:0
```

Use the same command with `configs/transformer_160m.json` for the baseline.

## Inference benchmarks

Single-GPU or `torchrun` fixed 2K benchmark:

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/benchmark_2k_distributed_inference.py \
  --config configs/yoco_160m.json \
  --checkpoint checkpoints/yoco/checkpoint_00040000.pt \
  --output runs/inference/yoco_w8_b1.json
```

Dynamic-shape benchmark:

```bash
python benchmarks/benchmark_dynamic_shape_inference.py \
  --config configs/yoco_160m.json \
  --checkpoint checkpoints/yoco/checkpoint_00040000.pt \
  --output runs/inference/yoco_dynamic_w1_b1.json
```

The dynamic trace uses a fixed seed and eight non-monotonic prompt/generation
patterns from 128 to 2048 prompt tokens and 16 to 64 generated tokens.

For the measured 4-node layout, run this on each node with its own rank:

```bash
NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
  bash scripts/run_4node_benchmark.sh dynamic \
  configs/yoco_160m.json checkpoints/yoco/checkpoint_00040000.pt \
  runs/inference/yoco_dynamic_w32_b1.json
```

## Recreate the checked-in figures

The small raw inference JSON files are included, so the main figures can be
regenerated without checkpoints:

```bash
python benchmarks/plot_2k_32gpu_inference.py \
  --raw-dir evaluation/raw/2k_32gpu
python benchmarks/plot_long_context_inference.py
```

Training plots require the two `rank0.jsonl` logs from a new formal run:

```bash
python benchmarks/plot_training_efficiency.py \
  --transformer-log runs/transformer/rank0.jsonl \
  --yoco-log runs/yoco/rank0.jsonl
```

## Scope

This is a method reproduction on public data. It does not contain official
pretrained weights and does not claim to reproduce the original private-data
training run. Context lengths above 2K in the long-context figure measure
system efficiency only, because these checkpoints were trained at 2K.
