# YOCO 160M: learned and conditional depth K/V aggregation

This repository combines the previously separate YOCO and Transformer baseline
trees into one reproducible project. It contains model and training code,
evaluation and inference benchmarks, pinned dataset metadata, measured tables,
and figures. Checkpoints and training data are intentionally excluded.

## Current SWA 160M baseline / A / B experiment

The new cohort uses YOCO's **sliding-window self-decoder**, with the paper's
1,024-token window, followed by the shared global K/V cross-decoder. Baseline,
learned K/V fusion (A), and conditional K/V fusion (B) share bitwise-equal backbone
initializations and the existing FineWeb-Edu/tokenizer/FP16 training recipe.
Each arm runs on 8×A100 for 40,000 updates and all eight downstream tasks;
every 5,000-update checkpoint is retained. The controller first checks real-data
eight-rank checkpoint restoration for all three arms. The SWA baseline completed
40,000 updates and all eight full tasks on November 7, 2026: validation loss
**3.180143**, eight-task macro **46.3682%**, with all eight scheduled checkpoints
retained and independently hashed. Method A is now training on eight GPUs;
method B and the final three-arm comparison remain pending. See the
[verified baseline result](docs/evidence/swa-160m-baseline-completed-20261107.json).
See [configuration, checks and live-status commands](docs/SWA_160M_AB.md).

The [reproduction skill](.agents/skills/yoco-reproduction/SKILL.md) makes the
training recipe immutable during fixes, acceleration and recovery. The active
supervisor detects nonfinite losses or sustained validation regression, archives
failed evidence, and replays from a verified full checkpoint with identical
settings and data order. Repeated failures enter bounded diagnosis under the
ongoing Goal; learning rates, precision and other training settings are never
automatically adjusted.

## Completed 400M A training

Method A at **400.59M parameters on 8×A100** completed 40,000 updates and eight
full downstream tasks on November 7, 2026: validation loss **2.918184**,
eight-task macro average **48.5270%**. The final checkpoint SHA256 and all eight
evaluation provenance records were verified independently. This run resumed after
the documented retention arithmetic repair, with compilation disabled; its
post-repair segment recorded no nonfinite forward events. Baseline/B were not
trained at this size. See the [completed run and repair record](docs/PLAN_A_400M_8GPU.md).

## Completed 160M A/B experiment

The current experiment compares two ways to aggregate K/V from all six
self-decoder layers, constructing shared memory once before layers 7–12:

| Model | Aggregation weights | Sources |
| --- | --- | --- |
| A (`learned_kv`) | Six global trainable softmax logits; shared across tokens | Layers 1–6 |
| B (`conditional_kv`) | Per-token scores from normalized source hidden states | Layers 1–6 |
| Original YOCO baseline | K/V from the final self-decoder output | Layer 6 |

**Completed three-way comparison (2026-11-05):** the original YOCO baseline
completed 40,000 updates on **8×A100**; A/B retain the published **8×A800** runs.
All three consumed 10,485,760,000 tokens and completed all eight full zero-shot tasks.

| Metric | New YOCO baseline · 8×A100 | A | B |
| --- | ---: | ---: | ---: |
| Final validation loss | 3.177219 | 3.160054 | 3.159059 |
| Eight-task macro average (%) | 45.6268 | 46.9417 | 45.9075 |
| Trunk 6ND (PFLOP-days) | 0.065308109 | 0.065308114 | 0.065308673 |

The baseline above is the newly completed eight-A100 run. Full-size model checks
confirm the same common backbone tensors at initialization; A adds 6 parameters,
B adds 774. Model/training hyperparameters and all task YAMLs, seeds and zero-shot
counts match. Baseline saves every 5,000 steps and retains all eight checkpoints;
its compiler/fused-optimizer execution options are documented separately. Remote A/B step-zero payloads and token-shard hashes are not available
for a fresh check, so baseline differences remain descriptive. A has the highest
downstream macro; B has the lowest final validation loss. The accelerated FP16 baseline trained in 3.97 hours
and completed training plus evaluation in 4.13 hours.
Its validation curve has no sustained regression like the superseded run.
See [reusable acceleration settings and commands](docs/BASELINE_A100_ACCELERATION.md).
6ND remains a compute proxy; timings across different execution options do not isolate architecture effects. See the [complete report, backbone audit, seven figures and CSVs](evaluation/ab_baseline_20261104/README.md).

[七张三组对比图 / Comparison figures](evaluation/ab_baseline_20261104/figures/README.md) · PNG / PDF / SVG

![Baseline A B quality comparison](evaluation/ab_baseline_20261104/figures/01_quality_summary.png)

- [A/B design and execution](docs/PLAN_AB_4GPU.md)
- [Design formulas](docs/plan.md)
- [Documentation index](docs/README.md)
- [Training and evaluation scripts](scripts/README.md)
- [Test commands](tests/README.md)

The original three-source `token` and `static` hidden-state fusion experiments
remain available as historical implementations. They differ from the six-source
K/V aggregation in A/B. Existing two-GPU baseline / token-v2 validation results
are documented separately in [the training review](docs/evidence/full-v2-convergence-review.md).

## Earlier YOCO / Transformer reproduction

The reports below describe the earlier 32-V100 YOCO/Transformer comparison,
separate from the completed A/B experiment above.

## Earlier reproduction result

![2K quality and inference summary](evaluation/figures/04_yoco_transformer_2k_32gpu_inference.png)

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
configs/                 original reproduction, historical fusion, and A/B configurations
tools/                   data preparation, deployment, verification, and summaries
docs/                    current A/B design, historical results, and tutorials
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
