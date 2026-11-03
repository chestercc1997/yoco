# Experiment entry points

The current campaign is A/B six-source K/V aggregation. Its configuration and
outputs are separate from historical baseline / three-source fusion runs.

| Entry | Purpose |
| --- | --- |
| `tools/deploy_plan_ab.py` | Package an immutable release, verify data, test inside the HPC3 image, and submit once with a recorded intent |
| `scripts/plan_ab_4gpu.sbatch` | Request one node and four GPUs without fixing the node |
| `scripts/plan_ab_4gpu_inside.sh` | Check both models with four-rank smoke/resume, train A then B, evaluate both on eight tasks |
| `tools/preflight_plan_ab.py` | Verify release hashes, corpus identity, and paired training settings |
| `tools/preflight_plan_ab_eval.py` | Verify all eight pinned datasets load offline |
| `tools/summarize_plan_ab.py` | Reject incomplete/smoke results and summarize A/B downstream scores |

Deployment helpers are specific to the documented HPC2/HPC3 environment. They
use an external authorized SSH helper; no credentials are stored here. The
current campaign is already submitted: inspect its recorded job before running
any deployment command. See [the campaign record](../docs/PLAN_AB_4GPU.md).

For another environment, use the model configurations with the generic trainer:

```bash
torchrun --standalone --nproc-per-node=4 -m runtime.train \
  --config configs/yoco_160m_plan_ab_A_4gpu.json --run-dir runs/plan-ab-A
```

Prepare and verify the training manifest first. Change `A` to `B` for the
conditional model. Each config specifies 40,000 updates; use a separate run
directory per model. Historical `plan_full_*` scripts run baseline/conditional
and are not the current A/B entry. `fusion_*` and `paired_2gpu_*` scripts refer
to earlier three-source hidden-state experiments.
