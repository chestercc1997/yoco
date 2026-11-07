# Model checks

Run from the repository root in an environment with the project's PyTorch dependencies:

```bash
python tests/test_plan_ab_kv.py
python tests/test_memory_fusion.py
python tests/test_conditional_kv.py
python tests/test_training_guard.py
```

`test_plan_ab_kv.py` covers all six source layers, global versus input-dependent
weights, explicit candidate K/V sums and reference gradients, common backbone
initialization, equal initial A/B outputs, the last-source baseline limit,
causality, shared incremental K/V cache, activation recomputation, and exact
optimizer continuation. The other two suites cover historical fusion behavior
and the original conditional implementation.

`test_training_guard.py` injects corrupted and nonfinite checkpoints, transient
and sustained validation regressions, an unauthorized learning-rate change,
and worker termination. It checks complete-state selection, archived failure
evidence, idempotent rollback, exact optimizer/scaler/RNG preservation, and
refusal to change the frozen reproduction recipe.

The HPC3 allocation additionally tests actual four-rank mixed-precision training
and checkpoint resume for both A/B before formal training. Passing implementation
checks does not establish convergence or downstream performance.
