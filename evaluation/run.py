"""Run a pinned zero-shot lm-evaluation-harness task."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.adapter import ReproductionLM
from evaluation.locked_task import apply_lock
from runtime.model import architecture
from runtime.train import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--locks-dir", type=Path, default=ROOT / "evaluation/data_locks")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    from lm_eval import simple_evaluate
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config

    args.output.mkdir(parents=True, exist_ok=True)
    manager = TaskManager()
    yaml_path = manager.task_index[args.task]["yaml_path"]
    task_config = load_yaml_config(yaml_path)
    lock_path = args.locks_dir / f"data_lock_{args.task}.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
    else:
        info = HfApi().dataset_info(task_config["dataset_path"])
        lock = {"dataset": info.id, "revision": info.sha, "task": args.task,
                "original_dataset_path": task_config["dataset_path"]}
        atomic_json(lock_path, lock)
    apply_lock(task_config, lock, os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    config = json.loads(args.config.read_text())
    provenance = {
        "architecture": architecture(config), "checkpoint_sha256": sha256(args.checkpoint),
        "dataset_lock": lock, "task_yaml_sha256": sha256(yaml_path), "num_fewshot": 0,
        "limit": args.limit, "seed": 42,
        "precision": config.get('evaluation_precision',config['training'].get('precision','fp16_mixed_fp32_params')),
        "execution": config.get('execution',{}),
        "versions": {name: importlib.metadata.version(name) for name in ("lm_eval", "datasets", "transformers", "torch")},
    }
    atomic_json(args.output / "provenance.json", provenance)
    task = ConfigurableTask(config=task_config)
    model = ReproductionLM(args.config, args.checkpoint, args.device)
    result = simple_evaluate(model=model, tasks=[task], num_fewshot=0, limit=args.limit,
        bootstrap_iters=1000, log_samples=True, random_seed=42, numpy_random_seed=42,
        torch_random_seed=42, fewshot_random_seed=42)
    result.update(model_step=model.step, architecture=architecture(config), smoke_only=args.limit is not None)
    atomic_json(args.output / "results.json", json.loads(json.dumps(result, default=str)))


if __name__ == "__main__":
    main()
