"""Download and verify the eight pinned lm-eval task datasets."""
import json
import os
from pathlib import Path
import sys
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from evaluation.locked_task import apply_lock
manager = TaskManager()
for name in ("arc_easy", "arc_challenge", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande", "sciq"):
    lock = json.loads((root / "runs/eval-full-v2/data_locks" / f"data_lock_{name}.json").read_text())
    config = load_yaml_config(manager.task_index[name]["yaml_path"])
    apply_lock(config, lock, os.environ["HF_HOME"])
    task = ConfigurableTask(config=config)
    print(f"READY {name}: {len(task.eval_docs)} documents", flush=True)
