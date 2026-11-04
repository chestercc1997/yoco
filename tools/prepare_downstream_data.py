"""Prepare or verify the eight pinned downstream datasets on any local host."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("arc_easy", "arc_challenge", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande", "sciq")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locks-dir", type=Path, default=ROOT / "evaluation/plan_ab_data_locks")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "data/cache/huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    if args.offline:
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")

    from huggingface_hub import snapshot_download
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config
    sys.path.insert(0, str(ROOT))
    from evaluation.locked_task import apply_lock

    manager = TaskManager()
    results = {}
    for name in TASKS:
        lock = json.loads((args.locks_dir / f"data_lock_{name}.json").read_text())
        if lock["task"] != name:
            raise ValueError(f"Task lock mismatch: {name}")
        if name == "piqa" and lock["revision"] == "b40a463b4ac9c77c6e74e23c5d873f5c2139d8fa":
            snapshot_download(repo_id=lock["dataset"], repo_type="dataset", revision=lock["revision"],
                              allow_patterns=["plain_text/*.parquet"], local_files_only=args.offline)
        config = load_yaml_config(manager.task_index[name]["yaml_path"])
        apply_lock(config, lock, os.environ["HF_HOME"])
        task = ConfigurableTask(config=config)
        split_counts = {split: len(docs) for split, docs in task.dataset.items()}
        docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
        if not len(docs):
            raise RuntimeError(f"Empty evaluation dataset: {name}")
        results[name] = {**lock, "splits": split_counts, "evaluation_documents": len(docs)}
        print(f"READY {'offline ' if args.offline else ''}{name}: {len(docs)} evaluation documents; {split_counts}", flush=True)
    report = {"ready": True, "offline": args.offline, "locks_dir": str(args.locks_dir.resolve()),
              "hf_home": os.environ["HF_HOME"], "tasks": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
