"""Resume the eight-GPU baseline to 40000 updates, then evaluate all locked tasks."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "runs/setup/baseline-full-8gpu-active.json"
PILOT = ROOT / "runs/yoco-160m-baseline-8gpu-pilot-20261104T0109Z"
METRICS = {"arc_easy": "acc_norm,none", "arc_challenge": "acc_norm,none", "boolq": "acc,none",
           "hellaswag": "acc_norm,none", "openbookqa": "acc_norm,none", "piqa": "acc_norm,none",
           "winogrande": "acc,none", "sciq": "acc_norm,none"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    release = run / "release"
    if ACTIVE.exists():
        active = json.loads(ACTIVE.read_text())
        if Path(active["run_dir"]).resolve() != run:
            raise RuntimeError(f"A baseline full run is already recorded: {active['run_dir']}; inspect it before starting another.")
    manifest_path = ROOT / "data/fineweb_edu_10bt/manifest.json"
    verified = json.loads((ROOT / "runs/setup/training-data-verification.json").read_text())
    assert verified["passed"] and verified["data_verified"] and digest(manifest_path) == verified["manifest_sha256"]
    ready = json.loads((ROOT / "runs/setup/downstream-offline.json").read_text())
    assert ready["ready"] and ready["offline"] and set(ready["tasks"]) == set(METRICS)
    pilot_meta = json.loads((PILOT / "latest.json").read_text())
    pilot_status = json.loads((PILOT / "pilot-status.json").read_text())
    assert pilot_meta["step"] == 100 and pilot_status["state"] == "passed" and not pilot_status["synthetic"]
    assert pilot_status["resume_verified_all_ranks"] and digest(PILOT / pilot_meta["file"]) == pilot_meta["sha256"]
    if not (run / "release-manifest.json").exists():
        if release.exists():
            raise RuntimeError("An incomplete release exists; inspect it before continuing.")
        release.mkdir()
        for folder in ("runtime", "evaluation"):
            shutil.copytree(ROOT / folder, release / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (release / "configs").mkdir()
        (release / "tools").mkdir()
        shutil.copyfile(ROOT / "tools/prepare_downstream_data.py", release / "tools/prepare_downstream_data.py")
        shutil.copyfile(ROOT / "requirements.a100-py312.lock.txt", release / "requirements.lock.txt")
        config = json.loads((ROOT / "configs/yoco_160m_baseline_8gpu.json").read_text())
        config["experiment"] = "yoco_160m_baseline_8gpu_full_a100"
        config["data"]["manifest"] = str(manifest_path)
        config["status"] = "AUTHORIZED_FULL_BASELINE_40000"
        assert "memory_fusion" not in config
        t = config["training"]
        assert t["max_updates"] == 40000 and t["world_size"] == 8
        assert t["world_size"] * t["microbatch"] * t["gradient_accumulation"] * t["sequence_length"] == 262144
        assert t["world_size"] * t["microbatch"] * t["validation_batches"] == 2048
        atomic_json(release / "configs/baseline.json", config)
        source_hashes = {str(p.relative_to(release / "runtime")): digest(p)
                         for p in sorted((release / "runtime").rglob("*.py"))}
        identity = {"model": config["model"], "tied_embeddings": config["tied_embeddings"], "seed": config["seed"],
                    "training": t, "data": json.loads(manifest_path.read_text()), "synthetic": False,
                    "world_size": 8, "model_source_sha256": source_hashes}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        assert fingerprint == pilot_meta["fingerprint"], "Continuation fingerprint differs from the tested pilot"
        shutil.copyfile(PILOT / pilot_meta["file"], run / pilot_meta["file"])
        assert digest(run / pilot_meta["file"]) == pilot_meta["sha256"]
        atomic_json(run / "checkpoint_00000100.json", pilot_meta)
        atomic_json(run / "latest.json", pilot_meta)
        hashes = {str(p.relative_to(release)): digest(p) for p in sorted(release.rglob("*")) if p.is_file()}
        record = {"created_at": now(), "source_pilot": str(PILOT), "resume_step": 100,
                  "resume_sha256": pilot_meta["sha256"], "fingerprint": fingerprint,
                  "manifest_sha256": verified["manifest_sha256"], "files": hashes}
        atomic_json(run / "release-manifest.json", record)
    release_manifest = json.loads((run / "release-manifest.json").read_text())
    assert all(digest(release / filename) == expected for filename, expected in release_manifest["files"].items())
    active = {"run_dir": str(run), "release": str(release), "status": str(run / "pipeline-status.json"),
              "heartbeat": str(run / "heartbeat.json"), "tmux_session": "yoco-baseline-full-8gpu",
              "world_size": 8, "max_updates": 40000, "resume_step": 100, "recorded_at": now()}
    atomic_json(ACTIVE, active)
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    lock = (ACTIVE.parent / "baseline-full-8gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("The baseline full pipeline is already running.")
    release = prepare(run)
    if args.prepare_only:
        print(f"Prepared release and verified step-100 continuation: {run}", flush=True)
        return
    if (run / "PIPELINE_SUCCESS").exists():
        print("The complete baseline pipeline already passed; nothing to launch.", flush=True)
        return
    os.environ.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      HF_HOME=str(ROOT / "data/cache/huggingface"), TIKTOKEN_CACHE_DIR=str(ROOT / "data/cache/tiktoken"),
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      MPLCONFIGDIR=str(ROOT / "data/cache/matplotlib"), YOCO_ENV_PREFIX=str(Path(sys.executable).parent.parent))
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    status = {"state": "starting", "run_dir": str(run), "pid": os.getpid(), "started_at": now(),
              "source_pilot": str(PILOT), "resume_step": 100, "target_step": 40000,
              "world_size": 8, "synthetic": False, "stages": {}}
    guard = threading.Lock()

    def update(stage, state, **fields):
        with guard:
            status["state"] = stage if state == "running" else status["state"]
            status["updated_at"] = now()
            status["stages"][stage] = {"state": state, **fields}
            atomic_json(run / "pipeline-status.json", status)
            print(json.dumps({"time": now(), "stage": stage, "state": state, **fields}), flush=True)

    def command(stage, argv, env=None):
        log_path = run / (stage + ".log")
        update(stage, "running", log=str(log_path))
        with log_path.open("a") as output:
            result = subprocess.run(argv, cwd=release, env=env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            update(stage, "failed", returncode=result.returncode, log=str(log_path))
            raise RuntimeError(f"{stage} failed; inspect {log_path}")
        update(stage, "passed", log=str(log_path))

    config = release / "configs/baseline.json"
    locks = release / "evaluation/plan_ab_data_locks"
    try:
        usage = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        assert len(usage.splitlines()) == 8 and all(int(x.strip()) < 1000 for x in usage.splitlines()), "GPUs are busy"
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count() == 8
        status["devices"] = [torch.cuda.get_device_name(i) for i in range(8)]
        status["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "tiktoken", "datasets", "lm_eval")}
        update("preflight", "passed", fingerprint_verified=True)
        command("offline-data", [sys.executable, "tools/prepare_downstream_data.py", "--offline",
                                  "--output", str(run / "downstream-offline.json")])
        command("evaluation-smoke", [sys.executable, "evaluation/run.py", "--config", str(config),
            "--checkpoint", str(run / "checkpoint_00000100.pt"), "--locks-dir", str(locks),
            "--task", "arc_easy", "--limit", "4", "--device", "cuda:0", "--output", str(run / "eval-smoke")])
        command("training", [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                             "-m", "runtime.train", "--config", str(config), "--run-dir", str(run), "--stop-after", "40000"])
        stage = json.loads((run / "stage_complete.json").read_text())
        assert stage == {"step": 40000, "formal_complete": True, "synthetic": False}
        latest = json.loads((run / "latest.json").read_text())
        checkpoint = run / latest["file"]
        assert latest["step"] == 40000 and digest(checkpoint) == latest["sha256"]
        expected_counts = json.loads((run / "downstream-offline.json").read_text())["tasks"]

        def evaluate(item):
            device, task = item
            output = run / "evaluation" / task
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
            command("eval-" + task, [sys.executable, "evaluation/run.py", "--config", str(config),
                "--checkpoint", str(checkpoint), "--locks-dir", str(locks), "--task", task,
                "--device", "cuda:0", "--output", str(output)], env=env)
            result = json.loads((output / "results.json").read_text())
            provenance = json.loads((output / "provenance.json").read_text())
            assert result["model_step"] == 40000 and not result["smoke_only"] and provenance["limit"] is None
            assert provenance["checkpoint_sha256"] == latest["sha256"]
            assert provenance["dataset_lock"] == json.loads((locks / f"data_lock_{task}.json").read_text())
            assert result["n-samples"][task]["effective"] == expected_counts[task]["evaluation_documents"]
            score = result["results"][task][METRICS[task]]
            assert math.isfinite(score) and 0 <= score <= 1
            return {"task": task, "metric": METRICS[task], "score": score,
                    "documents": expected_counts[task]["evaluation_documents"]}

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            scores = list(pool.map(evaluate, enumerate(METRICS)))
        summary = {"complete": True, "step": 40000, "checkpoint": str(checkpoint), "sha256": latest["sha256"],
                   "world_size": 8, "global_batch_tokens": 262144, "total_training_tokens": 40000 * 262144,
                   "tasks": scores, "macro_accuracy": sum(row["score"] for row in scores) / len(scores), "completed_at": now()}
        atomic_json(run / "summary.json", summary)
        status["state"] = "complete"
        update("pipeline", "passed", summary=str(run / "summary.json"))
        (run / "PIPELINE_SUCCESS").touch()
    except BaseException as error:
        status["state"] = "failed"
        update("pipeline", "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
