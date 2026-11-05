"""Verify and run the explicitly authorized FP32 baseline stability rerun."""
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
sys.path.insert(0, str(ROOT))
ACTIVE = ROOT / "runs/setup/baseline-stable-8gpu-active.json"
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
    if ACTIVE.exists() and Path(json.loads(ACTIVE.read_text())["run_dir"]).resolve() != run:
        raise RuntimeError("A stability rerun is already recorded; inspect it before starting another")
    from runtime.train import verify_fork
    manifest_path = ROOT / "data/fineweb_edu_10bt/manifest.json"
    config = json.loads((ROOT / "configs/yoco_160m_baseline_8gpu_fp32.json").read_text())
    source = PILOT / "checkpoint_00000100.pt"
    provenance = verify_fork(source, config, json.loads(manifest_path.read_text()), 8)
    assert config["training"]["checkpoint_every"] == 5000
    assert config["training"]["keep_last_checkpoints"] is None
    assert config["training"]["precision"] == "fp32"
    assert "memory_fusion" not in config
    if not (run / "release-manifest.json").exists():
        if release.exists(): raise RuntimeError("Incomplete release: inspect before continuing")
        release.mkdir()
        for folder in ("runtime", "evaluation"):
            shutil.copytree(ROOT / folder, release / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (release / "configs").mkdir()
        (release / "tools").mkdir()
        shutil.copyfile(ROOT / "tools/prepare_downstream_data.py", release / "tools/prepare_downstream_data.py")
        shutil.copyfile(ROOT / "requirements.a100-py312.lock.txt", release / "requirements.lock.txt")
        atomic_json(release / "configs/baseline.json", config)
        hashes = {str(p.relative_to(release)):digest(p) for p in sorted(release.rglob("*")) if p.is_file()}
        atomic_json(run / "release-manifest.json", {"created_at":now(), "files":hashes,
            "fork":provenance, "manifest_sha256":digest(manifest_path),
            "historical_run_preserved":str(ROOT / "runs/yoco-160m-baseline-8gpu-full-20261104T0114Z")})
    record = json.loads((run / "release-manifest.json").read_text())
    assert all(digest(release / name) == expected for name, expected in record["files"].items())
    atomic_json(ACTIVE, {"run_dir":str(run), "release":str(release), "status":str(run / "pipeline-status.json"),
        "heartbeat":str(run / "heartbeat.json"), "tmux_session":"yoco-baseline-stable-8gpu",
        "world_size":8, "max_updates":40000, "resume_step":100, "precision":"fp32",
        "checkpoint_every":5000, "keep_all_checkpoints":True, "recorded_at":now()})
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    lock = (ACTIVE.parent / "baseline-stable-8gpu.lock").open("a")
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
                      NCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo",
                      HF_HOME=str(ROOT / "data/cache/huggingface"), TIKTOKEN_CACHE_DIR=str(ROOT / "data/cache/tiktoken"),
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      MPLCONFIGDIR=str(ROOT / "data/cache/matplotlib"), YOCO_ENV_PREFIX=str(Path(sys.executable).parent.parent))
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    status = {"state": "starting", "run_dir": str(run), "pid": os.getpid(), "started_at": now(),
              "single_node_transport":{"NCCL_SOCKET_IFNAME":"lo","GLOO_SOCKET_IFNAME":"lo"},
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
            "--checkpoint", str(PILOT / "checkpoint_00000100.pt"), "--locks-dir", str(locks),
            "--task", "arc_easy", "--limit", "4", "--device", "cuda:0", "--output", str(run / "eval-smoke")])
        source = PILOT / "checkpoint_00000100.pt"
        train_command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                         "-m", "runtime.train", "--config", str(config)]
        if not (run / "resume-verification.json").exists():
            split = run / "checks/resumed"
            direct = run / "checks/uninterrupted"
            if not (split / 'checkpoint_00000102.json').exists():
                command("check-first", train_command + ["--run-dir",str(split),"--resume-from",str(source),"--stop-after","102"])
            if not (split / 'checkpoint_00000104.json').exists():
                command("check-resume", train_command + ["--run-dir",str(split),"--resume-from",str(source),"--stop-after","104"])
            if not (direct / 'checkpoint_00000104.json').exists():
                command("check-direct", train_command + ["--run-dir",str(direct),"--resume-from",str(source),"--stop-after","104"])
            a = torch.load(split / "checkpoint_00000104.pt", map_location="cpu", weights_only=False, mmap=True)
            b = torch.load(direct / "checkpoint_00000104.pt", map_location="cpu", weights_only=False, mmap=True)
            import numpy as np
            assert a["step"] == b["step"] == 104 and a["world_size"] == b["world_size"] == 8
            assert a["scaler"] == b["scaler"] == {}
            worst = 0.
            def compare(left, right):
                nonlocal worst
                if isinstance(left, torch.Tensor):
                    assert torch.allclose(left,right,rtol=1e-5,atol=1e-7), "Resume state differs"
                    if left.is_floating_point(): worst=max(worst,float((left-right).abs().max()))
                elif isinstance(left,np.ndarray): assert np.array_equal(left,right)
                elif isinstance(left,dict):
                    assert left.keys() == right.keys()
                    for key in left: compare(left[key],right[key])
                elif isinstance(left,(list,tuple)):
                    assert len(left) == len(right)
                    for x,y in zip(left,right): compare(x,y)
                else: assert left == right
            compare(a["model"],b["model"]); compare(a["optimizer"],b["optimizer"]); compare(a["rngs"],b["rngs"])
            del a,b
            for rank in range(8):
                events=[json.loads(line) for line in (split / f"rank{rank}.jsonl").read_text().splitlines()]
                assert [e["step"] for e in events if e["event"] == "started"] == [100,102]
                assert not any(e["event"] == "overflow" for e in events)
                assert [e["step"] for e in events if e["event"] == "train"] == [101,102,103,104]
            atomic_json(run / "resume-verification.json", {"passed":True,"world_size":8,"real_data":True,
                "source_step":100,"split_resume_step":102,"final_step":104,
                "all_rank_rng_verified":True,
                "model_and_optimizer_max_absolute_difference":worst,"completed_at":now()})
            update("eight-gpu-resume-check", "passed", max_absolute_difference=worst)
        command("training", train_command + ["--run-dir",str(run),"--resume-from",str(source),"--stop-after","40000"])
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
    raise SystemExit('FP32 rerun retired at user request. Use tools/run_baseline_matched.py with the A/B FP16 recipe.')
