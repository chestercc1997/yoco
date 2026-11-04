"""Resumable local preparation; never launches formal training or Slurm jobs."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "runs/setup"
LOGS = ROOT / "logs/setup"
STATE = SETUP / "local-preparation.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-workers", type=int, default=14)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--tokenize-workers", type=int, default=14)
    parser.add_argument("--gpu-smoke", action="store_true", help="Requires GPU device access")
    args = parser.parse_args()
    SETUP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (SETUP / "local-preparation.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Local preparation is already running; inspect its status file.")
    os.chdir(ROOT)
    os.environ.update(HF_HOME=str(ROOT / "data/cache/huggingface"),
                      TIKTOKEN_CACHE_DIR=str(ROOT / "data/cache/tiktoken"),
                      MPLCONFIGDIR=str(ROOT / "data/cache/matplotlib"),
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(key, None)
    status = {"pid": os.getpid(), "python": sys.executable, "state": "running", "stages": {},
              "formal_training_started": False, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    state_lock = threading.Lock()

    def update(stage, state, **fields):
        with state_lock:
            status["stages"][stage] = {"state": state, **fields}
            status["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            temporary = STATE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(STATE)
            print(json.dumps({"stage": stage, "state": state, **fields}), flush=True)

    def command(stage, argv, attempts=1):
        update(stage, "running", log=str(LOGS / (stage + ".log")))
        with (LOGS / (stage + ".log")).open("a") as log:
            for attempt in range(attempts):
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    update(stage, "passed")
                    return
                if attempt + 1 < attempts:
                    time.sleep(min(30, 5 * (attempt + 1)))
            update(stage, "failed", returncode=result.returncode)
            raise RuntimeError(f"{stage} failed; inspect {LOGS / (stage + '.log')}")

    def dependencies():
        plan = json.loads((SETUP / "pip-plan.json").read_text())
        pending = []
        for row in plan["install"]:
            data = row["download_info"]
            filename = unquote(urlsplit(data["url"]).path.rsplit("/", 1)[-1])
            path = ROOT / "wheelhouse" / filename
            # aria2 control files distinguish sparse, incomplete files from finished archives.
            if path.is_file() and not Path(str(path)+".aria2").exists() and digest(path) == data["archive_info"]["hashes"]["sha256"]:
                continue
            pending.extend([data["url"], "  out="+filename,
                            "  checksum=sha-256="+data["archive_info"]["hashes"]["sha256"]])
        if pending:
            inputs = SETUP / "pip-pending.aria2"
            inputs.write_text("\n".join(pending) + "\n")
            command("dependency-download", ["aria2c", "--input-file="+str(inputs),
                "--dir="+str(ROOT / "wheelhouse"), "--max-concurrent-downloads=12",
                "--max-connection-per-server=8", "--split=8", "--min-split-size=8M",
                "--file-allocation=none", "--continue=true", "--auto-file-renaming=false",
                "--check-integrity=true", "--console-log-level=warn", "--summary-interval=60"], attempts=4)
        command("dependency-install", [sys.executable, "-m", "pip", "install", "--no-index",
            "--find-links="+str(ROOT / "wheelhouse"), "--no-build-isolation", "-r", "requirements.txt",
            "peft==0.15.2", "accelerate==1.8.1"])
        command("dependency-check", [sys.executable, "-m", "pip", "check"])
        pinned = {"torch": "2.7.1", "numpy": "2.1.2", "tiktoken": "0.11.0", "pyarrow": "20.0.0",
                  "huggingface-hub": "0.33.2", "datasets": "3.6.0", "transformers": "4.53.1", "lm_eval": "0.4.5"}
        versions = {name: importlib.metadata.version(name) for name in pinned}
        assert all(versions[name].split("+")[0] == version for name, version in pinned.items()), versions
        (ROOT / "requirements.a100-py312.lock.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True))
        update("environment", "passed", versions=versions)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    download = pool.submit(command, "training-download", [sys.executable, "-u", "tools/download_pinned_data.py",
        "--workers", str(args.download_workers), "--connections", str(args.connections)], 4)
    try:
        dependencies()
        command("model-contracts", [sys.executable, "tests/test_plan_ab_kv.py"])
        command("initial-models", [sys.executable, "tools/prepare_local_models.py"])
        command("downstream-online", [sys.executable, "tools/prepare_downstream_data.py",
            "--output", "runs/setup/downstream-online.json"], attempts=3)
        command("downstream-offline", [sys.executable, "tools/prepare_downstream_data.py", "--offline",
            "--output", "runs/setup/downstream-offline.json"])
        if args.gpu_smoke:
            try:
                command("gpu-smoke", ["bash", "scripts/smoke_local_a100.sh"])
            except RuntimeError as error:
                update("gpu-smoke", "failed", error=str(error))
        else:
            update("gpu-smoke", "pending", reason="GPU device access required outside sandbox")
        download.result()
        data_config = SETUP / "data-preparation-config.json"
        shutil.copyfile(ROOT / "configs/yoco_160m_plan_ab_A_4gpu.json", data_config)
        command("tokenization", [sys.executable, "tools/prepare_data.py", "--offline", "--config", str(data_config),
                                 "--workers", str(args.tokenize_workers)])
        command("training-data-verification", [sys.executable, "tools/preflight_4gpu.py", "--output",
                                                "runs/setup/training-data-verification.json"])
        for arm, filename in (("baseline", "yoco_160m_plan_full_baseline_4gpu.json"),
                               ("A", "yoco_160m_plan_ab_A_4gpu.json"), ("B", "yoco_160m_plan_ab_B_4gpu.json")):
            command("manifest-"+arm, [sys.executable, "tools/verify_data.py", "--config", "configs/"+filename])
        status["state"] = "ready" if status["stages"]["gpu-smoke"]["state"] == "passed" else "data_and_environment_ready_gpu_unverified"
        update("preparation", "passed", formal_training_started=False)
    except BaseException as error:
        status["state"] = "failed"
        update("preparation", "failed", error=str(error))
        raise
    finally:
        pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
