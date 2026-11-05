"""Validate and train only method A at 400M, then run all eight locked tasks."""
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
ACTIVE = ROOT / "runs/setup/400m-A-active.json"
INITIALIZATION = ROOT / "runs/setup/yoco-400m-A-initialization.pt"
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
    run.mkdir(parents=True,exist_ok=True)
    release=run/"release"
    if ACTIVE.exists() and Path(json.loads(ACTIVE.read_text())["run_dir"]).resolve()!=run:
        raise RuntimeError("A 400M A run is already recorded; inspect before creating another")
    config=json.loads((ROOT/"configs/yoco_400m_plan_A_8gpu.json").read_text())
    reference=json.loads((ROOT/"configs/yoco_160m_baseline_8gpu_matched.json").read_text())
    preflight=json.loads((ROOT/"runs/setup/yoco-400m-A-preflight.json").read_text())
    assert preflight['passed'] and preflight['A_parameters']==400590860
    assert preflight['common_weights_bitwise_equal'] and preflight['last_source_limit_matches_baseline']
    assert config['training']==reference['training'] and config['data']==reference['data']
    assert config['model']==preflight['model'] and config['memory_fusion']==preflight['memory_fusion']
    assert config['training']['precision']=='fp16_mixed_fp32_params'
    speed_path=ROOT/'runs/diagnostics/400m-A-speed-20261105/numerical-summary.json'
    speed=json.loads(speed_path.read_text())
    measured={row['case']:row for row in speed['records']}
    chosen=speed['selected']
    assert chosen and measured[chosen]['accepted'],'No verified 400M acceleration candidate'
    assert measured[chosen]['process_clean_exit'] and measured['eager']['process_clean_exit']
    assert measured[chosen]['numerical_checks_passed'],'400M acceleration numerical check failed'
    assert measured['eager']['median_seconds']/measured[chosen]['median_seconds']>=1.05
    assert config['execution']=={k:v for k,v in measured[chosen]['execution'].items() if k!='probe_settings'}
    assert digest(INITIALIZATION)==preflight['initialization_sha256']
    if not (run/'release-manifest.json').exists():
        if release.exists():raise RuntimeError('Incomplete release; inspect before continuing')
        release.mkdir()
        for folder in ('runtime','evaluation'):
            shutil.copytree(ROOT/folder,release/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (release/'configs').mkdir();(release/'tools').mkdir()
        shutil.copyfile(ROOT/'tools/prepare_downstream_data.py',release/'tools/prepare_downstream_data.py')
        shutil.copyfile(ROOT/'requirements.a100-py312.lock.txt',release/'requirements.lock.txt')
        atomic_json(release/'configs/A.json',config)
        atomic_json(release/'configs/baseline-reference.json',json.loads((ROOT/'configs/yoco_400m_baseline_8gpu.json').read_text()))
        shutil.copyfile(INITIALIZATION,run/'initialization.pt')
        hashes={str(p.relative_to(release)):digest(p) for p in sorted(release.rglob('*')) if p.is_file()}
        atomic_json(run/'release-manifest.json',{'created_at':now(),'files':hashes,
            'initialization_sha256':preflight['initialization_sha256'],
            'manifest_sha256':digest(Path(config['data']['manifest'])),
            'preflight':preflight,'training_config':config['training'],'execution':config['execution'],
            'acceleration_validation':speed})
        atomic_json(run/'model-preflight.json',preflight)
        atomic_json(run/'acceleration-validation.json',speed)
    record=json.loads((run/'release-manifest.json').read_text())
    assert all(digest(release/name)==expected for name,expected in record['files'].items())
    assert digest(run/'initialization.pt')==record['initialization_sha256']
    atomic_json(ACTIVE,{'run_dir':str(run),'release':str(release),'status':str(run/'pipeline-status.json'),
        'heartbeat':str(run/'heartbeat.json'),'tmux_session':'yoco-400m-A-8gpu',
        'world_size':8,'method':'A','parameters':400590860,'max_updates':40000,
        'initial_step':0,'formal_resume_step':4,'precision':'fp16_mixed_fp32_params',
        'checkpoint_every':5000,'keep_all_checkpoints':True,'recorded_at':now()})
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    lock = (ACTIVE.parent / "400m-A-8gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("The 400M method A pipeline is already running.")
    release = prepare(run)
    if args.prepare_only:
        print(f"Prepared immutable 400M A release and verified initialization: {run}",flush=True)
        return
    if (run / "PIPELINE_SUCCESS").exists():
        print("The complete 400M method A pipeline already passed; nothing to launch.", flush=True)
        return
    os.environ.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      NCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo", TORCHINDUCTOR_COMPILE_THREADS="2",
                      TORCHINDUCTOR_CACHE_DIR=str(ROOT / "data/cache/torchinductor"),
                      HF_HOME=str(ROOT / "data/cache/huggingface"), TIKTOKEN_CACHE_DIR=str(ROOT / "data/cache/tiktoken"),
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      MPLCONFIGDIR=str(ROOT / "data/cache/matplotlib"), YOCO_ENV_PREFIX=str(Path(sys.executable).parent.parent))
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    status = {"state": "starting", "run_dir": str(run), "pid": os.getpid(), "started_at": now(),
              "single_node_transport":{"NCCL_SOCKET_IFNAME":"lo","GLOO_SOCKET_IFNAME":"lo"},
              "initialization": str(run / "initialization.pt"), "resume_step": 4, "target_step": 40000,
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

    config = release / "configs/A.json"
    locks = release / "evaluation/plan_ab_data_locks"
    try:
        usage = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        assert len(usage.splitlines()) == 8 and all(int(x.strip()) < 1000 for x in usage.splitlines()), "GPUs are busy"
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count() == 8
        status["devices"] = [torch.cuda.get_device_name(i) for i in range(8)]
        status["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "tiktoken", "datasets", "lm_eval")}
        update("preflight", "passed", fingerprint_verified=True)
        if not (run / 'downstream-offline.json').exists():
            command("offline-data", [sys.executable, "tools/prepare_downstream_data.py", "--offline",
                                      "--output", str(run / "downstream-offline.json")])
        ready=json.loads((run / 'downstream-offline.json').read_text())
        assert ready['ready'] and ready['offline']
        source = run / 'checks/resumed/checkpoint_00000004.pt'
        train_command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                         "-m", "runtime.train", "--config", str(config)]
        if not (run / "resume-verification.json").exists():
            split = run / "checks/resumed"
            direct = run / "checks/uninterrupted"
            if not (split / 'checkpoint_00000002.json').exists():
                command("check-first", train_command + ["--run-dir",str(split),"--init-model",str(run/"initialization.pt"),"--stop-after","2"])
            if not (split / 'checkpoint_00000004.json').exists():
                command("check-resume", train_command + ["--run-dir",str(split),"--init-model",str(run/"initialization.pt"),"--stop-after","4"])
            if not (direct / 'checkpoint_00000004.json').exists():
                command("check-direct", train_command + ["--run-dir",str(direct),"--init-model",str(run/"initialization.pt"),"--stop-after","4"])
            repeat = run / "checks/uninterrupted-repeat"
            if not (repeat / 'checkpoint_00000004.json').exists():
                command("check-repeat", train_command + ["--run-dir",str(repeat),"--init-model",str(run/"initialization.pt"),"--stop-after","4"])
            a = torch.load(split / "checkpoint_00000004.pt", map_location="cpu", weights_only=False, mmap=True)
            b = torch.load(direct / "checkpoint_00000004.pt", map_location="cpu", weights_only=False, mmap=True)
            repeated = torch.load(repeat / "checkpoint_00000004.pt", map_location="cpu", weights_only=False, mmap=True)
            import numpy as np
            torch.set_num_threads(4)
            assert a["step"] == b["step"] == 4 and a["world_size"] == b["world_size"] == 8
            assert a["scaler"] == b["scaler"] and a["scaler"]["scale"] > 0
            worst = 0.
            def compare(left, right, path=''):
                nonlocal worst
                if isinstance(left, torch.Tensor):
                    if not left.is_floating_point() or path.endswith('/step'):
                        assert torch.equal(left,right), f"Exact resume state differs: {path}"
                    else:
                        # AMP kernels need a rounding tolerance; counters/RNG remain exact.
                        tolerance=2e-5 if path.startswith('model/') else 1e-6
                        assert torch.allclose(left,right,rtol=1e-3,atol=tolerance), f"Resume state differs: {path}"
                        worst=max(worst,float((left-right).abs().max()))
                elif isinstance(left,np.ndarray): assert np.array_equal(left,right)
                elif isinstance(left,dict):
                    assert left.keys() == right.keys()
                    for key in left: compare(left[key],right[key],path+'/'+str(key))
                elif isinstance(left,(list,tuple)):
                    assert len(left) == len(right)
                    for i,(x,y) in enumerate(zip(left,right)): compare(x,y,path+'/'+str(i))
                else: assert left == right
            compare(a["model"],b["model"],'model'); compare(a["optimizer"],b["optimizer"],'optimizer'); compare(a["rngs"],b["rngs"],'rngs')
            def model_difference(left,right):
                diff=reference=0.; maximum=0.
                for key in left:
                    delta=left[key]-right[key]
                    maximum=max(maximum,float(delta.abs().max()))
                    diff+=float(delta.double().square().sum());reference+=float(right[key].double().square().sum())
                return {'max_absolute_difference':maximum,'relative_l2':math.sqrt(diff/reference)}
            resumed_difference=model_difference(a['model'],b['model'])
            repeat_difference=model_difference(repeated['model'],b['model'])
            assert resumed_difference['relative_l2']<2e-6 and resumed_difference['max_absolute_difference']<2e-5
            assert repeat_difference['relative_l2']<2e-6 and repeat_difference['max_absolute_difference']<2e-5
            def training_losses(directory):
                return {e['step']:e['loss'] for e in map(json.loads,(directory/'rank0.jsonl').read_text().splitlines()) if e['event']=='train'}
            split_losses=training_losses(split); direct_losses=training_losses(direct)
            loss_difference=max(abs(split_losses[step]-direct_losses[step]) for step in direct_losses)
            assert loss_difference<5e-5
            del a,b,repeated
            for rank in range(8):
                events=[json.loads(line) for line in (split / f"rank{rank}.jsonl").read_text().splitlines()]
                assert [e["step"] for e in events if e["event"] == "started"] == [0,2]
                assert not any(e["event"] == "overflow" for e in events)
                assert [e["step"] for e in events if e["event"] == "train"] == [1,2,3,4]
            atomic_json(run / "resume-verification.json", {"passed":True,"world_size":8,"real_data":True,
                "source_step":0,"split_resume_step":2,"final_step":4,
                "all_rank_rng_verified":True,
                "fp16_numerical_tolerance":{'model_max_absolute':2e-5,'model_relative_l2':2e-6,'optimizer_atol':1e-6,'loss_absolute':5e-5},
                "resumed_model_difference":resumed_difference,"repeat_model_difference":repeat_difference,
                "max_training_loss_difference":loss_difference,
                "model_and_optimizer_max_absolute_difference":worst,"completed_at":now()})
            update("eight-gpu-resume-check", "passed", max_absolute_difference=worst)
        assert source.exists()
        if not (run/'eval-smoke/results.json').exists():
            command('evaluation-smoke',[sys.executable,'evaluation/run.py','--config',str(config),
                '--checkpoint',str(source),'--locks-dir',str(locks),'--task','arc_easy','--limit','4',
                '--device','cuda:0','--output',str(run/'eval-smoke')])
        smoke=json.loads((run/'eval-smoke/results.json').read_text())
        assert smoke['model_step']==4 and smoke['smoke_only']
        command("training", train_command + ["--run-dir",str(run),"--resume-from",str(source),"--stop-after","40000"])
        stage = json.loads((run / "stage_complete.json").read_text())
        assert stage == {"step": 40000, "formal_complete": True, "synthetic": False}
        latest = json.loads((run / "latest.json").read_text())
        checkpoint = run / latest["file"]
        assert latest["step"] == 40000 and digest(checkpoint) == latest["sha256"]
        training_events=[json.loads(line) for folder in (run/'checks/resumed',run)
                         for line in (folder/'rank0.jsonl').read_text().splitlines()]
        updates={e['step']:e for e in training_events if e['event']=='train'}
        assert sorted(updates)==list(range(1,40001))
        validations={e['step']:e for e in training_events if e['event']=='validation' and e['step']%250==0}
        assert sorted(validations)==list(range(250,40001,250))
        best=float('inf');maximum_regression=0.
        for step in sorted(validations):
            value=validations[step]['loss'];assert math.isfinite(value)
            best=min(best,value);maximum_regression=max(maximum_regression,value-best)
        review={'final_validation_loss':validations[40000]['loss'],'best_validation_loss':best,
            'maximum_validation_regression':maximum_regression,
            'overflow_retries':sum(e['event']=='overflow' for e in training_events),
            'maximum_gradient_norm':max(e['grad_norm'] for e in updates.values()),
            'convergence_warning':validations[40000]['loss']-best>.1,
            'note':'Pipeline completion proves training/evaluation completion; assess convergence separately.'}
        atomic_json(run/'training-review.json',review)
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
                   "world_size": 8, "method":"A", "parameters":400590860, "global_batch_tokens": 262144, "total_training_tokens": 40000 * 262144,
                   "tasks": scores, "macro_accuracy": sum(row["score"] for row in scores) / len(scores), "completed_at": now()}
        summary['training_review']=review
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
