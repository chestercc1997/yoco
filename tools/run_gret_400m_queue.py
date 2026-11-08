"""Persistently start the queued baseline after the complete gRet 400M B pipeline."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_swa_ab import read, atomic_json, now, digest
from tools.watch_swa_training import process_identity, same_process

TASK_COUNTS = {'arc_easy':2376, 'arc_challenge':1172, 'boolq':3270, 'hellaswag':10042,
               'openbookqa':500, 'piqa':1838, 'winogrande':1267, 'sciq':1000}
QUEUE = ROOT / 'runs/setup/gret-400m-training-queue.json'


def predecessor_ready(run):
    """Training alone or partial/smoke downstream results cannot release the queue."""
    run = Path(run)
    if not all((run / name).exists() for name in ('PIPELINE_SUCCESS', 'summary.json', 'B/PIPELINE_SUCCESS', 'B/summary.json')):
        return False
    root = read(run / 'summary.json');b = read(run / 'B/summary.json')
    if not (root.get('complete') and b.get('complete') and b.get('step') == 40000
            and b.get('world_size') == 8 and b.get('architecture') == 'YOCO-gRet' and b.get('parameters') == 400591884):
        return False
    tasks = b.get('tasks', [])
    if len(tasks) != 8 or {t['task']:t['documents'] for t in tasks} != TASK_COUNTS:return False
    report = Path(root['report_output']) / 'summary.json'
    return report.exists() and read(report).get('complete') is True


def gpus_idle():
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    values = usage.splitlines()
    return len(values) == 8 and all(int(x.strip()) < 1000 for x in values)


def existing_guard(run):
    """Recover ownership if the queue stopped between spawning and recording a guard."""
    matches = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:argv = [v.decode(errors='replace') for v in (directory / 'cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError, PermissionError, ProcessLookupError):continue
        if (str(ROOT / 'tools/watch_gret_400m_baseline.py') in argv and '--run-dir' in argv
                and argv[argv.index('--run-dir')+1] == str(run)):
            matches.append(process_identity(int(directory.name)))
    matches = [p for p in matches if p]
    assert len(matches) <= 1, 'Duplicate baseline supervisors require diagnosis'
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    lock = args.queue.with_suffix('.lock').open('a')
    try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('The gRet training queue is already running')
    while True:
        registry = read(args.queue)
        for task in registry['tasks']:
            if task['kind'] != 'gret-400m-baseline':raise ValueError('Unknown queued task kind')
            if task['state'] in ('cancelled','pipeline_complete_review_pending','needs_diagnosis'):continue
            run = Path(task['run_dir'])
            if (run / 'PIPELINE_SUCCESS').exists():
                assert read(run / 'summary.json')['complete']
                task.update(state='pipeline_complete_review_pending', completed_at=now())
                continue
            identity = task.get('guard_process')
            if identity:
                if same_process(identity):task['state'] = 'running'
                else:
                    task.update(state='needs_diagnosis', reason='Supervisor exited before pipeline success; inspect guard-state.json, do not blindly restart')
                continue
            if not task.get('prepared'):
                task['state'] = 'preparing';continue
            identity = existing_guard(run)
            if identity:
                task.update(state='running', guard_process=identity);continue
            assert read(run / 'release-manifest.json')['source_B_run'] == task['after_run']
            if not predecessor_ready(task['after_run']):
                task['state'] = 'waiting_for_predecessor';continue
            if not gpus_idle():
                task['state'] = 'waiting_for_idle_gpus';continue
            if (run / 'pipeline-status.json').exists():
                task.update(state='needs_diagnosis', reason='Existing baseline pipeline without a live supervisor; preserve evidence')
                continue
            else:
                with (run / 'guard.log').open('a') as output:
                    child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'tools/watch_gret_400m_baseline.py'),
                        '--run-dir', str(run)], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_identity(child.pid)
                assert identity is not None
            task.update(state='running', guard_process=identity, started_at=now(),
                        predecessor_summary_sha256=digest(Path(task['after_run']) / 'summary.json'))
        registry['heartbeat_at'] = now()
        atomic_json(args.queue, registry)
        print(registry['heartbeat_at'], [(t['id'],t['state']) for t in registry['tasks']], flush=True)
        if all(t['state'] in ('cancelled','pipeline_complete_review_pending','needs_diagnosis') for t in registry['tasks']):return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':main()
