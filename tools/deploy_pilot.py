"""Local tmux controller: verify, transfer, verify again, submit one GPU pilot."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/data/share/jzhou945/cchen099/yoco'
SKILL = Path('/hpc2hdd/home/jzhou945/cchen099/.codex-home/skills/hpc3')
STATE = ROOT / 'runs/setup/pilot-controller-20261030.json'
RUN = 'runs/yoco-160m-4h100-pilot-20261030'
ENV = os.environ.copy()
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    ENV.pop(key, None)
ENV.update(NO_PROXY='*', no_proxy='*', SSH_ASKPASS=str(SKILL/'scripts/askpass.py'),
           SSH_ASKPASS_REQUIRE='force', DISPLAY='hpc3-auth',
           TIKTOKEN_CACHE_DIR=str(ROOT/'data/cache/tiktoken'))


def status(stage, **extra):
    record = dict(stage=stage, updated_unix=time.time(), pid=os.getpid(), **extra)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(STATE)
    print(json.dumps(record), flush=True)


def call(args, capture=False):
    result = subprocess.run([str(x) for x in args], cwd=ROOT, env=ENV,
                            stdin=subprocess.DEVNULL, start_new_session=True,
                            stdout=subprocess.PIPE if capture else None, text=True)
    if result.returncode and result.stdout:
        print(result.stdout, flush=True)
    result.check_returncode()
    return result.stdout


def remote(command):
    return call([sys.executable, SKILL/'scripts/remote.py', command], capture=True)


def scp(paths, destination):
    call(['scp', '-r', '-o', 'StrictHostKeyChecking=yes', '-o', 'GSSAPIAuthentication=no',
          '-o', 'NumberOfPasswordPrompts=1', '-o', 'ServerAliveInterval=30',
          '-o', 'ServerAliveCountMax=6', '-o', 'ConnectTimeout=30', *paths,
          'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+destination])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    lock = (ROOT/'runs/setup/pilot-controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists():
        prior = json.loads(STATE.read_text())
        if prior.get('job_id'):
            raise RuntimeError('A job was already submitted; inspect it instead of submitting again')
    status('WAITING_FOR_WHEELS')
    while not (ROOT/'runs/setup/wheels-ready.json').exists():
        time.sleep(30)
    status('DEPLOYING_CODE_AND_ENVIRONMENT')
    print(remote('mkdir -p '+REMOTE+'/data/cache '+REMOTE+'/runs/setup '+REMOTE+'/logs'), flush=True)
    archive = ROOT/'runs/setup/pilot-code-20261030.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in ('runtime', 'configs', 'tools', 'tests', 'scripts', 'provenance', 'evaluation/figures/training_efficiency_validation_points.csv', 'requirements.hpc3-training.txt'):
            tar.add(ROOT/name, arcname=name, filter=lambda x: None if '__pycache__' in x.name else x)
    scp([str(archive)], REMOTE+'/runs/setup/')
    assert remote('sha256sum '+REMOTE+'/runs/setup/'+archive.name).split()[0] == digest(archive)
    scp([str(ROOT/'data/cache/tiktoken')], REMOTE+'/data/cache/')
    scp([str(ROOT/'wheelhouse')], REMOTE+'/')
    print(remote('cd '+REMOTE+' && tar xzf runs/setup/pilot-code-20261030.tar.gz && '
                 'bash scripts/hpc3_container.sh .venv/bin/python -m pip install --no-index --find-links=wheelhouse -r requirements.hpc3-training.txt && '
                 'bash scripts/hpc3_container.sh .venv/bin/python -m pip check && '
                 'bash scripts/hpc3_container.sh .venv/bin/python -m pip freeze > runs/setup/requirements.hpc3.freeze.txt && '
                 'bash scripts/hpc3_container.sh .venv/bin/python tests/test_data_recovery.py'), flush=True)
    status('WAITING_FOR_TOKENIZATION')
    manifest = ROOT/'data/fineweb_edu_10bt/manifest.json'
    while not manifest.exists():
        alive = subprocess.run(['tmux', 'has-session', '-t', 'yoco-tokenize-20261030'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not alive:
            raise RuntimeError('Tokenizer stopped without a ready manifest; inspect tokenize-20261030.log')
        time.sleep(30)
    status('VERIFYING_LOCAL_CORPUS')
    call([ROOT/'.venv/bin/python', 'tools/preflight_4gpu.py', '--output', 'runs/setup/preflight-hpc2.json'])
    status('TRANSFERRING_VERIFIED_TOKENS')
    scp([str(ROOT/'data/fineweb_edu_10bt/tokens'), str(manifest)], REMOTE+'/data/fineweb_edu_10bt/')
    # Config and shard metadata have absolute local paths after preparation.
    # Rebase only path fields; token byte hashes remain unchanged.
    rebase = '''import json
from pathlib import Path
base=Path(REMOTE_ROOT)
folder=base/'data/fineweb_edu_10bt'
paths=[folder/'manifest.json']+sorted((folder/'tokens').glob('*.json'))
for path in paths:
    data=json.loads(path.read_text())
    rows=data['shards'] if 'shards' in data else [data]
    for row in rows:
        row['source_file']=str(folder/'raw/sample/10BT'/Path(row['source_file']).name)
        for split in ('train','validation'):
            row[split]['path']=str(folder/'tokens'/Path(row[split]['path']).name)
    path.write_text(json.dumps(data,indent=2)+'\\n')
for arch in ('yoco','transformer'):
    path=base/'configs'/(arch+'_160m_4gpu.json')
    data=json.loads(path.read_text())
    data['data']['manifest']='data/fineweb_edu_10bt/manifest.json'
    path.write_text(json.dumps(data,indent=2)+'\\n')
'''.replace('REMOTE_ROOT', repr(REMOTE))
    print(remote('python3 -c '+shlex.quote(rebase)), flush=True)
    status('VERIFYING_REMOTE_CORPUS')
    print(remote('cd '+REMOTE+' && bash scripts/hpc3_container.sh .venv/bin/python tools/preflight_4gpu.py --output runs/setup/preflight-hpc3.json'), flush=True)
    # One submission attempt, recorded remotely before returning its id.
    submit = '''import json, subprocess, time
from pathlib import Path
record=Path('runs/setup/submitted-pilot-20261030.json')
if record.exists():
    print(record.read_text())
else:
    job=subprocess.check_output(['sbatch','--parsable','scripts/pilot_4gpu.sbatch']).decode().strip()
    assert job.split(';')[0].isdigit(), job
    data=dict(job_id=job.split(';')[0],submitted_unix=time.time(),stop_after=1000,gpus=4)
    record.write_text(json.dumps(data,indent=2)+'\\n')
    print(json.dumps(data))
'''
    status('SUBMITTING_GPU_PILOT')
    output = remote('cd '+REMOTE+' && python3 -c '+shlex.quote(submit))
    submitted = json.loads(output)
    job = submitted['job_id']
    status('GPU_JOB_SUBMITTED', **submitted)
    # Keep status honest while queued, then collect compact evidence when done.
    while True:
        time.sleep(60)
        queue = remote('squeue -h -j '+job+' -o "%T|%R"').strip()
        if queue:
            status('GPU_JOB_'+queue.split('|')[0], job_id=job, scheduler=queue)
            continue
        accounting = remote('sacct -n -P -j '+job+' --format=JobID,State,ExitCode').strip()
        status('GPU_JOB_ENDED', job_id=job, accounting=accounting)
        print(remote('cd '+REMOTE+' && tar --exclude="*.pt" -czf runs/setup/pilot-results-20261030.tar.gz '+RUN+' logs/pilot-'+job+'.out logs/pilot-'+job+'.err runs/setup/preflight-hpc3.json runs/setup/submitted-pilot-20261030.json'), flush=True)
        call(['scp', '-o', 'StrictHostKeyChecking=yes', '-o', 'GSSAPIAuthentication=no',
              '-o', 'NumberOfPasswordPrompts=1', 'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+REMOTE+'/runs/setup/pilot-results-20261030.tar.gz', ROOT/'runs/setup/'])
        expected = remote('sha256sum '+REMOTE+'/runs/setup/pilot-results-20261030.tar.gz').split()[0]
        assert digest(ROOT/'runs/setup/pilot-results-20261030.tar.gz') == expected
        status('RESULTS_COLLECTED', job_id=job, accounting=accounting, archive='runs/setup/pilot-results-20261030.tar.gz', archive_sha256=expected)
        break


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        previous = json.loads(STATE.read_text()) if STATE.exists() else {}
        status('FAILED', error=str(exc), job_id=previous.get('job_id'), previous_stage=previous.get('stage'))
        raise
