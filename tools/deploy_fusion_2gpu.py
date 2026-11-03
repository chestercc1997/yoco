"""Deploy an immutable experiment directory; audit before one recorded submission."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = '/data/share/jzhou945/cchen099/yoco'
TAG = 'fusion-token-2gpu-20261101-v1'
RELEASE = BASE + '/experiments/' + TAG
SKILL = Path('/hpc2hdd/home/jzhou945/cchen099/.codex-home/skills/hpc3')
ENV = os.environ.copy()
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    ENV.pop(key, None)
ENV.update(NO_PROXY='*', no_proxy='*', SSH_ASKPASS=str(SKILL/'scripts/askpass.py'),
           SSH_ASKPASS_REQUIRE='force', DISPLAY='hpc3-auth')


def call(args):
    subprocess.run([str(a) for a in args], cwd=ROOT, env=ENV, check=True,
                   stdin=subprocess.DEVNULL, start_new_session=True)


def remote(command):
    call([sys.executable, SKILL/'scripts/remote.py', command])


def main():
    lock = (ROOT/'runs/setup/fusion-2gpu-deploy.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = [p for p in (ROOT/'runtime').rglob('*.py')]
    files += list((ROOT/'configs').glob('yoco_160m_fusion_*.json'))
    files += [ROOT/p for p in ('tests/test_memory_fusion.py', 'tools/preflight_fusion.py',
                              'scripts/fusion_token_2gpu.sbatch', 'scripts/fusion_token_inside.sh')]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    local_manifest = ROOT/'runs/setup/fusion-release-manifest.json'
    local_manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    archive = ROOT/'runs/setup'/f'{TAG}.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)))
        tar.add(local_manifest, arcname='release_manifest.json')
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    # Never overwrite an existing release, even after an interrupted deployment.
    remote('test ! -e '+shlex.quote(RELEASE)+' && mkdir -p '+shlex.quote(BASE+'/experiments'))
    call(['scp', '-o', 'StrictHostKeyChecking=yes', '-o', 'GSSAPIAuthentication=no',
          '-o', 'NumberOfPasswordPrompts=1', '-o', 'ConnectTimeout=30', archive,
          'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+BASE+'/runs/setup/'+archive.name])
    remote('cd '+shlex.quote(BASE)+' && echo '+shlex.quote(sha+'  runs/setup/'+archive.name)+
           ' | sha256sum -c - && mkdir '+shlex.quote(RELEASE)+
           ' && tar xzf '+shlex.quote(BASE+'/runs/setup/'+archive.name)+' -C '+shlex.quote(RELEASE)+
           ' && ln -s '+shlex.quote(BASE+'/data')+' '+shlex.quote(RELEASE+'/data')+
           ' && python3 '+shlex.quote(RELEASE+'/tools/preflight_fusion.py'))
    submit = r'''
import fcntl,json,subprocess,time
from pathlib import Path
root=Path(RELEASE)
with (root/'submission.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    record=root/'submitted.json'
    if record.exists():
        print(record.read_text())
    else:
        assert json.loads((root/'preflight.json').read_text())['passed']
        # Refuse automatic retry if submission acknowledgement was lost.
        intent=root/'submission-intent.json'
        if intent.exists():raise RuntimeError('Existing submission intent; reconcile scheduler before retry')
        intent.write_text(json.dumps(dict(started_unix=time.time(),job_name='yoco160-fusion2'))+'\n')
        output=subprocess.check_output(['sbatch','--parsable',str(root/'scripts/fusion_token_2gpu.sbatch')]).decode().strip()
        job=output.split(';')[0]
        assert job.isdigit(),output
        result=dict(job_id=job,submitted_unix=time.time(),release=str(root),gpus=2,stop_after=1000,
                    run_dir=BASE+'/runs/yoco-160m-fusion-token-2h100-pilot-20261101-v1')
        record.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
'''.replace('RELEASE', repr(RELEASE)).replace('BASE', repr(BASE))
    remote('python3 -c '+shlex.quote(submit))
    call(['scp', '-o', 'StrictHostKeyChecking=yes', '-o', 'GSSAPIAuthentication=no',
          '-o', 'NumberOfPasswordPrompts=1',
          'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+RELEASE+'/submitted.json',
          ROOT/'runs/setup/submitted-fusion-token-2gpu-20261101-v1.json'])
    print('Submission recorded locally. Inspect this job; do not submit again.', flush=True)


if __name__ == '__main__':
    main()
