"""Submit one independent two-GPU arm of the controlled 160M pilot."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import tarfile
from deploy_fusion_2gpu import ROOT, BASE, call, remote

ARMS = {
    'baseline': ('baseline-2gpu-20261101-v2', 'configs/yoco_160m_baseline_2gpu.json', 'yoco160-base2'),
    'dynamic': ('fusion-token-2gpu-20261101-v2', 'configs/yoco_160m_fusion_token_v2_2gpu.json', 'yoco160-dynv2'),
}


def main():
    p=argparse.ArgumentParser();p.add_argument('arm',choices=ARMS);a=p.parse_args()
    tag,config,jobname=ARMS[a.arm]
    release=BASE+'/experiments/'+tag;run=BASE+'/runs/yoco-160m-'+tag
    lock=(ROOT/'runs/setup'/f'{tag}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stage=ROOT/'runs/setup'/f'{tag}-package';stage.mkdir(exist_ok=True)
    script=f'''#!/usr/bin/env bash
#SBATCH --job-name={jobname}
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir={release}
#SBATCH --output={BASE}/logs/{tag}-%j.out
#SBATCH --error={BASE}/logs/{tag}-%j.err
set -euo pipefail
base={BASE}
release={release}
image=/data/share/jzhou945/cchen099/vertikv-work/images/pytorch-25.06-py3.sif
exec srun --kill-on-bad-exit=1 /bin/singularity exec --cleanenv --no-home --nv \\
  --bind "$base:$base" --pwd="$release" \\
  --env "PYTHONPATH=$release" --env "TIKTOKEN_CACHE_DIR=$base/data/cache/tiktoken" \\
  --env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 --env HF_HUB_OFFLINE=1 \\
  --env TOKENIZERS_PARALLELISM=false --env "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}" \\
  "$image" bash -c {shlex.quote("exec bash " + release + "/scripts/paired_2gpu_inside.sh " + release + " " + config + " " + run)}
'''
    sbatch=stage/'job.sbatch';sbatch.write_text(script);call(['bash','-n',sbatch])
    files={str(path.relative_to(ROOT)):path for path in (ROOT/'runtime').rglob('*.py')}
    files.update({str(path.relative_to(ROOT)):path for path in (ROOT/'configs').glob('yoco_160m_*.json')})
    for name in ('tools/preflight_fusion.py','tools/probe_fusion_routing.py','tests/test_memory_fusion.py','scripts/paired_2gpu_inside.sh'):
        files[name]=ROOT/name
    files['job.sbatch']=sbatch
    manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
    m=stage/'release_manifest.json';m.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'runs/setup'/f'{tag}.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in files.items():tar.add(path,arcname=name)
        tar.add(m,arcname='release_manifest.json')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    remote('test ! -e '+shlex.quote(release))
    call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',
          archive,'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+BASE+'/runs/setup/'+archive.name])
    remote('cd '+shlex.quote(BASE)+' && echo '+shlex.quote(sha+'  runs/setup/'+archive.name)+
           ' | sha256sum -c - && mkdir '+shlex.quote(release)+
           ' && tar xzf '+shlex.quote(BASE+'/runs/setup/'+archive.name)+' -C '+shlex.quote(release)+
           ' && ln -s '+shlex.quote(BASE+'/data')+' '+shlex.quote(release+'/data')+
           ' && python3 '+shlex.quote(release+'/tools/preflight_fusion.py')+' --config '+shlex.quote(config))
    submit='''
import fcntl,json,subprocess,time
from pathlib import Path
root=Path(RELEASE)
with (root/'submission.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    record=root/'submitted.json'
    if record.exists():print(record.read_text())
    else:
        assert json.loads((root/'preflight.json').read_text())['passed']
        intent=root/'submission-intent.json'
        if intent.exists():raise RuntimeError('Existing intent: reconcile scheduler before retry')
        intent.write_text(json.dumps(dict(time=time.time(),job_name=JOBNAME)))
        job=subprocess.check_output(['sbatch','--parsable',str(root/'job.sbatch')]).decode().strip().split(';')[0]
        assert job.isdigit(),job
        result=dict(job_id=job,arm=ARM,submitted_unix=time.time(),release=str(root),run_dir=RUN,config=CONFIG,gpus=2,stop_after=1000)
        record.write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result),flush=True)
'''
    for key,value in [('RELEASE',release),('JOBNAME',jobname),('ARM',a.arm),('RUN',run),('CONFIG',config)]:
        submit=submit.replace(key,repr(value))
    remote('python3 -c '+shlex.quote(submit))
    call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',
          'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+release+'/submitted.json',ROOT/'runs/setup'/f'submitted-{tag}.json'])


if __name__=='__main__':main()
