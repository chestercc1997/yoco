"""Continue both approved two-GPU arms from step 1000 without changing the recipe."""
import fcntl,hashlib,json,shlex
from pathlib import Path
from deploy_fusion_2gpu import ROOT,BASE,call,remote


def upload(path,destination):
    call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',path,
          'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+destination])


def main():
    lock=(ROOT/'runs/setup/full-v2-deploy.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=json.loads((ROOT/'runs/setup/paired-v2-active-jobs.json').read_text());submitted=[]
    inside=ROOT/'scripts/full_2gpu_inside.sh';inside_sha=hashlib.sha256(inside.read_bytes()).hexdigest()
    remote_inside=BASE+'/runs/setup/full_2gpu_inside-'+inside_sha[:12]+'.sh'
    upload(inside,remote_inside)
    remote('echo '+shlex.quote(inside_sha+'  '+remote_inside)+' | sha256sum -c -')
    for old in records:
        arm=old['arm'];tag=('baseline' if arm=='baseline' else 'fusion-token')+'-2gpu-full-20261101-v2'
        release=old['release'];config=old['config'];run=BASE+'/runs/yoco-160m-'+tag
        config_audit='exec '+BASE+'/.venv/bin/python tools/preflight_fusion.py --check-only --config '+shlex.quote(config)
        remote('cd '+shlex.quote(release)+' && /bin/singularity exec --cleanenv --no-home --bind '+shlex.quote(BASE+':'+BASE)+
               ' --pwd='+shlex.quote(release)+' --env '+shlex.quote('PYTHONPATH='+release)+
               ' /data/share/jzhou945/cchen099/vertikv-work/images/pytorch-25.06-py3.sif bash -c '+shlex.quote(config_audit))
        command='exec bash '+shlex.quote(remote_inside)+' '+shlex.quote(release)+' '+shlex.quote(config)+' '+shlex.quote(run)
        script=f'''#!/usr/bin/env bash
#SBATCH --job-name=yoco160-{'basefull' if arm=='baseline' else 'dynfull'}
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --chdir={release}
#SBATCH --output={BASE}/logs/{tag}-%j.out
#SBATCH --error={BASE}/logs/{tag}-%j.err
set -euo pipefail
base={BASE}
release={release}
image=/data/share/jzhou945/cchen099/vertikv-work/images/pytorch-25.06-py3.sif
exec srun --kill-on-bad-exit=1 /bin/singularity exec --cleanenv --no-home --nv \\
 --bind "$base:$base" --pwd="$release" --env "PYTHONPATH=$release" \\
 --env "TIKTOKEN_CACHE_DIR=$base/data/cache/tiktoken" --env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \\
 --env HF_HUB_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false --env "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}" \\
 "$image" bash -c {shlex.quote(command)}
'''
        path=ROOT/'runs/setup'/(tag+'.sbatch');path.write_text(script);call(['bash','-n',path])
        remote_script=BASE+'/runs/setup/'+path.name;upload(path,remote_script)
        request=dict(old=old,tag=tag,run=run,script=remote_script,script_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
                     inside=remote_inside,inside_sha=inside_sha,record='runs/setup/submitted-'+tag+'.json')
        code='''import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
r=REQUEST;base=Path(BASE);old=r['old'];source=Path(old['run_dir']);run=Path(r['run']);record=base/r['record']
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
with (base/'runs/setup/full-v2-submit.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if record.exists():print(record.read_text())
    else:
        history=subprocess.check_output(['sacct','-n','-P','-j',old['job_id'],'--format=JobID,State,ExitCode']).decode()
        row=next(x.split('|') for x in history.splitlines() if x.split('|')[0]==old['job_id'])
        assert row[1:3]==['COMPLETED','0:0'],row
        assert not run.exists(),'Run already exists: reconcile before retry'
        assert digest(Path(r['script']))==r['script_sha']
        assert digest(Path(r['inside']))==r['inside_sha']
        meta=json.loads((source/'latest.json').read_text());assert meta['step']==1000
        checkpoint=source/meta['file'];assert digest(checkpoint)==meta['sha256']
        run.mkdir()
        os.link(checkpoint,run/checkpoint.name)
        shutil.copy2(checkpoint.with_suffix('.json'),(run/checkpoint.name).with_suffix('.json'))
        shutil.copy2(source/'latest.json',run/'latest.json')
        origin=dict(source_run=str(source),source_job_id=old['job_id'],step=1000,fingerprint=meta['fingerprint'],
                    checkpoint_sha256=meta['sha256'],checkpoint_file=meta['file'],target_step=40000)
        (run/'resume_origin.json').write_text(json.dumps(origin,indent=2)+'\\n')
        intent=record.with_suffix('.intent.json')
        if intent.exists():raise RuntimeError('Submission intent exists; reconcile scheduler')
        intent.write_text(json.dumps(dict(time=time.time(),run=str(run))))
        job=subprocess.check_output(['sbatch','--parsable',r['script']]).decode().strip().split(';')[0]
        assert job.isdigit(),job
        result=dict(job_id=job,arm=old['arm'],release=old['release'],config=old['config'],run_dir=str(run),
                    gpus=2,start_step=1000,stop_after=40000,submitted_unix=time.time(),
                    collection_tag=r['tag'],log_prefix=r['tag'],submission_record=r['record'],
                    source_pilot_run=str(source),launcher_sha256=r['script_sha'],inside_script_sha256=r['inside_sha'])
        record.write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result),flush=True)
'''.replace('REQUEST',repr(request)).replace('BASE',repr(BASE))
        remote('python3 -c '+shlex.quote(code))
        dest=ROOT/request['record']
        call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',
              'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+BASE+'/'+request['record'],dest])
        submitted.append(json.loads(dest.read_text()))
    (ROOT/'runs/setup/full-v2-active-jobs.json').write_text(json.dumps(submitted,indent=2)+'\n')


if __name__=='__main__':main()
