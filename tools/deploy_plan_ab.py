"""Deploy and submit the isolated four-GPU A/B campaign once."""
import fcntl
import hashlib
import json
import shlex
import tarfile
from deploy_fusion_2gpu import ROOT, BASE, call, remote

TAG='plan-ab-full-4gpu-20261103-v2'
RELEASE=BASE+'/experiments/'+TAG


def main():
    setup=ROOT/'runs/setup'
    with (setup/(TAG+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=list((ROOT/'runtime').rglob('*.py'))+list((ROOT/'evaluation').glob('*.py'))+list((ROOT/'evaluation/plan_ab_data_locks').glob('*.json'))
        files += [ROOT/n for n in ('configs/yoco_160m_plan_ab_A_4gpu.json','configs/yoco_160m_plan_ab_B_4gpu.json',
            'tools/preflight_plan_ab.py','tools/summarize_plan_ab.py','tests/test_plan_ab_kv.py',
            'scripts/plan_ab_4gpu.sbatch','scripts/plan_ab_4gpu_inside.sh','tools/preflight_plan_ab_eval.py')]
        manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
        m=setup/(TAG+'-manifest.json'); m.write_text(json.dumps(manifest,indent=2)+'\n')
        archive=setup/(TAG+'.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for p in files: tar.add(p,arcname=str(p.relative_to(ROOT)))
            tar.add(m,arcname='release_manifest.json')
        sha=hashlib.sha256(archive.read_bytes()).hexdigest()
        remote('test ! -e '+shlex.quote(RELEASE))
        call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1','-o','ConnectTimeout=30',archive,
              'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+BASE+'/runs/setup/'+archive.name])
        remote('cd '+shlex.quote(BASE)+' && echo '+shlex.quote(sha+'  runs/setup/'+archive.name)+' | sha256sum -c - && mkdir '+shlex.quote(RELEASE)+
               ' && tar xzf '+shlex.quote(BASE+'/runs/setup/'+archive.name)+' -C '+shlex.quote(RELEASE)+
               ' && ln -s '+shlex.quote(BASE+'/data')+' '+shlex.quote(RELEASE+'/data')+' && python3 '+shlex.quote(RELEASE+'/tools/preflight_plan_ab.py'))
        # Check evaluation dependencies and both model contracts inside the same image as training, before requesting GPUs.
        image='/data/share/jzhou945/cchen099/vertikv-work/images/pytorch-25.06-py3.sif'
        inside='cd '+shlex.quote(RELEASE)+' && '+BASE+'/.venv/bin/python tests/test_plan_ab_kv.py && '+BASE+'/.venv/bin/python tools/preflight_plan_ab_eval.py'
        remote('/bin/singularity exec --cleanenv --no-home --bind '+shlex.quote(BASE+':'+BASE)+' --pwd='+shlex.quote(RELEASE)+
               ' --env '+shlex.quote('PYTHONPATH='+RELEASE)+' '+shlex.quote(image)+' bash -c '+shlex.quote(inside))
        code='''import fcntl,json,subprocess,time
from pathlib import Path
root=Path(RELEASE)
with (root/'submission.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    record=root/'submitted.json'
    if record.exists():print(record.read_text())
    else:
        assert json.loads((root/'preflight.json').read_text())['passed']
        assert not (root/'submission-intent.json').exists(),'Reconcile submission intent before retry'
        for arm in ['A','B']:
            assert not (Path(BASE)/('runs/yoco-160m-plan-ab-4gpu-20261103-v2-'+arm)).exists()
        (root/'submission-intent.json').write_text(json.dumps(dict(started_unix=time.time(),job_name='yoco-plan-ab-4gpu'))+'\\n')
        output=subprocess.check_output(['sbatch','--parsable',str(root/'scripts/plan_ab_4gpu.sbatch')]).decode().strip()
        job=output.split(';')[0]; assert job.isdigit(),output
        result=dict(job_id=job,submitted_unix=time.time(),release=str(root),gpus=4,steps_per_arm=40000,
                    arms={'A':'learned_kv','B':'conditional_kv'},order=['A','B','eight_task_evaluation'],
                    baseline_training=False,replaces_job_id='793493',initial_weights=[1/6]*6,
                    runs={arm:BASE+'/runs/yoco-160m-plan-ab-4gpu-20261103-v2-'+arm for arm in ['A','B']})
        record.write_text(json.dumps(result,indent=2)+'\\n'); print(json.dumps(result),flush=True)
'''.replace('RELEASE',repr(RELEASE)).replace('BASE',repr(BASE))
        remote('python3 -c '+shlex.quote(code))
        for src,dst in [(RELEASE+'/submitted.json',setup/('submitted-'+TAG+'.json')),
                        (BASE+'/runs/setup/plan-ab-4gpu-v1-cancelled-baseline.json',setup/'plan-ab-4gpu-v1-cancelled-baseline.json')]:
            call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',
                  'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+src,dst])
        (setup/'plan-ab-active-job.json').write_text((setup/('submitted-'+TAG+'.json')).read_text())
        print('A/B job submitted and recorded; do not resubmit.',flush=True)


if __name__=='__main__':main()
