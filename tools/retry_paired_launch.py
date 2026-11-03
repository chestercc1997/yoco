"""Repair the launcher only after confirming both original jobs failed pre-training."""
import hashlib,json,shlex
from pathlib import Path
from deploy_fusion_2gpu import ROOT,BASE,call,remote


def main():
    records=[]
    for tag in ('baseline-2gpu-20261101-v2','fusion-token-2gpu-20261101-v2'):
        original=json.loads((ROOT/'runs/setup'/('submitted-'+tag+'.json')).read_text())
        release=original['release'];run=original['run_dir'];config=original['config']
        script=(ROOT/'runs/setup'/(tag+'-package/job.sbatch')).read_text()
        original_line='  "$image" bash "$release/scripts/paired_2gpu_inside.sh" "$release" '+config+' '+run
        command='exec bash '+shlex.quote(release+'/scripts/paired_2gpu_inside.sh')+' '+shlex.quote(release)+' '+shlex.quote(config)+' '+shlex.quote(run)
        replacement='  "$image" bash -c '+shlex.quote(command)
        assert original_line in script
        script=script.replace(original_line,replacement)
        local=ROOT/'runs/setup'/(tag+'-retry1.sbatch');local.write_text(script)
        call(['bash','-n',local])
        remote_script=BASE+'/runs/setup/'+local.name
        call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',local,
              'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+remote_script])
        sha=hashlib.sha256(local.read_bytes()).hexdigest()
        # Same SIF and argument style as the retry; no GPU action on login.
        audit='cd '+shlex.quote(release)+' && /bin/singularity exec --cleanenv --no-home'
        audit+=' --bind '+shlex.quote(BASE+':'+BASE)+' --pwd='+shlex.quote(release)
        audit+=' --env '+shlex.quote('PYTHONPATH='+release)
        audit+=' /data/share/jzhou945/cchen099/vertikv-work/images/pytorch-25.06-py3.sif bash -c '
        audit+=shlex.quote('exec '+BASE+'/.venv/bin/python tools/preflight_fusion.py --check-only --config '+shlex.quote(config))
        remote(audit)
        remote_record='runs/setup/submitted-'+tag+'-retry1.json'
        request=dict(original=original,script=remote_script,sha256=sha,record=remote_record)
        code='''import fcntl,hashlib,json,subprocess,time
from pathlib import Path
request=REQUEST
old=request['original'];base=Path(BASE)
record=base/request['record']
with (base/'runs/setup/paired-v2-retry.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if record.exists():print(record.read_text())
    else:
        accounting=subprocess.check_output(['sacct','-n','-P','-j',old['job_id'],'--format=JobID,State,ExitCode']).decode()
        row=next(line.split('|') for line in accounting.splitlines() if line.split('|')[0]==old['job_id'])
        assert row[1]=='FAILED' and row[2]=='1:0',row
        run=Path(old['run_dir'])
        assert not (run/'rank0.jsonl').exists(), 'Training already started; review instead of retry'
        assert not (run/'smoke/rank0.jsonl').exists()
        script=Path(request['script'])
        assert hashlib.sha256(script.read_bytes()).hexdigest()==request['sha256']
        intent=record.with_suffix('.intent.json')
        if intent.exists():raise RuntimeError('Submission intent exists; reconcile before retry')
        intent.write_text(json.dumps(dict(time=time.time(),previous_job_id=old['job_id'])))
        job=subprocess.check_output(['sbatch','--parsable',str(script)]).decode().strip().split(';')[0]
        assert job.isdigit(),job
        result=dict(old);result.update(job_id=job,previous_job_id=old['job_id'],submitted_unix=time.time(),
            submission_record=request['record'],launcher_sha256=request['sha256'],
            retry_reason='SIF wrapper misclassified positional code directory; launcher-only repair')
        record.write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result),flush=True)
'''.replace('REQUEST',repr(request)).replace('BASE',repr(BASE))
        remote('python3 -c '+shlex.quote(code))
        dest=ROOT/remote_record
        call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',
              'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+BASE+'/'+remote_record,dest])
        records.append(json.loads(dest.read_text()))
    (ROOT/'runs/setup/paired-v2-active-jobs.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
