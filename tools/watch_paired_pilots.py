"""Local CPU watcher: collect completed jobs, run read-only diagnostics, compare.

Does not submit jobs or start full training. Logs survive the interactive turn.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from deploy_fusion_2gpu import ROOT, BASE, ENV, SKILL

STATE=ROOT/'runs/setup/paired-v2-watcher.json'
RECORDS=[ROOT/'runs/setup'/('submitted-'+tag+'.json') for tag in ('baseline-2gpu-20261101-v2','fusion-token-2gpu-20261101-v2')]
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','UNKNOWN'}


def remote(command,timeout=180):
    return subprocess.check_output([sys.executable,str(SKILL/'scripts/remote.py'),command],cwd=ROOT,env=ENV,
                                   stdin=subprocess.DEVNULL,start_new_session=True,timeout=timeout).decode()


def scp(source,target):
    subprocess.run(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no',
                    '-o','NumberOfPasswordPrompts=1','-o','ConnectTimeout=30',str(source),str(target)],
                   env=ENV,stdin=subprocess.DEVNULL,start_new_session=True,check=True,timeout=300)


def status(stage,**extra):
    value=dict(stage=stage,time_unix=time.time(),pid=os.getpid(),**extra)
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATE)
    print(json.dumps(value),flush=True)


def main():
    import fcntl
    import argparse
    global STATE
    parser=argparse.ArgumentParser()
    parser.add_argument('--records',type=Path,default=ROOT/'runs/setup/paired-v2-active-jobs.json')
    parser.add_argument('--state',type=Path,default=STATE)
    parser.add_argument('--report',type=Path,default=ROOT/'docs/evidence/paired-v2-convergence-review.json')
    parser.add_argument('--compare-script',type=Path,default=ROOT/'tools/compare_paired_pilots.py')
    parser.add_argument('--timeout-hours',type=float,default=72)
    args=parser.parse_args();STATE=args.state
    lock=STATE.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=args.records
    records=json.loads(active.read_text()) if active.exists() else [json.loads(p.read_text()) for p in RECORDS]
    jobs=[r['job_id'] for r in records]
    status('MONITORING',job_ids=jobs)
    started=time.time()
    while time.time()-started<args.timeout_hours*3600:
        query='''import json,subprocess
jobs=JOBS
result={}
for job in jobs:
    text=subprocess.check_output(['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode'],universal_newlines=True)
    result[job]=next((dict(state=p[1].split()[0],exit_code=p[2]) for p in (line.split('|') for line in text.splitlines()) if p[0]==job),dict(state='UNKNOWN'))
print(json.dumps(result))'''.replace('JOBS',repr(jobs))
        try:
            states=json.loads(remote('python3 -c '+shlex.quote(query)))
            status('MONITORING',jobs=states)
        except (subprocess.SubprocessError,ValueError) as exc:
            status('STATUS_RETRY',error=str(exc));time.sleep(60);continue
        if all(s['state'] not in ACTIVE for s in states.values()):
            break
        time.sleep(60)
    else:
        status('MONITOR_TIMEOUT',job_ids=jobs);return
    if any(s['state']!='COMPLETED' or s.get('exit_code')!='0:0' for s in states.values()):
        status('JOB_FAILED_REVIEW_REQUIRED',jobs=states);return
    status('READ_ONLY_GRADIENT_DIAGNOSTICS',jobs=states)
    script=ROOT/'tools/probe_gradient_flow.py'
    remote_script=BASE+'/runs/setup/probe-gradient-flow-20261101-v2.py'
    scp(script,'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+remote_script)
    expected=hashlib.sha256(script.read_bytes()).hexdigest()
    assert remote('sha256sum '+shlex.quote(remote_script)).split()[0]==expected
    archives={}
    for r in records:
        run=Path(r['run_dir']);release=r['release'];tag=Path(release).name
        command='cd '+shlex.quote(BASE)+' && bash scripts/hpc3_container.sh .venv/bin/python '+shlex.quote(remote_script)
        command+=' --release='+shlex.quote(release)+' --run-dir='+shlex.quote(str(run))
        print(remote(command,timeout=900),flush=True)
        archive_name=r.get('collection_tag',tag)+'-results.tar.gz'
        log_prefix=r.get('log_prefix',tag)
        parts=['runs/'+run.name,'experiments/'+tag+'/submitted.json','experiments/'+tag+'/release_manifest.json',
               'logs/'+log_prefix+'-'+r['job_id']+'.out','logs/'+log_prefix+'-'+r['job_id']+'.err']
        if r.get('submission_record'):
            parts.append(r['submission_record'])
        command='cd '+shlex.quote(BASE)+' && tar --exclude="*.pt" --exclude="*.tmp" -czf '+shlex.quote('runs/setup/'+archive_name)+' '+' '.join(shlex.quote(p) for p in parts)
        command+=' && sha256sum '+shlex.quote('runs/setup/'+archive_name)
        expected=remote(command).split()[0]
        local=ROOT/'runs/setup'/archive_name
        scp('jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+BASE+'/runs/setup/'+archive_name,local)
        assert hashlib.sha256(local.read_bytes()).hexdigest()==expected
        archives[r['arm']]=str(local)
        status('COLLECTING',archives=archives,last_archive_sha256=expected)
    report=args.report
    subprocess.run([sys.executable,str(args.compare_script),'--baseline',archives['baseline'],
                    '--dynamic',archives['dynamic'],'--output',str(report)],cwd=ROOT,check=True)
    verdict=json.loads(report.read_text())['verdict']
    status('REVIEW_COMPLETE',job_ids=jobs,report=str(report),verdict=verdict)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('WATCHER_ERROR',error=str(exc));raise
