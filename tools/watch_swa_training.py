"""Attach to the SWA cohort; detect numerical failures and restore healthy state.

Keeps failed attempts intact. Repeated numerical failures produce a bounded,
read-only diagnosis for the active Goal, rather than an unlimited retry loop.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ARMS=('baseline','A','B')


def read(path):return json.loads(Path(path).read_text())


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def verify_contract(run):
    """No recovery may change *any* frozen config, model code, or data manifest."""
    release=run/'release';manifest=read(run/'release-manifest.json')
    for name,expected in manifest['files'].items():
        if name.endswith('.py') or name.startswith('configs/') or name=='preflight.json':
            assert sha(release/name)==expected,f'Frozen release changed: {name}'
    assert sha(ROOT/'tools/run_swa_ab.py')==manifest['files']['tools/run_swa_ab.py'],'Controller changed after freeze'
    for arm in ARMS:
        frozen=read(release/f'configs/{arm}.json')
        assert read(ROOT/f'configs/yoco_160m_swa_{arm}_8gpu.json')==frozen,f'{arm}: training setting changed'
        assert sha(frozen['data']['manifest'])==manifest['data_manifest_sha256']
    contract={'config_sha256':{a:sha(release/f'configs/{a}.json') for a in ARMS},
              'release_manifest_sha256':sha(run/'release-manifest.json'),
              'data_manifest_sha256':manifest['data_manifest_sha256'],
              'no_training_setting_changes':True}
    path=run/'guard-contract.json'
    if path.exists():assert read(path)==contract,'Recovery contract changed'
    else:write(path,contract)
    return contract


def events(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue  # A concurrent writer may have a partial final line.
    return result


def unhealthy(rows,regression=.15,relative=.05,consecutive=3):
    """Ignore individual noisy batches/AMP backoffs; require sustained validation regression."""
    best=float('inf');bad=[]
    # Deduplicate completed validations from a resumed worker.
    vals={r['step']:r for r in rows if r['event']=='validation'}
    for step,row in sorted(vals.items()):
        value=row['loss']
        if not math.isfinite(value):return {'kind':'nonfinite_validation','step':step,'rollback_before':step}
        if value>best+max(regression,relative*best):bad.append(step)
        else:bad=[]
        if len(bad)>=consecutive:return {'kind':'sustained_validation_regression','step':step,
            'rollback_before':bad[0],'best_loss':best,'observed_loss':value,'consecutive':len(bad)}
        best=min(best,value)
    nonfinite=[r for r in rows if r['event']=='nonfinite_forward']
    if nonfinite:
        row=nonfinite[-1];step=row.get('attempted_update',row['step']+1)
        return {'kind':'nonfinite_forward','step':step,'rollback_before':step}
    return None


def checkpoint_healthy(path,expected_fingerprint=None):
    import torch
    path=Path(path);meta=read(path.with_suffix('.json'))
    assert meta['file']==path.name and sha(path)==meta['sha256'],'Checkpoint checksum mismatch'
    resolved=read(path.parent/'resolved_config.json')
    identity=hashlib.sha256(json.dumps(resolved['identity'],sort_keys=True).encode()).hexdigest()
    assert meta['fingerprint']==identity
    if expected_fingerprint:assert identity==expected_fingerprint,'Checkpoint changes model/training identity'
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==meta['step'] and saved['world_size']==8 and saved['fingerprint']==identity
    counts={'tensors':0}
    def finite(value):
        if isinstance(value,torch.Tensor):
            counts['tensors']+=1
            if value.is_floating_point():assert bool(torch.isfinite(value).all()),'Nonfinite checkpoint tensor'
        elif isinstance(value,dict):
            for item in value.values():finite(item)
        elif isinstance(value,(tuple,list)):
            for item in value:finite(item)
    finite(saved['model']);finite(saved['optimizer'])
    assert len(saved['rngs'])==8 and all(set(r)=={'cpu','cuda','python','numpy'} for r in saved['rngs'])
    assert math.isfinite(saved['scaler']['scale']) and saved['scaler']['scale']>0
    assert saved['scaler']['_growth_tracker']>=0
    return {'path':str(path),'step':meta['step'],'sha256':meta['sha256'],'fingerprint':identity,
            'finite_tensors_checked':counts['tensors'],'full_optimizer_scaler_eight_rank_rng':True}


def select_checkpoint(target,issue):
    candidates=sorted(target.glob('checkpoint_*.json'),reverse=True)
    fallback=target/'checks/resumed/checkpoint_00000004.json'
    if fallback.exists():candidates.append(fallback)
    expected=None
    if (target/'resolved_config.json').exists():
        expected=hashlib.sha256(json.dumps(read(target/'resolved_config.json')['identity'],sort_keys=True).encode()).hexdigest()
    rejected=[]
    for meta_path in candidates:
        try:
            meta=read(meta_path)
            if meta['step']>=issue['rollback_before']:continue
            proof=checkpoint_healthy(meta_path.with_suffix('.pt'),expected)
            return proof,rejected
        except Exception as error:rejected.append({'path':str(meta_path),'reason':str(error)})
    raise RuntimeError(f'No verified healthy checkpoint: {rejected}')


def recover(run,arm,issue,attempt):
    """Journaled rollback: archive failed logs/artifacts, retain their verified prefix."""
    target=run/arm;archive=target/'recovery-attempts'/f'{attempt:03d}'
    journal=archive/'recovery.json'
    if journal.exists():record=read(journal)
    else:
        proof,rejected=select_checkpoint(target,issue)
        record={'state':'prepared','arm':arm,'issue':issue,'source':proof,'rejected_checkpoints':rejected,
                'attempt':attempt,'created_at':now(),'policy':'identical frozen runtime/config; replay every batch after source'}
        write(journal,record)
    step=record['source']['step'];archive.mkdir(parents=True,exist_ok=True)
    # Idempotent after an interrupted rollback: archived originals always win.
    for rank in range(8):
        path=target/f'rank{rank}.jsonl';old=archive/path.name
        if not old.exists() and path.exists():path.replace(old)
        if old.exists():
            keep=[r for r in events(old) if (r.get('attempted_update',r.get('step',0)+1)<=step
                if r['event'] in ('overflow','nonfinite_forward') else r.get('step',0)<=step)]
            tmp=path.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in keep));tmp.replace(path)
    for path in list(target.glob('checkpoint_*')):
        try:checkpoint_step=int(path.name.split('_')[1].split('.')[0])
        except (ValueError,IndexError):continue
        if checkpoint_step>step:path.replace(archive/path.name)
    for name in ('latest.json','stage_complete.json','heartbeat.json','summary.json','training-review.json','PIPELINE_SUCCESS'):
        path=target/name
        if path.exists() and not (archive/name).exists():path.replace(archive/name)
    if Path(record['source']['path']).parent==target:write(target/'latest.json',read(Path(record['source']['path']).with_suffix('.json')))
    for name in ('evaluation',):
        path=target/name
        if path.exists() and not (archive/name).exists():path.replace(archive/name)
    for path in target.glob('nonfinite_forward_*'):
        if not (archive/path.name).exists():path.replace(archive/path.name)
    record['state']='rolled_back';record['completed_at']=now();write(journal,record)
    return record


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return {'pid':pid,'ppid':int(fields[1]),'start_ticks':fields[19]}
    except (FileNotFoundError,ProcessLookupError):return None


def same_process(identity):
    actual=process_identity(identity['pid'])
    return actual is not None and actual['start_ticks']==identity['start_ticks']


def stop_tree(identity):
    # PID/start-time checks prevent signals reaching an unrelated reused PID.
    if not same_process(identity):return
    identities={identity['pid']:identity}
    for _ in range(4):
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            child=process_identity(int(path.name))
            if child and child['ppid'] in identities:identities[child['pid']]=child
    for sig,seconds in ((signal.SIGTERM,20),(signal.SIGKILL,5)):
        for pid,item in identities.items():
            if same_process(item):
                try:os.kill(pid,sig)
                except ProcessLookupError:pass
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline and any(same_process(item) for item in identities.values()):time.sleep(.5)
    assert not any(same_process(item) for item in identities.values()),'Owned worker tree failed to stop'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--poll-seconds',type=float,default=10)
    p.add_argument('--stall-seconds',type=float,default=600)
    p.add_argument('--max-recoveries-per-arm',type=int,default=2)
    p.add_argument('--replace-existing',action='store_true',help='Reload only this cohort supervisor; leave its training controller/workers running')
    args=p.parse_args();run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    if args.replace_existing:
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
            try:argv=(directory/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            argv=[a.decode(errors='replace') for a in argv if a]
            if (any(a.endswith('/watch_swa_training.py') for a in argv)
                    and '--run-dir' in argv and argv[argv.index('--run-dir')+1]==str(run)):
                identity=process_identity(int(directory.name))
                if identity and same_process(identity):
                    os.kill(identity['pid'],signal.SIGTERM)
                    deadline=time.monotonic()+10
                    while same_process(identity) and time.monotonic()<deadline:time.sleep(.1)
                    assert not same_process(identity),'Previous supervisor did not exit; training is left intact'
    lock=(run/'guard.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A recovery guard already owns this cohort')
    import torch
    torch.set_num_threads(4)
    state=read(run/'guard-state.json') if (run/'guard-state.json').exists() else {'recoveries':[]}
    verify_contract(run)
    # Journals, rather than a possibly interrupted status write, own recovery history.
    state['recoveries']=[]
    for arm in ARMS:
        for journal in sorted((run/arm/'recovery-attempts').glob('*/recovery.json')):
            record=read(journal)
            if record['state']!='rolled_back':
                pipeline=read(run/'pipeline-status.json')
                assert process_identity(pipeline['pid']) is None,'Incomplete rollback with a live controller needs diagnosis'
                record=recover(run,arm,record['issue'],record['attempt'])
            state['recoveries'].append(record)
    child=None;attached=None;failure_seen=None
    def update(status,**fields):
        state.update(state=status,updated_at=now(),**fields);write(run/'guard-state.json',state)
    def launch():
        verify_contract(run)
        output=(run/'guard-controller.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/run_swa_ab.py'),'--run-dir',str(run)],
            cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        output.close();return process
    try:
        while True:
            if (run/'PIPELINE_SUCCESS').exists():
                update('training_and_evaluation_complete',goal_remaining='export, inspect convergence, publish');return
            pipeline=read(run/'pipeline-status.json') if (run/'pipeline-status.json').exists() else {}
            if attached is None:
                candidate=process_identity(pipeline.get('pid',-1))
                if candidate:
                    command=Path(f"/proc/{candidate['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
                    assert 'run_swa_ab.py' in command and str(run) in command,'Refuse to attach to an unrelated process'
                    attached=candidate;update('monitoring',controller_pid=attached['pid'])
                else:
                    if pipeline:
                        # A restarted guard must inspect/roll back a dead formal
                        # worker before launching anything against its old logs.
                        attached={'pid':pipeline.get('pid',-1),'ppid':-1,'start_ticks':'dead-controller'}
                    else:
                        child=launch();attached=process_identity(child.pid)
                        assert attached is not None
                        update('monitoring',controller_pid=child.pid);time.sleep(args.poll_seconds);continue
            alive=same_process(attached)
            if alive and pipeline.get('pid')!=attached['pid']:
                # Preparing/verifying a frozen release precedes publishing its PID.
                time.sleep(args.poll_seconds);continue
            stage=pipeline.get('state','');arm=next((a for a in ARMS if stage==a+'-training'),None)
            # A controller failure records the failed stage separately from state='failed'.
            if arm is None and stage=='failed':
                arm=next((a for a in ARMS if pipeline.get('stages',{}).get(a+'-training',{}).get('state') in ('failed','running')),None)
            if arm is None:
                # Catch a nonfinite final validation even if the controller moved
                # from training to review/evaluation between monitoring polls.
                arm=next((a for a in ARMS if not (run/a/'PIPELINE_SUCCESS').exists()
                    and a+'-training' in pipeline.get('stages',{})
                    and unhealthy(events(run/a/'rank0.jsonl'))),None)
            if arm:
                rows=events(run/arm/'rank0.jsonl')
                # Any rank can be the first to fail forward; rank0 alone is insufficient.
                all_rows=rows+[r for rank in range(1,8) for r in events(run/arm/f'rank{rank}.jsonl') if r['event']=='nonfinite_forward']
                issue=unhealthy(all_rows)
                heartbeat=run/arm/'heartbeat.json'
                last_activity=heartbeat.stat().st_mtime if heartbeat.exists() else datetime.datetime.fromisoformat(pipeline['updated_at']).timestamp()
                if issue is None and alive and time.time()-last_activity>args.stall_seconds:
                    step=max((r['step'] for r in rows if r['event']=='train'),default=4)
                    issue={'kind':'training_stall','step':step,'rollback_before':step+1}
                if issue is None and (not alive or stage=='failed'):
                    step=max((r['step'] for r in rows if r['event']=='train'),default=4)
                    issue={'kind':'worker_exit','step':step,'rollback_before':step+1}
                if issue:
                    update('failure_detected',arm=arm,issue=issue)
                    # Give the trainer time to finish its diagnostic snapshot before killing peers.
                    if issue['kind']=='nonfinite_forward' and alive:
                        if failure_seen is None:failure_seen=time.monotonic()
                        if time.monotonic()-failure_seen<120:
                            time.sleep(args.poll_seconds);continue
                    stop_tree(attached)
                    if child:child.wait(timeout=10)
                    history=[r for r in state['recoveries'] if r['arm']==arm]
                    if len(history)>=args.max_recoveries_per_arm:
                        snapshots=sorted((run/arm).glob(f"nonfinite_forward_{issue['step']:08d}_rank*.pt"))
                        diagnosis=run/arm/'automatic-diagnosis.json'
                        if snapshots:
                            with (run/arm/'automatic-diagnosis.log').open('a') as output:
                                subprocess.run([sys.executable,str(ROOT/'tools/diagnose_training_failure.py'),
                                    '--release',str(run/'release'),'--snapshot',str(snapshots[-1]),'--output',str(diagnosis)],
                                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=output,stderr=subprocess.STDOUT,timeout=300)
                        update('needs_goal_diagnosis',arm=arm,issue=issue,diagnosis=str(diagnosis),
                               reason='Bounded replay exhausted; active Goal must diagnose/validate a code repair before resuming')
                        return
                    verify_contract(run)
                    recovery=recover(run,arm,issue,len(history)+1)
                    state['recoveries'].append(recovery);update('restarting',arm=arm,recovery=recovery)
                    child=launch();attached=process_identity(child.pid);failure_seen=None
                    assert attached is not None
                    update('monitoring',controller_pid=child.pid)
                else:
                    update('monitoring',controller_pid=attached['pid'],arm=arm,
                           completed_step=max((r['step'] for r in rows if r['event']=='train'),default=4))
            elif alive:update('monitoring',controller_pid=attached['pid'],stage=stage)
            elif not alive:
                update('needs_goal_diagnosis',reason='Failure outside formal training; do not silently roll back preflight or evaluation',pipeline=pipeline)
                return
            time.sleep(args.poll_seconds)
    except BaseException as error:
        update('guard_error',error=str(error));raise


if __name__=='__main__':main()
