"""Monitor the immutable gRet 830M baseline experiment; recover without recipe changes."""
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
sys.path.insert(0,str(ROOT))
from tools.watch_swa_training import read,now,write,sha,events,unhealthy,checkpoint_healthy,select_checkpoint,recover,process_identity,same_process,owned_tree,stop_tree
ARMS=('baseline',)


def verify_contract(run):
    release=run/'release';manifest=read(run/'release-manifest.json')
    assert manifest['architecture']=='YOCO-gRet' and manifest['scale']=='830M'
    for name,expected in manifest['files'].items():
        if name.endswith('.py') or name.startswith('configs/') or name=='preflight.json':
            assert sha(release/name)==expected,f'Frozen release changed: {name}'
            if name.startswith(('runtime/','tools/','evaluation/')):assert sha(ROOT/name)==expected,f'Root source changed after freezing: {name}'
    reference=read(release/'configs/SWA400M-reference.json');gret=read(release/'configs/gRet400M-reference.json')
    configs={a:read(release/f'configs/{a}.json') for a in ARMS}
    for arm,cfg in configs.items():
        assert read(ROOT/f'configs/yoco_830m_gret_{arm}_8gpu.json')==cfg,f'{arm}: training setting changed'
        for key in ('architecture','model','seed','tied_embeddings','training','data','execution'):assert cfg[key]==configs['baseline'][key],key
        for key in ('training','data','seed','tied_embeddings'):assert cfg[key]==reference[key]==gret[key],key
        assert cfg['execution']==gret['execution'] and not cfg['execution']['compile']
        assert cfg['model'].get('sliding_window') is None and cfg['model']['n_layers']==24 and cfg['model']['dim']==1536
        assert cfg['model']['hidden_dim']==4608
        assert cfg['model']['n_self_heads']==6 and cfg['model']['n_attn_heads']==cfg['model']['n_attn_kv_heads']==12
        assert cfg['training']['precision']=='fp16_mixed_fp32_params' and cfg['training']['keep_last_checkpoints'] is None
        if arm=='baseline':assert 'memory_fusion' not in cfg
        assert sha(cfg['data']['manifest'])==manifest['data_manifest_sha256']
    contract=dict(architecture='YOCO-gRet',scale='830M',
        config_sha256={a:sha(release/f'configs/{a}.json') for a in ARMS},
        release_manifest_sha256=sha(run/'release-manifest.json'),data_manifest_sha256=manifest['data_manifest_sha256'],
        no_training_setting_changes=True,arm_order=list(ARMS))
    path=run/'guard-contract.json'
    if path.exists():assert read(path)==contract,'Recovery contract changed'
    else:write(path,contract)
    return contract


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
            if (any(a.endswith('/watch_gret_830m_baseline.py') for a in argv)
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
    child=None;attached=None;failure_seen=None;owned=[]
    ownership_path=run/'owned-processes.json'
    previous_ownership=read(ownership_path) if ownership_path.exists() else None
    def update(status,**fields):
        state.update(state=status,updated_at=now(),**fields);write(run/'guard-state.json',state)
    def launch():
        verify_contract(run)
        output=(run/'guard-controller.log').open('a')
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/run_gret_830m_baseline.py'),'--run-dir',str(run)],
            cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        output.close();return process
    try:
        while True:
            if (run/'PIPELINE_SUCCESS').exists():
                update('training_and_evaluation_complete',goal_remaining='inspect rendered reports and publish');return
            pipeline=read(run/'pipeline-status.json') if (run/'pipeline-status.json').exists() else {}
            if attached is None:
                candidate=process_identity(pipeline.get('pid',-1))
                if candidate:
                    command=Path(f"/proc/{candidate['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
                    assert 'run_gret_830m_baseline.py' in command and str(run) in command,'Refuse to attach to an unrelated process'
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
            if previous_ownership and previous_ownership['controller']['pid']==attached['pid']:
                prior=previous_ownership['controller']
                if (not alive or prior['start_ticks']==attached['start_ticks']):
                    if not alive:attached=prior
                    owned=previous_ownership['processes']
                previous_ownership=None
            owned=owned_tree(attached,owned)
            write(ownership_path,{'controller':attached,'processes':owned,'observed_at':now()})
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
                    stop_tree(attached,owned)
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
                    child=launch();attached=process_identity(child.pid);failure_seen=None;owned=[]
                    assert attached is not None
                    update('monitoring',controller_pid=child.pid)
                else:
                    update('monitoring',controller_pid=attached['pid'],arm=arm,
                           completed_step=max((r['step'] for r in rows if r['event']=='train'),default=4))
            elif alive:update('monitoring',controller_pid=attached['pid'],stage=stage)
            elif not alive:
                stop_tree(attached,owned)
                update('needs_goal_diagnosis',reason='Failure outside formal training; do not silently roll back preflight or evaluation',pipeline=pipeline)
                return
            time.sleep(args.poll_seconds)
    except BaseException as error:
        update('guard_error',error=str(error));raise


if __name__=='__main__':main()
