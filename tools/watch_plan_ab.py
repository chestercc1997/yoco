"""Collect successful A/B results and publish the prepared repository to main.

Run on HPC2 after preparing a clean local commit. The watcher does not submit,
cancel or modify GPU jobs. Failed or incomplete campaigns are never published.
"""
import argparse
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
from deploy_fusion_2gpu import ROOT, BASE, ENV, SKILL

STATE=ROOT/'runs/setup/plan-ab-publish-watcher.json'
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','UNKNOWN'}


def status(stage,**values):
    value=dict(stage=stage,updated_unix=time.time(),pid=os.getpid(),**values)
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATE)
    print(json.dumps(value),flush=True)


def remote(command):
    return subprocess.check_output([sys.executable,str(SKILL/'scripts/remote.py'),command],
        cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,start_new_session=True,timeout=240).decode()


def git(*args):
    env=dict(ENV,GIT_TERMINAL_PROMPT='0')
    return subprocess.check_output(['git','-c','http.proxy=','-c','https.proxy=',*args],cwd=ROOT,env=env,stderr=subprocess.STDOUT,timeout=180).decode().strip()


def collect(record):
    code='''import hashlib,json,tarfile
from pathlib import Path
r=RECORD;base=Path(BASE);release=Path(r['release'])
assert (release/'runs/eval-plan-ab/PIPELINE_SUCCESS').exists(),'No complete evaluation marker'
paths=[]
for arm,folder in r['runs'].items():
    run=Path(folder)
    assert json.loads((run/'full_complete.json').read_text())['step']==40000
    paths += [run/n for n in ['rank0.jsonl','full_complete.json','stage_complete.json','resolved_config.json','heartbeat.json']]
evaldir=release/'runs/eval-plan-ab'
paths += [evaldir/'PIPELINE_SUCCESS',evaldir/'summary.json',release/'submitted.json',release/'release_manifest.json']
paths += sorted((evaldir/'results').glob('*/*/results.json'))+sorted((evaldir/'results').glob('*/*/provenance.json'))
assert len(list((evaldir/'results').glob('*/*/results.json')))==16
paths += [base/'logs'/('plan-ab-4gpu-'+r['job_id']+'.'+ext) for ext in ['out','err']]
archive=base/'runs/setup'/('collected-plan-ab-'+r['job_id']+'.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for p in paths:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
h=hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
print(json.dumps(dict(archive=str(archive),sha256=h.hexdigest())))
'''.replace('RECORD',repr(record)).replace('BASE',repr(BASE))
    info=json.loads(remote('python3 -c '+shlex.quote(code)))
    archive=ROOT/'runs/setup'/Path(info['archive']).name
    subprocess.run(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',
        '-o','ConnectTimeout=30','jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+info['archive'],str(archive)],
        cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,start_new_session=True,check=True,timeout=900)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==info['sha256']
    dest=ROOT/'runs'/('collected-plan-ab-'+record['job_id']);dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            path=Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts and member.isfile(),member.name
            target=dest/path;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('wb') as out:
                import shutil
                shutil.copyfileobj(src,out)
    (dest/'collection.json').write_text(json.dumps(info,indent=2)+'\n')
    return dest


def report(record,dest):
    release=dest/Path(record['release']).relative_to(BASE)
    evaldir=release/'runs/eval-plan-ab'
    summary=json.loads((evaldir/'summary.json').read_text())
    assert len(summary['rows'])==8
    training={};curves={}
    for arm,folder in record['runs'].items():
        rows=[json.loads(x) for x in (dest/Path(folder).relative_to(BASE)/'rank0.jsonl').read_text().splitlines()]
        train=[x for x in rows if x['event']=='train'];val=[x for x in rows if x['event']=='validation']
        assert [x['step'] for x in train]==list(range(1,40001))
        assert [x['step'] for x in val]==list(range(250,40001,250))
        assert train[-1]['tokens']==10485760000
        training[arm]=dict(final_step=40000,tokens=train[-1]['tokens'],validation_loss=val[-1]['loss'],
            training_seconds=sum(x['seconds'] for x in train),peak_memory_bytes=max(x['peak_memory_bytes'] for x in train),
            overflow_events=sum(x['event']=='overflow' for x in rows))
        curves[arm]={x['step']:x['loss'] for x in val}
    for row in summary['rows']:
        task=row['task'];data=[]
        for arm in ('A','B'):
            folder=evaldir/'results'/arm/task
            r=json.loads((folder/'results.json').read_text());p=json.loads((folder/'provenance.json').read_text())
            assert r['model_step']==40000 and not r['smoke_only'] and p['limit'] is None
            assert row[arm]==r['results'][task][row['metric']]
            data.append(p)
        for key in ['dataset_lock','task_yaml_sha256']:assert data[0][key]==data[1][key]
    out=ROOT/'docs/evidence';out.mkdir(exist_ok=True)
    result=dict(job_id=record['job_id'],complete=True,release=record['release'],training=training,downstream=summary,
        collection=json.loads((dest/'collection.json').read_text()),single_seed=True)
    (out/'plan-ab-results.json').write_text(json.dumps(result,indent=2)+'\n')
    csv='step,A_validation_loss,B_validation_loss,B_minus_A\n'+''.join(f'{s},{curves["A"][s]},{curves["B"][s]},{curves["B"][s]-curves["A"][s]}\n' for s in curves['A'])
    (out/'plan-ab-validation.csv').write_text(csv)
    text=f'''# A/B 六层 K/V 聚合完整结果

HPC3 作业 {record['job_id']} 完成。A 为六个全局可学习权重，B 为输入条件权重；两组各四卡、40000 步、10,485,760,000 tokens。旧 baseline 未重训。

| 模型 | 最终验证 loss | 下游八项宏平均 |
| --- | ---: | ---: |
| A | {training['A']['validation_loss']:.6f} | {summary['A_macro']:.6f} |
| B | {training['B']['validation_loss']:.6f} | {summary['B_macro']:.6f} |

| 下游任务 | 指标 | A | B | B−A |
| --- | --- | ---: | ---: | ---: |
'''
    for row in summary['rows']:text+=f"| {row['task']} | {row['metric']} | {row['A']:.6f} | {row['B']:.6f} | {row['delta']:+.6f} |\n"
    text+='\n这是单 seed 结果，不能独自证明普遍收益或梯度机制的因果作用。完整数值见 [JSON](plan-ab-results.json)，验证曲线见 [CSV](plan-ab-validation.csv)。逐步日志、原始评测与 SHA256 回传记录保留在本地 runs/ 下。\n'
    (out/'plan-ab-results.md').write_text(text)
    p=ROOT/'README.md';text=p.read_text();start=text.index('**Status on 2026-11-03:**');end=text.index('\n\n- [A/B design',start)
    text=text[:start]+f"**A/B campaign complete:** job {record['job_id']} finished both 40,000-step runs and all 16 downstream evaluations. See [the full result](docs/evidence/plan-ab-results.md)."+text[end:];p.write_text(text)
    p=ROOT/'docs/README.md';text=p.read_text().replace('当前作业 **794750** 已排队；实现测试及八项离线数据加载通过，完整训练和下游结果尚未产生。','作业 **794750** 已完成两组训练与全部下游测试，见 [完整结果](evidence/plan-ab-results.md)。');p.write_text(text)
    p=ROOT/'docs/PLAN_AB_4GPU.md';text=p.read_text();text+='\n\n完整训练和 16 项下游评测已完成，验收和结果见 [A/B 结果](evidence/plan-ab-results.md)。此前状态段为提交时快照。\n';p.write_text(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepared-head',required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    with STATE.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=json.loads((ROOT/'runs/setup/plan-ab-active-job.json').read_text())
        assert git('branch','--show-current')=='main'
        assert git('rev-parse','HEAD')==args.prepared_head
        assert not git('status','--porcelain','--untracked-files=no'),'Tracked changes require review'
        status('CHECKED' if args.check_only else 'WAITING_FOR_COMPLETE_EVALUATION',job_id=record['job_id'],prepared_head=args.prepared_head)
        if args.check_only:return
        started=time.time()
        while time.time()-started<8*86400:
            try:
                code='''import json,subprocess
job=JOB
rows=subprocess.check_output(['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode']).decode().splitlines()
r=next((x.split('|') for x in rows if x.split('|')[0]==job),[job,'UNKNOWN',''])
print(json.dumps(dict(state=r[1].split()[0],exit_code=r[2])))'''.replace('JOB',repr(record['job_id']))
                state=json.loads(remote('python3 -c '+shlex.quote(code)))
                status('WAITING_FOR_COMPLETE_EVALUATION',job_id=record['job_id'],prepared_head=args.prepared_head,job=state)
            except (subprocess.SubprocessError,ValueError) as exc:
                status('STATUS_RETRY',job_id=record['job_id'],error=str(exc));time.sleep(300);continue
            if state['state']=='COMPLETED' and state['exit_code']=='0:0':break
            if state['state'] not in ACTIVE:
                status('EXPERIMENT_FAILED_NO_PUBLISH',job_id=record['job_id'],job=state);return
            time.sleep(300)
        else:
            status('WATCH_TIMEOUT_NO_PUBLISH',job_id=record['job_id']);return
        assert git('rev-parse','HEAD')==args.prepared_head and not git('status','--porcelain','--untracked-files=no'),'Local changes require review'
        status('COLLECTING',job_id=record['job_id'])
        dest=collect(record);report(record,dest)
        git('diff','--check')
        git('add','README.md','docs/README.md','docs/PLAN_AB_4GPU.md','docs/evidence/plan-ab-results.json','docs/evidence/plan-ab-results.md','docs/evidence/plan-ab-validation.csv')
        git('commit','-m','Report completed 160M learned and conditional KV experiments')
        git('fetch','origin','main')
        git('merge-base','--is-ancestor','origin/main','HEAD')
        head=git('rev-parse','HEAD')
        try:git('push','origin','HEAD:main')
        except subprocess.SubprocessError:
            if git('ls-remote','--heads','origin','main').split()[0]!=head:raise
        assert git('ls-remote','--heads','origin','main').split()[0]==head
        status('PUBLISHED',job_id=record['job_id'],commit=head,repository='https://github.com/chestercc1997/yoco')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('NEEDS_REVIEW_NO_FURTHER_ACTION',error=str(exc));raise
