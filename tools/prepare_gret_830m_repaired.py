"""Freeze a new validated repair lineage; never edit the failed release."""
import datetime,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_swa_ab import read,atomic_json,digest,now
from tools.watch_swa_training import checkpoint_healthy

def main():
 import torch
 torch.set_num_threads(4)
 proof=read(ROOT/'runs/setup/gret-830m-repair-fixed-buckets-validation.json');assert proof['passed']
 source=Path(proof['source_checkpoint']);oldrun=source.parent.parent
 health=checkpoint_healthy(source);assert health['step']==5000 and health['sha256']==proof['source_checkpoint_sha256']
 cfg=read(ROOT/'configs/yoco_830m_gret_baseline_repaired_8gpu.json')
 original=read(oldrun/'release/configs/baseline.json')
 assert cfg==dict(original,execution=dict(original['execution'],deterministic_algorithms=True,ddp_find_unused_parameters=True))
 # Every training/data/model setting remains identical; deterministic execution is explicit.
 for k in ('training','data','model','seed','architecture','tied_embeddings'):
  assert cfg[k]==read(source.parent/'resolved_config.json')['config'][k]
 prep=ROOT/'runs/setup/gret-830m-fixed-buckets-repair-preparation.json'
 if prep.exists():
  run=Path(read(prep)['run_dir']);assert (run/'release-manifest.json').exists();print(run);return
 run=ROOT/'runs'/('yoco-gret-830m-baseline-repaired-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));run.mkdir();release=run/'release';release.mkdir()
 shutil.copytree(ROOT/'runtime',release/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 (release/'evaluation').mkdir()
 for p in (ROOT/'evaluation').glob('*.py'):shutil.copyfile(p,release/'evaluation'/p.name)
 for f in ['plan_ab_data_locks','data_locks']:shutil.copytree(ROOT/'evaluation'/f,release/'evaluation'/f)
 for f in ['tools','configs','initializations','history']:(release/f).mkdir()
 names=['run_gret_830m_repaired.py','watch_gret_830m_repaired.py','export_gret_830m_repaired.py','gret830_recovery_support.py','check_gret_830m_repair.py','prepare_gret_830m_repaired.py','run_swa_ab.py','watch_swa_training.py','check_400m_restore.py','diagnose_training_failure.py','prepare_downstream_data.py','run_830m_baseline_queue.py']
 for n in names:shutil.copyfile(ROOT/'tools'/n,release/'tools'/n)
 for f in ['SWA400M-reference.json','gRet400M-reference.json']:shutil.copyfile(oldrun/'release/configs'/f,release/'configs'/f)
 for f in ['preflight.json','requirements.lock.txt']:shutil.copyfile(oldrun/'release'/f,release/f)
 shutil.copyfile(oldrun/'release/initializations/baseline.pt',release/'initializations/baseline.pt')
 atomic_json(release/'configs/baseline.json',cfg);atomic_json(release/'configs/model-repair.json',proof)
 historical={}
 for rank in range(8):
  rows=[]
  for folder in [source.parent/'checks/resumed',source.parent]:
   for line in (folder/f'rank{rank}.jsonl').read_text().splitlines():
    try:e=json.loads(line)
    except json.JSONDecodeError:continue
    step=e.get('attempted_update',e.get('step',0)+(1 if e['event'] in ['overflow','nonfinite_forward'] else 0))
    if step<=5000:rows.append(e)
  steps={e['step'] for e in rows if e['event']=='train'};assert steps==set(range(1,5001))
  assert not any(e['event']=='nonfinite_forward' for e in rows)
  path=release/f'history/rank{rank}.jsonl';path.write_text(''.join(json.dumps(e)+'\n' for e in rows));historical[str(rank)]=digest(path)
 files={str(p.relative_to(release)):digest(p) for p in sorted(release.rglob('*')) if p.is_file()}
 previous=read(oldrun/'release-manifest.json')
 atomic_json(run/'recovery-source.json',dict(source_checkpoint=str(source),source_sha256=health['sha256'],source_step=5000,source_run=str(oldrun),health=health,historical_prefix_rank_sha256=historical,model_repair_sha256=digest(release/'configs/model-repair.json'),no_training_setting_changes=True,execution_change=proof['execution_change']))
 atomic_json(run/'release-manifest.json',dict(created_at=now(),architecture='YOCO-gRet',scale='830M',files=files,parameters={'baseline':866737152},data_manifest_sha256=digest(cfg['data']['manifest']),source_SWA_run=previous['source_SWA_run'],recovery_source_sha256=digest(run/'recovery-source.json')))
 from tools.watch_gret_830m_repaired import verify_contract
 contract=verify_contract(run)
 atomic_json(prep,dict(run_dir=str(run),release=str(release),prepared_at=now(),source_checkpoint=str(source),guard_contract=contract))
 print(run,flush=True)
if __name__=='__main__':main()
