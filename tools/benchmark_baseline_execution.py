"""Measure eight-GPU execution choices without changing the A/B training recipe."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=ROOT/'runs/diagnostics/baseline-speed-20261104'
    run.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',TORCHINDUCTOR_COMPILE_THREADS='2',
             TORCHINDUCTOR_CACHE_DIR=str(ROOT/'data/cache/torchinductor'))
    source=ROOT/'runs/yoco-160m-baseline-8gpu-pilot-20261104T0109Z/checkpoint_00000100.pt'
    recipe=json.loads((ROOT/'evaluation/ab_baseline_20261104/training_protocol.json').read_text())
    base=json.loads((ROOT/'configs/yoco_160m_baseline_8gpu.json').read_text())
    base['training'].update(checkpoint_every=5000,keep_last_checkpoints=None)
    base['data']['manifest']=str(ROOT/'data/fineweb_edu_10bt/manifest.json')
    for arm in ('A','B'):
        assert base['model']==recipe[arm]['model']
        assert {k:v for k,v in base['training'].items() if k not in ('checkpoint_every','keep_last_checkpoints')} == {k:v for k,v in recipe[arm]['training'].items() if k!='checkpoint_every'}
    cases=[('legacy',True,False,False),('no_recompute_fused',False,True,False),('compiled',False,True,True),('compiled_loss',False,True,True)]
    results=[]
    for name,recompute,fused,compile_model in cases:
        directory=run/name;directory.mkdir(exist_ok=True)
        config=dict(base,execution={'checkpoint_activations':recompute,'fused_adamw':fused,'compile':compile_model,'allow_tf32':True})
        if name=='compiled_loss':config['execution']['compile_loss']=True
        path=directory/'config.json';path.write_text(json.dumps(config,indent=2)+'\n')
        summary=directory/'benchmark_summary.json'
        if not summary.exists():
            print(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':name,'state':'running'}),flush=True)
            with (directory/'console.log').open('a') as output:
                process=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                    '-m','runtime.train','--config',str(path),'--run-dir',str(directory),'--resume-from',str(source),
                    '--benchmark-updates','16'],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
            if process.returncode:
                results.append({'case':name,'failed':True,'returncode':process.returncode})
                print(json.dumps(results[-1]),flush=True)
                continue
        result=json.loads(summary.read_text());result['case']=name;results.append(result)
        print(json.dumps(result),flush=True)
    (run/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
