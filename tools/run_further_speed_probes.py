"""Run independent, bounded probes of execution-only acceleration options."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/yoco_160m_baseline_8gpu_matched.json')
    parser.add_argument('--init-model',type=Path)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/yoco-160m-baseline-8gpu-pilot-20261104T0109Z/checkpoint_00000100.pt')
    parser.add_argument('--cases',help='Comma-separated case names')
    parser.add_argument('--updates',type=int,default=60)
    args=parser.parse_args();run=args.run_root.resolve();run.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text())
    source=args.source
    cases=[('reference',[],{}),
           ('cuda_graphs',['--compile-mode','reduce-overhead'],{}),
           ('bucket64',['--bucket-cap-mb','64'],{}),
           ('no_ddp_compiler',['--disable-ddp-optimizer'],{}),
           ('numa0',['--numa0-affinity'],{}),
           ('nccl_simple',[],{'NCCL_PROTO':'Simple'}),
           ('autotune',['--compile-mode','max-autotune-no-cudagraphs'],{}),
           ('reference_repeat',[],{}),
           ('cuda_graphs_cached',['--compile-mode','reduce-overhead','--prewarm-rope','--mark-cudagraph-steps'],{}),
           ('eager',[],{}),
           ('faithful_compiled',['--emulate-precision-casts'],{}),
           ('faithful_model_only',['--emulate-precision-casts'],{}),
           ('regional',['--emulate-precision-casts','--compile-scope','layers'],{}),
           ('ffn',['--emulate-precision-casts','--compile-scope','ffn'],{}),
           ('legacy_400m',[],{})]
    if args.cases:
        names=set(args.cases.split(','));assert names<={c[0] for c in cases}
        cases=[c for c in cases if c[0] in names]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'data/cache/torchinductor'))
    results=[]
    for name,extra,override in cases:
        folder=run/name;folder.mkdir(exist_ok=True)
        c=copy.deepcopy(config);c['execution']['probe_settings']={'arguments':extra,'environment':override}
        if name=='eager':c['execution'].update(compile=False,compile_loss=False)
        if name.startswith('faithful_'):c['execution']['emulate_precision_casts']=True
        if name=='faithful_model_only':c['execution']['compile_loss']=False
        if name in ('regional','ffn'):
            c['execution'].update(compile_loss=False,emulate_precision_casts=True,
                                  compile_scope='layers' if name=='regional' else 'ffn')
        if name=='legacy_400m':c['execution'].update(compile=False,compile_loss=False,checkpoint_activations=True,fused_adamw=False)
        path=folder/'config.json';path.write_text(json.dumps(c,indent=2)+'\n')
        print(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':name,'state':'running'}),flush=True)
        summary=folder/'benchmark_summary.json'
        if not summary.exists():
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                'tools/probe_baseline_speed.py','--config',str(path),'--run-dir',str(folder),
                '--updates',str(args.updates),'--capture-gradient',*extra]
            command += ['--init-model',str(args.init_model.resolve())] if args.init_model else ['--source',str(source.resolve())]
            started=time.monotonic()
            with (folder/'console.log').open('a') as output:
                process=subprocess.Popen(command,cwd=ROOT,env=dict(env,**override),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=480)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    code=124
            if code:
                result={'case':name,'failed':True,'returncode':code,'wall_seconds':time.monotonic()-started}
                results.append(result);print(json.dumps(result),flush=True);continue
        result=json.loads(summary.read_text());result['case']=name;results.append(result)
        (run/'partial-summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    previous={}
    if (run/'summary.json').exists():previous={r['case']:r for r in json.loads((run/'summary.json').read_text())}
    previous.update({r['case']:r for r in results})
    (run/'summary.json').write_text(json.dumps(list(previous.values()),indent=2)+'\n')


if __name__=='__main__':main()
