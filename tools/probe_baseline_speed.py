"""Isolated speed/profiler probes around the unchanged published trainer.

Run under torchrun. Patches affect this process only, never a completed release.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import runtime.train as trainer
PROBE_SOURCE_SHA256=trainer.sha256(Path(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    origin=parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--source',type=Path)
    origin.add_argument('--init-model',type=Path)
    parser.add_argument('--updates',type=int,default=60)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--compile-mode',default='default')
    parser.add_argument('--bucket-cap-mb',type=float)
    parser.add_argument('--disable-ddp-optimizer',action='store_true')
    parser.add_argument('--capture-gradient',action='store_true')
    parser.add_argument('--numa0-affinity',action='store_true')
    parser.add_argument('--prewarm-rope',action='store_true')
    parser.add_argument('--mark-cudagraph-steps',action='store_true')
    parser.add_argument('--emulate-precision-casts',action='store_true')
    parser.add_argument('--compile-scope',choices=['model','layers','ffn'],default='model')
    args=parser.parse_args()
    rank=int(os.environ.get('RANK',0))
    if args.numa0_affinity:
        allowed=os.sched_getaffinity(0)
        selected=set(range(rank*5,rank*5+5))&allowed
        if not selected:raise RuntimeError('No CPUs available in the requested NUMA-0 affinity')
        os.sched_setaffinity(0,selected)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    original_compile=torch.compile
    def compile_model(model,**kwargs):
        if args.prewarm_rope and isinstance(model,torch.nn.Module):
            with torch.no_grad():
                dummy=torch.empty(1,1,model.args.dim,device=next(model.parameters()).device)
                model.self_decoder.build_rel_pos(dummy,0)
                model.cross_decoder.build_rel_pos(dummy,0)
        kwargs['mode']=args.compile_mode
        if args.emulate_precision_casts:
            kwargs.pop('mode',None)
            kwargs['options']={**torch._inductor.list_mode_options()[args.compile_mode],
                               'emulate_precision_casts':True}
        if isinstance(model,torch.nn.Module) and args.compile_scope!='model':
            for decoder in (model.self_decoder,model.cross_decoder):
                for layer in decoder.layers:
                    target=layer if args.compile_scope=='layers' else layer.ffn
                    target.forward=original_compile(target.forward,**kwargs)
            return model
        return original_compile(model,**kwargs)
    torch.compile=compile_model
    if args.disable_ddp_optimizer:torch._dynamo.config.optimize_ddp=False
    original_ddp=trainer.DDP
    def ddp(model,**kwargs):
        if args.bucket_cap_mb is not None:kwargs['bucket_cap_mb']=args.bucket_cap_mb
        wrapped=original_ddp(model,**kwargs)
        if args.mark_cudagraph_steps:
            wrapped.register_forward_pre_hook(lambda *unused:torch.compiler.cudagraph_mark_step_begin())
        return wrapped
    trainer.DDP=ddp
    model_holder={}
    original_build=trainer.build_model
    def build(*values,**kwargs):
        model_holder['model']=original_build(*values,**kwargs)
        return model_holder['model']
    trainer.build_model=build
    original_step=torch.optim.AdamW.step
    captured=False
    def optimizer_step(self,*values,**kwargs):
        nonlocal captured
        if rank==0 and args.capture_gradient and not captured:
            gradients={name:p.grad.detach().cpu() for name,p in model_holder['model'].named_parameters() if p.grad is not None}
            torch.save(gradients,args.run_dir/'first_update_gradients.pt')
            captured=True
        return original_step(self,*values,**kwargs)
    torch.optim.AdamW.step=optimizer_step
    profiler=None
    def trace_ready(prof):
        prof.export_chrome_trace(str(args.run_dir/'trace.json.gz'))
        averages=prof.key_averages()
        records=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                      self_device_us=e.self_device_time_total,total_device_us=e.device_time_total) for e in averages]
        records.sort(key=lambda e:e['self_device_us'],reverse=True)
        trainer.atomic_json(args.run_dir/'profile_summary.json',records)
        (args.run_dir/'profile_table.txt').write_text(averages.table(sort_by='self_device_time_total',row_limit=40))
    if rank==0 and args.profile:
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=8,warmup=2,active=3,repeat=1),on_trace_ready=trace_ready,
            record_shapes=True)
        profiler.start()
    original_log=trainer.log
    def log(run,local_rank,event,**fields):
        original_log(run,local_rank,event,**fields)
        if profiler is not None and event=='train':profiler.step()
    trainer.log=log
    sys.argv=['runtime.train','--config',str(args.config),'--run-dir',str(args.run_dir),
              '--benchmark-updates',str(args.updates)]
    sys.argv += ['--resume-from',str(args.source)] if args.source else ['--init-model',str(args.init_model)]
    try:trainer.main()
    finally:
        if profiler is not None:profiler.stop()
    if rank==0:
        trainer.atomic_json(args.run_dir/'probe_options.json',dict(compile_mode=args.compile_mode,
            bucket_cap_mb=args.bucket_cap_mb,disable_ddp_optimizer=args.disable_ddp_optimizer,
            updates=args.updates,profile=args.profile,capture_gradient=args.capture_gradient,
            numa0_affinity=args.numa0_affinity,nccl_proto=os.environ.get('NCCL_PROTO'),
            prewarm_rope=args.prewarm_rope,mark_cudagraph_steps=args.mark_cudagraph_steps,
            emulate_precision_casts=args.emulate_precision_casts,
            compile_scope=args.compile_scope,
            trainer_source_sha256=trainer.sha256(ROOT/'runtime/train.py'),probe_source_sha256=PROBE_SOURCE_SHA256))


if __name__=='__main__':main()
