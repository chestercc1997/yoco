"""Audit exact checkpoint restoration on every rank before any forward/update."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.distributed as dist


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-repair',type=Path)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.release.resolve()))
    import runtime.train as trainer
    holder={}
    build=trainer.build_model
    def capture_build(*values,**kwargs):
        model=build(*values,**kwargs);holder['model']=model;return model
    trainer.build_model=capture_build
    load_optimizer=torch.optim.AdamW.load_state_dict
    def capture_optimizer(self,state):
        result=load_optimizer(self,state);holder['optimizer']=self;return result
    torch.optim.AdamW.load_state_dict=capture_optimizer
    load_scaler=torch.amp.GradScaler.load_state_dict
    def capture_scaler(self,state):
        result=load_scaler(self,state);holder['scaler']=self;return result
    torch.amp.GradScaler.load_state_dict=capture_scaler
    log=trainer.log
    def audited_log(run,rank,event,**fields):
        log(run,rank,event,**fields)
        if event!='started':return
        saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
        errors=[];counts={'tensors':0}
        def exact(left,right,path):
            if isinstance(left,torch.Tensor):
                counts['tensors']+=1
                if left.dtype!=right.dtype or not torch.equal(left.detach().cpu(),right.detach().cpu()):errors.append(path)
            elif isinstance(left,np.ndarray):
                if not np.array_equal(left,right):errors.append(path)
            elif isinstance(left,dict):
                if left.keys()!=right.keys():errors.append(path+':keys');return
                for key in left:exact(left[key],right[key],path+'/'+str(key))
            elif isinstance(left,(tuple,list)):
                if len(left)!=len(right):errors.append(path+':length');return
                for i,(x,y) in enumerate(zip(left,right)):exact(x,y,path+'/'+str(i))
            elif left!=right:errors.append(path)
        exact(holder['model'].state_dict(),saved['model'],'model')
        exact(holder['optimizer'].state_dict(),saved['optimizer'],'optimizer')
        exact(holder['scaler'].state_dict(),saved['scaler'],'scaler')
        rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),
             'python':random.getstate(),'numpy':np.random.get_state()}
        exact(rng,saved['rngs'][rank],'rng')
        assert fields['step']==saved['step']
        report={'rank':rank,'step':saved['step'],'passed':not errors,
                'checked_tensors':counts['tensors'],'mismatches':errors,
                'no_forward_or_optimizer_updates':True}
        reports=[None]*dist.get_world_size()
        dist.all_gather_object(reports,report)
        if rank==0:trainer.atomic_json(run/'exact-restore.json',
            {'passed':all(r['passed'] for r in reports),'ranks':reports,'world_size':len(reports),
             'source_checkpoint':str(args.checkpoint),'model_optimizer_scaler_rng_bitwise_equal':all(r['passed'] for r in reports)})
        dist.barrier();dist.destroy_process_group()
        if not all(r['passed'] for r in reports):raise RuntimeError('Exact source restoration failed')
        raise SystemExit(0)
    trainer.log=audited_log
    sys.argv=['runtime.train','--config',str(args.config),'--run-dir',str(args.run_dir),
              '--resume-from',str(args.checkpoint)]
    if args.model_repair:sys.argv+=['--model-repair',str(args.model_repair)]
    trainer.main()


if __name__=='__main__':main()
